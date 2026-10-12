import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from prc.stack.cfg import DEPLOY, HBASE_DIR, KEY, SEQ_OUT, STACK_OOF, SUBMITTING, TARGET, WORK, ARTIFACTS, ensure

ap = argparse.ArgumentParser()
ap.add_argument('--v36', default=str(ARTIFACTS / 'stack_inputs' / 'v36.parquet'))
ap.add_argument('--v20', default=str(ARTIFACTS / 'stack_inputs' / 'v20.parquet'))
ap.add_argument('--v53', default=str(ARTIFACTS / 'stack_inputs' / 'v53.parquet'))
ap.add_argument('--layers', default=str(ARTIFACTS / 'stack_inputs' / 'layers_s4_pool.parquet'))
ap.add_argument('--rk-frame', default=str(DEPLOY / 'rk_frame.parquet'))
ap.add_argument('--stack-dataset', default=str(STACK_OOF / 'stack_dataset.parquet'))
ap.add_argument('--hbase-hold-dir', default=str(HBASE_DIR))
ap.add_argument('--hbase-rk-dir', default=str(WORK / 'hbase' / 'final'))
ap.add_argument('--seq-dir', default=str(SEQ_OUT))
ap.add_argument('--mlp-npz', default=None)
ap.add_argument('--ids', default=str(SUBMITTING))
ap.add_argument('--summer-months', default='6,7')
ap.add_argument('--out', default=str(DEPLOY))
a = ap.parse_args()

OUT = Path(a.out)
ensure(OUT)
ids = pd.read_parquet(a.ids, columns=[KEY])[KEY].astype('int64').to_numpy()
n = len(ids)
SUMMER = [int(x) for x in a.summer_months.split(',')]


def ld(path):
    t = pd.read_parquet(path)
    t[KEY] = t[KEY].astype('int64')
    return t.set_index(KEY)[TARGET].reindex(ids).to_numpy(float)


v20, v36, v53 = ld(a.v20), ld(a.v36), ld(a.v53)
rk = pickle.load(open(a.rk_frame, 'rb'))[1] if a.rk_frame.endswith('.pkl') else pd.read_parquet(a.rk_frame)
rk = rk.copy()
rk[KEY] = rk[KEY].astype('int64')
rkr = rk.set_index(KEY).reindex(ids)
HB = ['raw', 'drop80k', 'drop20k', 'clip3600', 'defects']
NN = ['cgru', 'cnn_blkM', 'cnn_w96s6', 'cnn_w128']
cf = pd.read_parquet(a.layers)
cf[KEY] = cf[KEY].astype('int64')
cf = cf.set_index(KEY).reindex(ids)
M26 = {'dA': v20 - v36, 'dS4': cf.c_s4.fillna(0).to_numpy(float), 'dPool': cf.c_pool.fillna(0).to_numpy(float), 'dHand': v53 - v36}


def hbase_rk(v):
    f = Path(a.hbase_rk_dir) / f'base_{v}_rk.npy'
    if f.exists():
        b = np.load(f)
        assert len(b) == n
        return b - v36
    p = pd.Series(np.nan, index=ids)
    for t in range(1, 13):
        g = Path(a.hbase_rk_dir) / f'base_{v}_T{t}.npy'
        if g.exists():
            p.loc[np.load(Path(a.hbase_rk_dir) / f'ids_T{t}.npy')] = np.load(g)
    return np.nan_to_num(p.to_numpy(float) - v36)


for v in HB:
    M26['H_base_' + v] = hbase_rk(v)
for nm in NN:
    t = pd.read_parquet(f'{a.seq_dir}/nnh3_{nm}.parquet')
    t = t[t.ano == 2026].copy()
    t[KEY] = t[KEY].astype('int64')
    t = t.drop_duplicates(KEY).set_index(KEY)
    M26['nn_' + nm] = t.nn_c.reindex(ids).fillna(0).to_numpy(float)
if a.mlp_npz:
    z = np.load(a.mlp_npz, allow_pickle=True)
    M26['dMLP'] = np.nan_to_num(pd.Series(z['rk'], index=z['ids_rk'].astype('int64')).reindex(ids).to_numpy(float) - v36)
MEM = list(M26)


def ctx(df_cov, sem, m_, jul, hour, stat, mov, dist, nstop, b36_, mem_df):
    cols = ['dS4', 'dPool'] + ['nn_' + x for x in NN]
    nact = (mem_df[cols].abs() > 1e-9).sum(axis=1).to_numpy(float)
    asum = mem_df[cols].abs().sum(axis=1).to_numpy(float)
    hr = np.nan_to_num(hour.astype(float))
    return pd.DataFrame({'c_cov': df_cov.astype(float), 'c_sem': np.nan_to_num(sem.astype(float)), 'c_b1': ((m_ >= 900) & (m_ < 3600)).astype(float), 'c_b2': (m_ >= 3600).astype(float), 'c_jul': jul.astype(float),
                         'c_nact': nact, 'c_asum': asum / 100.0, 'c_stat': np.nan_to_num(stat.astype(float)) / 100.0, 'c_mov': np.nan_to_num(mov.astype(float)) / 100.0, 'c_dist': np.nan_to_num(dist.astype(float)) / 1000.0,
                         'c_hs': np.sin(2 * np.pi * hr / 24), 'c_hc': np.cos(2 * np.pi * hr / 24), 'c_b36': b36_ / 1000.0, 'c_nstop': np.nan_to_num(nstop.astype(float))})


m26 = pd.DataFrame({k: M26[k] for k in MEM})
c26 = ctx(rkr.adsb_cov.to_numpy(), rkr.sem_aobt.to_numpy(), rkr.m.to_numpy(float), np.isin(rkr.mes.to_numpy(), SUMMER), rkr.hour.to_numpy(), rkr.adsb_stat_time.to_numpy(), rkr.adsb_mov_time.to_numpy(), rkr.adsb_dist.to_numpy(), rkr.adsb_n_stop_runs.to_numpy(), v36, m26)
dep26 = pd.concat([pd.DataFrame({KEY: ids, 'apt': rkr.ADEP_mvt.to_numpy(), 'mes': rkr.mes.to_numpy(), 'v36': v36, 'v53': v53, 'sem_aobt': np.nan_to_num(rkr.sem_aobt.to_numpy(float))}), m26, c26], axis=1)
dep26.to_parquet(OUT / 'dep26.parquet', index=False)
print('dep26', dep26.shape)

D = pd.read_parquet(a.stack_dataset)
D[KEY] = D[KEY].astype('int64')
b36 = D.b36.to_numpy(float)
M25 = {k: D[k].to_numpy(float) for k in ['dA', 'dS4', 'dPool', 'dHand']}
for v in HB:
    b = np.load(f'{a.hbase_hold_dir}/base_{v}_hold.npy')
    assert len(b) == len(D)
    M25['H_base_' + v] = np.nan_to_num(b - b36)
for nm in NN:
    t = pd.read_parquet(f'{a.seq_dir}/nnh3_{nm}.parquet')
    t = t[t.ano == 2025].copy()
    t[KEY] = t[KEY].astype('int64')
    t = t.drop_duplicates(KEY)
    M25['nn_' + nm] = np.nan_to_num(D[[KEY]].merge(t[[KEY, 'nn_c']], on=KEY, how='left').nn_c.fillna(0).to_numpy())
if a.mlp_npz:
    M25['dMLP'] = np.nan_to_num(pd.Series(z['ho'], index=z['ids_ho'].astype('int64')).reindex(D[KEY].to_numpy()).to_numpy(float) - b36)
m25 = pd.DataFrame({k: M25[k] for k in MEM})
c25 = ctx(D.ctx_cov.to_numpy(), D.ctx_sem_aobt.to_numpy(), D.ctx_m.to_numpy(float), (D.mes.to_numpy() == 7), D.ctx_hour.to_numpy(), D.x_adsb_stat_time.to_numpy(), D.x_adsb_mov_time.to_numpy(), D.x_adsb_dist.to_numpy(), D.x_adsb_n_stop_runs.to_numpy(), b36, m25)
dep25 = pd.concat([pd.DataFrame({KEY: D[KEY].values, 'y': D.y.values, 'b36': D.b36.values, 'hand': D.hand.values, 'gig': D.gig.values, 'apt': D.apt.values, 'mes': D.mes.values, 'dia_key': D.dia_key.values, 'bloco': D.bloco.values, 'sem_aobt': np.nan_to_num(D.ctx_sem_aobt.to_numpy(float))}), m25, c25], axis=1)
dep25.to_parquet(OUT / 'dep25.parquet', index=False)
print('dep25', dep25.shape)
l25 = dep25.apt != 'LIRF'
l26 = dep26.apt != 'LIRF'
for k in MEM:
    print('%-16s 2025: rms %7.1f  nz %5.1f%%   |  2026: rms %7.1f  nz %5.1f%%' % (k, np.sqrt(np.mean(dep25.loc[l25, k] ** 2)), 100 * (dep25.loc[l25, k].abs() > 1e-9).mean(), np.sqrt(np.mean(dep26.loc[l26, k] ** 2)), 100 * (dep26.loc[l26, k].abs() > 1e-9).mean()))
