import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from prc.paths import ARTIFACTS, MODELS, RANKING, WORK

KEY = 'MVT_ID_mvt'
TARGET = 'TAXITIME_SEC_mvt'
HB = ['raw', 'drop80k', 'drop20k', 'clip3600', 'defects']


def series(path):
    d = pd.read_parquet(path)
    d[KEY] = d[KEY].astype('int64')
    return d.set_index(KEY)[TARGET].astype(float)


def members(ids, v36, a_pred, v53, layers, hbase_dir, months):
    M = {'dA': (a_pred.reindex(ids) - v36.reindex(ids)).to_numpy(float),
         'dS4': layers.c_s4.reindex(ids).fillna(0).to_numpy(float),
         'dPool': layers.c_pool.reindex(ids).fillna(0).to_numpy(float),
         'dHand': (v53.reindex(ids) - v36.reindex(ids)).to_numpy(float)}
    for v in HB:
        p = pd.Series(np.nan, index=ids)
        for t in months:
            f = Path(hbase_dir) / f'base_{v}_T{t}.npy'
            if f.exists():
                p.loc[np.load(Path(hbase_dir) / f'ids_T{t}.npy')] = np.load(f)
        M['H_base_' + v] = np.nan_to_num(p.to_numpy(float) - v36.reindex(ids).to_numpy(float))
    return pd.DataFrame(M, index=ids)


def apply(coefs, X, cell, v36):
    top = coefs['top']
    out = np.tile(v36[:, None], (1, len(top)))
    for c in np.unique(cell):
        s = cell == c
        for k, j in enumerate(top):
            e = coefs['coefs'].get((c, int(j)))
            if e is None:
                continue
            out[s, k] = v36[s] + X.loc[s, e['cols']].to_numpy(float) @ e['w']
    return out.mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v36', default=str(ARTIFACTS / 'v36_final.parquet'))
    ap.add_argument('--a', default=str(ARTIFACTS / 'base_a_full_final' / 'base_a_full_final.parquet'))
    ap.add_argument('--v53', default=str(ARTIFACTS / 'stack_inputs' / 'v53.parquet'))
    ap.add_argument('--layers', default=str(ARTIFACTS / 'stack_inputs' / 'layers_s4_pool.parquet'))
    ap.add_argument('--hbase-dir', default=str(WORK / 'hbase' / 'final'))
    ap.add_argument('--coefs', default=str(MODELS / 'stack' / 'perm_coefs_nonets.pkl'))
    ap.add_argument('--months', default='2,6')
    ap.add_argument('--reference', default=str(ARTIFACTS / 'reference' / 'stack_old' / 'dep25.parquet'))
    ap.add_argument('--reference-oof', default=str(ARTIFACTS / 'oof' / 'stack' / 'perm_oof_topk_nonets.npy'))
    ap.add_argument('--cut', type=float, default=1.5)
    ap.add_argument('--out', default=str(ARTIFACTS / 'stack_pred.parquet'))
    a = ap.parse_args()
    months = [int(x) for x in a.months.split(',')]
    v36 = series(a.v36)
    rk = pd.read_parquet(RANKING, columns=[KEY, 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt'])
    rk = rk[rk.PHASE_mvt == 'DEP'].drop_duplicates(KEY)
    fr = pd.DataFrame({'ADEP_mvt': rk.ADEP_mvt.to_numpy(), 'mes': pd.to_datetime(rk.MVT_TIME_UTC_mvt).dt.month.to_numpy(),
                       'sem_aobt': rk.AOBT_3_flt.isna().astype(float).to_numpy()}, index=rk[KEY].astype('int64').to_numpy())
    fr = fr[fr.mes.isin(months)]
    ids = fr.index
    layers = pd.read_parquet(a.layers)
    layers[KEY] = layers[KEY].astype('int64')
    X = members(ids, v36, series(a.a), series(a.v53), layers.set_index(KEY), a.hbase_dir, months)
    cell = np.where(fr.ADEP_mvt.to_numpy() == 'LIRF', np.where(fr.sem_aobt.to_numpy(float) == 1, 'LIRF_sem', 'LIRF_ok'), fr.ADEP_mvt.to_numpy())
    coefs = pickle.load(open(a.coefs, 'rb'))
    pred = apply(coefs, X.reset_index(drop=True), cell, v36.reindex(ids).to_numpy(float))
    v53 = series(a.v53).reindex(ids).to_numpy(float)
    apt = fr.ADEP_mvt.to_numpy()
    ref = pd.read_parquet(a.reference, columns=['apt', 'hand', 'gig'])
    oof = np.load(a.reference_oof)
    ok = ~ref.gig.to_numpy(bool)
    use = apt != 'LIRF'
    for ap_ in np.unique(apt):
        if ap_ == 'LIRF':
            continue
        s25 = (ref.apt.to_numpy() == ap_) & ok
        r25 = float(np.sqrt(np.mean((oof - ref.hand.to_numpy(float))[s25] ** 2)))
        for m_ in months:
            s26 = (apt == ap_) & (fr.mes.to_numpy() == m_)
            if not s26.any():
                continue
            r26 = float(np.sqrt(np.mean((pred - v53)[s26] ** 2)))
            if r26 / r25 > a.cut:
                use[s26] = False
                print('corta', ap_, m_, round(r26 / r25, 2))
    pred = np.where(use, pred, v53)
    out = v36.copy()
    out.loc[ids] = pred
    out.rename(TARGET).reset_index().to_parquet(a.out, index=False)
    print(len(ids), float(np.sqrt(np.mean((pred - v36.reindex(ids).to_numpy(float)) ** 2))), int((~use).sum()))


if __name__ == '__main__':
    main()
