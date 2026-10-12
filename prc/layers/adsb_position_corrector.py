import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adsb'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import numpy as np, pandas as pd, lightgbm as lgb
from prc.paths import FEATURES, OOF, RANKING, SUBMITTING
from own_flight_features import TRAINING_ALL
HOLDOUT = OOF / 'champion_h045_holdout.parquet'
from adsb_corrector_champion import frame_base, BASE, ADSB, P
from adsb_corrector_callsign import add_blocks, obs_mask
import pos_features as pf
from pos_features import POS_FULL, POS_LEAN

P2 = dict(P, n_jobs=12)
SEEDS = (3, 11, 29)
APPROVED = ['EDDF', 'EDDM', 'EHAM', 'LSZH']
CORE = ['ds0', 'dsfm', 'dsl', 'gs0', 'gsfm', 'tfm_off', 'kin']
STAB_ALL = {
    'EDDF': [f for f in ADSB if f not in ('adsb_gs_mean_mov', 'adsb_mov_time')],
    'EDDM': list(ADSB), 'EHAM': list(ADSB), 'LEBL': list(ADSB), 'LSZH': list(ADSB), 'LFPG': list(ADSB), 'LTFM': list(ADSB),
    'EGLL': ['adsb_disp', 'adsb_first_is_stat', 'adsb_gs_max', 'adsb_last_off'],
    'LEMD': [f for f in ADSB if f not in ('adsb_disp', 'adsb_dist', 'adsb_gs_max', 'adsb_gs_mean_mov', 'adsb_last_off', 'adsb_mov_time')],
    'LIRF': [f for f in ADSB if f not in ('adsb_gs_mean_mov', 'adsb_last_stat_off', 'adsb_mov_time', 'adsb_n_stop_runs')],
}


def mask_airport_feats(d, feats, keep):
    d = d.copy()
    for f in feats:
        ok = d.ADEP_mvt.map(lambda a: f in keep.get(a, [])).to_numpy(bool)
        d.loc[~ok, f] = np.nan
    return d


def pos_keep(drift_auc, apts=APPROVED):
    keep = {}
    for ap in apts:
        x = drift_auc[drift_auc.apt == ap]
        keep[ap] = list(CORE) + [f for f in POS_FULL if f not in CORE and not ((x[f] - 0.5).abs().max() > 0.15)]
    return keep


def prep(d, keep):
    return mask_airport_feats(mask_airport_feats(d, ADSB, STAB_ALL), POS_FULL, keep)


def oof_correction(d, feats, mask, y, seeds=SEEDS):
    c = pd.Series(0.0, index=d.index); tgt = np.clip(y, 0, 3600) - d.pred
    for mes, g in d.groupby('mes'):
        for b in sorted(g.bloco.unique()):
            te = (d.bloco == b).to_numpy() & mask; tr = (d.mes == mes).to_numpy() & (d.bloco != b).to_numpy() & mask
            c[te] = np.mean([lgb.LGBMRegressor(**dict(P2, random_state=s)).fit(d.loc[tr, feats], tgt[tr]).predict(d.loc[te, feats]) for s in seeds], axis=0)
    return c


def boot_delta(d, q0, q1, y, B=2000, seed=7):
    df = pd.DataFrame({'d': d.dia.values, 'm': d.mes.values, 'e0': (q0 - y) ** 2, 'e1': (q1 - y) ** 2, 'n': 1})
    g = df.groupby(['m', 'd']).sum().reset_index(); rng = np.random.RandomState(seed)
    mm = [g[g.m == m] for m in g.m.unique()]; res = np.empty(B)
    for b in range(B):
        s0 = s1 = nn = 0.0
        for gg in mm:
            ix = rng.randint(0, len(gg), len(gg)); s0 += gg.e0.values[ix].sum(); s1 += gg.e1.values[ix].sum(); nn += gg.n.values[ix].sum()
        res[b] = np.sqrt(s1 / nn) - np.sqrt(s0 / nn)
    r0, r1 = np.sqrt(df.e0.mean()), np.sqrt(df.e1.mean())
    return dict(rmse0=r0, rmse1=r1, delta=r1 - r0, lo=np.percentile(res, 5), hi=np.percentile(res, 95))


def _arr_tables(pts_dir, label='final'):
    cols = ['MVT_ID_mvt', 'ADES_mvt', 'STAND_mvt']
    a25 = pd.read_parquet(Path(pts_dir) / 'e2_arr_2025.parquet').merge(pd.read_parquet(TRAINING_ALL, columns=cols), on='MVT_ID_mvt')
    a26 = pd.read_parquet(Path(pts_dir) / f'e2_arr_{label}.parquet').merge(pd.read_parquet(RANKING, columns=cols), on='MVT_ID_mvt')
    return pf.stand_tables(a25, a26, 'cons')


def build(out_path, pts_dir, apts, drift_csv, base_path=None, v36_path=None, label='final'):
    keep_all = pos_keep(pd.read_csv(drift_csv).query("`mode` == 'cons'") if 'mode' in pd.read_csv(drift_csv, nrows=1).columns else pd.read_csv(drift_csv), APPROVED)
    keep = {a: keep_all[a] for a in apts}
    assert base_path and v36_path
    t25, t26 = _arr_tables(pts_dir, label)
    h = pd.read_parquet(HOLDOUT); idx = h.index
    t = pd.read_parquet(TRAINING_ALL, columns=['MVT_ID_mvt', 'CALLSIGN_flt'])
    h['MVT_ID_mvt'] = h.MVT_ID_mvt.astype('int64'); t['MVT_ID_mvt'] = t.MVT_ID_mvt.astype('int64')
    h = h.merge(t, on='MVT_ID_mvt', how='left'); h.index = idx
    own = pd.read_parquet(FEATURES / 'adsb_own_cs_2025.parquet'); own['MVT_ID_mvt'] = own.MVT_ID_mvt.astype('int64')
    h = h.merge(own, on='MVT_ID_mvt', how='left'); h.index = idx; h['adsb_cov'] = h.adsb_cov.fillna(0)
    h = pf.add_pos_features(h, pd.read_parquet(Path(pts_dir) / 'e2_pts_2025.parquet'), t25)
    h = frame_base(h); h = add_blocks(h)
    y = h.TAXITIME_SEC_mvt.astype(float); mask = obs_mask(h)
    car_cats = sorted(h.FLIGHT_mvt.astype(str).str.extract(r'^([A-Z]{2,3})')[0].dropna().unique())
    rk = pd.read_parquet(RANKING); rk = rk[rk.PHASE_mvt == 'DEP'].reset_index(drop=True); rk['MVT_ID_mvt'] = rk.MVT_ID_mvt.astype('int64')
    o = pd.read_parquet(FEATURES / f'adsb_own_cs_{label}.parquet'); o['MVT_ID_mvt'] = o.MVT_ID_mvt.astype('int64')
    rk = rk.merge(o, on='MVT_ID_mvt', how='left'); rk['adsb_cov'] = rk.adsb_cov.fillna(0)
    rk = pf.add_pos_features(rk, pd.read_parquet(Path(pts_dir) / f'e2_pts_{label}.parquet'), t26)
    v36 = pd.read_parquet(v36_path); v36['MVT_ID_mvt'] = v36.MVT_ID_mvt.astype('int64')
    rk = rk.merge(v36.rename(columns={'TAXITIME_SEC_mvt': 'pred'}), on='MVT_ID_mvt', how='left'); rk['pred'] = rk.pred.astype(float)
    rk = frame_base(rk, car_cats); rk['mes'] = rk.MVT_TIME_UTC_mvt.dt.month
    el = (obs_mask(rk) & (rk.adsb_cov == 1).to_numpy() & rk.ADEP_mvt.isin(apts).to_numpy() & rk.mes.isin([1, 7]).to_numpy())
    dP, rP = prep(h, keep), prep(rk, keep); feats = BASE + ADSB + ['adsb_cov'] + POS_FULL
    c = pd.Series(0.0, index=rk.index)
    for mes in (1, 7):
        trm = (h.mes == mes).to_numpy() & mask; tgt = (np.clip(y, 0, 3600) - h.pred)[trm]; am = el & (rk.mes == mes).to_numpy()
        if am.any():
            c[am] = np.mean([lgb.LGBMRegressor(**dict(P2, random_state=s)).fit(dP.loc[trm, feats], tgt).predict(rP.loc[am, feats]) for s in SEEDS], axis=0)
    new = pd.Series(np.where(el, np.maximum(rk.pred + c, 0.0), np.nan), index=rk.MVT_ID_mvt.to_numpy())
    tpl = pd.read_parquet(SUBMITTING); fin = pd.read_parquet(base_path)
    assert (fin.MVT_ID_mvt.values == tpl.MVT_ID_mvt.values).all()
    nv = new.reindex(fin.MVT_ID_mvt.astype('int64').to_numpy()).to_numpy()
    out = tpl[['MVT_ID_mvt']].copy(); out['TAXITIME_SEC_mvt'] = np.where(np.isnan(nv), fin.TAXITIME_SEC_mvt.to_numpy(), nv).astype(tpl.TAXITIME_SEC_mvt.dtype)
    assert len(out) == len(tpl) and out.TAXITIME_SEC_mvt.notna().all() and (out.TAXITIME_SEC_mvt >= 0).all() and out.MVT_ID_mvt.is_unique
    out.to_parquet(out_path, index=False)
    print('linhas', len(out), 'alteradas', int((~np.isnan(nv)).sum()), 'por aeroporto', rk[el].groupby('ADEP_mvt').size().to_dict())


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', required=True); ap.add_argument('--pts-dir', default=str(FEATURES)); ap.add_argument('--apts', default=','.join(APPROVED))
    ap.add_argument('--drift-csv', default=str(FEATURES / 'drift_pos_auc.csv')); ap.add_argument('--base', required=True); ap.add_argument('--v36', required=True)
    ap.add_argument('--label', default='final')
    a = ap.parse_args()
    build(a.build, a.pts_dir, a.apts.split(','), a.drift_csv, a.base, a.v36, a.label)
