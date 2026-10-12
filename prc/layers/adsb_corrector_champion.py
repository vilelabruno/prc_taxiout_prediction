import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adsb'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd, lightgbm as lgb
from prc.paths import FEATURES, OOF, RANKING, SUBMITTING
HOLDOUT = OOF / 'champion_h045_holdout.parquet'
ADSB = ['adsb_first_off', 'adsb_last_off', 'adsb_first_moving_off', 'adsb_last_stat_off', 'adsb_first_is_stat',
        'adsb_stat_time', 'adsb_mov_time', 'adsb_dist', 'adsb_disp', 'adsb_gs_max', 'adsb_gs_mean_mov', 'adsb_n_stop_runs']
P = dict(objective='regression', n_estimators=400, learning_rate=0.04, num_leaves=32, min_child_samples=60, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.8, reg_lambda=3.0, verbose=-1, n_jobs=-1, random_state=3)

APTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
STABLE = ['LEBL', 'LSZH', 'EDDM']


def frame_base(d, car_cats=None):
    d = d.copy()
    d['m'] = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds()
    d['mvt_aobt'] = (d.MVT_TIME_UTC_mvt - d.AOBT_3_flt).dt.total_seconds()
    d['mvt_lobt'] = (d.MVT_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds()
    d['sem_aobt'] = d.AOBT_3_flt.isna().astype(int)
    d['hour'] = d.SCHED_TIME_UTC_mvt.dt.hour; d['dow'] = d.SCHED_TIME_UTC_mvt.dt.dayofweek
    d['apt'] = pd.Categorical(d.ADEP_mvt.astype(str), categories=APTS).codes
    pf = d.FLIGHT_mvt.astype(str).str.extract(r'^([A-Z]{2,3})')[0]
    d['car'] = pd.Categorical(pf, categories=car_cats if car_cats is not None else sorted(pf.dropna().unique())).codes
    return d
BASE = ['m', 'mvt_aobt', 'mvt_lobt', 'sem_aobt', 'hour', 'dow', 'apt', 'car', 'pred']

def build(base_path, out_path, label='final'):
    h = pd.read_parquet(HOLDOUT); idx = h.index
    own = pd.read_parquet(FEATURES / 'adsb_own_2025.parquet'); own['MVT_ID_mvt'] = own.MVT_ID_mvt.astype('int64')
    h['MVT_ID_mvt'] = h.MVT_ID_mvt.astype('int64'); h = h.merge(own, on='MVT_ID_mvt', how='left'); h.index = idx
    h['adsb_cov'] = h.adsb_cov.fillna(0)
    pf = h.FLIGHT_mvt.astype(str).str.extract(r'^([A-Z]{2,3})')[0]; car_cats = sorted(pf.dropna().unique())
    h = frame_base(h, car_cats); y = h.TAXITIME_SEC_mvt.astype(float)
    normal = h.pred <= 4000; feats = BASE + ADSB + ['adsb_cov']
    m = lgb.LGBMRegressor(**P).fit(h.loc[normal, feats], (np.clip(y, 0, 3600) - h.pred)[normal])
    rk = pd.read_parquet(RANKING); rk = rk[rk.PHASE_mvt == 'DEP'].copy()
    rk['MVT_ID_mvt'] = rk.MVT_ID_mvt.astype('int64')
    base = pd.read_parquet(base_path); base['MVT_ID_mvt'] = base.MVT_ID_mvt.astype('int64')
    rk = rk.merge(base.rename(columns={'TAXITIME_SEC_mvt': 'pred'}), on='MVT_ID_mvt', how='left')
    o26 = pd.read_parquet(FEATURES / f'adsb_own_{label}.parquet'); o26['MVT_ID_mvt'] = o26.MVT_ID_mvt.astype('int64')
    rk = rk.merge(o26, on='MVT_ID_mvt', how='left'); rk['adsb_cov'] = rk.adsb_cov.fillna(0)
    rk = frame_base(rk, car_cats)
    ok = rk.ADEP_mvt.isin(STABLE) & (rk.adsb_cov == 1) & (rk.pred <= 4000)
    c = pd.Series(0.0, index=rk.index); c[ok] = m.predict(rk.loc[ok, feats])
    out = base.copy().set_index('MVT_ID_mvt')
    corr = pd.Series(c.to_numpy(), index=rk.MVT_ID_mvt.to_numpy())
    out.loc[corr.index, 'TAXITIME_SEC_mvt'] = (out.loc[corr.index, 'TAXITIME_SEC_mvt'] + corr).clip(lower=0)
    out = out.reset_index()[['MVT_ID_mvt', 'TAXITIME_SEC_mvt']]
    tpl = pd.read_parquet(SUBMITTING); out = tpl[['MVT_ID_mvt']].merge(out, on='MVT_ID_mvt', how='left')
    out['TAXITIME_SEC_mvt'] = out.TAXITIME_SEC_mvt.astype(tpl.TAXITIME_SEC_mvt.dtype)
    assert out.TAXITIME_SEC_mvt.notna().all() and out.MVT_ID_mvt.is_unique and len(out) == len(base)
    out.to_parquet(out_path, index=False)
    cc = c[ok]
    print(f'linhas corrigidas {int(ok.sum())} | correção média {cc.mean():.1f} s, p5/p95 {cc.quantile(.05):.0f}/{cc.quantile(.95):.0f} | por aeroporto: {rk[ok].groupby("ADEP_mvt").size().to_dict()}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True); ap.add_argument('--build', required=True); ap.add_argument('--label', default='final')
    a = ap.parse_args()
    build(a.base, a.build, a.label)
