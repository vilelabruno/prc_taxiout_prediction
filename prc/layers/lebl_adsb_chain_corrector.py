import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adsb_corrector_champion import frame_base, BASE, ADSB, P

CHAIN = ['ob_A_off', 'ob_B_off', 'ob_B_mv_off', 'ob_A_gap', 'ob_B_dwell', 'ob_B_dist', 'ob_A_dist', 'ob_A_n', 'ob_A_gs0', 'ob_B_gs0',
         'ad_start', 'ad_startB', 'ad_mvB', 'ad_st_aob', 'ad_st_lob', 'ad_st_iob', 'ad_late', 'ob_B_pos0_lat', 'ob_B_pos0_lon']
FEATS = BASE + ADSB + ['adsb_cov'] + CHAIN
P3 = dict(P, n_jobs=8)


def add_chain_columns(d):
    d = d.copy()
    d['ad_start'] = d.m - d.ob_A_off; d['ad_startB'] = d.m - d.ob_B_off; d['ad_mvB'] = d.m - d.ob_B_mv_off
    d['ad_st_aob'] = d.ad_start - d.aob; d['ad_st_lob'] = d.ad_start - d.lob; d['ad_st_iob'] = d.ad_start - d.iob
    d['ad_late'] = (d.ad_start > 300).astype(float).where(d.ad_start.notna())
    return d


def obs_mask(d):
    return (d.AOBT_3_flt.notna() & (d.mvt_aobt > 0) & (d.mvt_aobt <= 3600) & (d.pred <= 4000)).to_numpy()


def fit_apply(train, apply, seeds=(3, 11, 29)):
    mt, ma = obs_mask(train), obs_mask(apply)
    tgt = (np.clip(train.y, 0, 3600) - train.pred)[mt]
    c = pd.Series(0.0, index=apply.index)
    if ma.any():
        c[ma] = np.mean([lgb.LGBMRegressor(**dict(P3, random_state=s)).fit(train.loc[mt, FEATS], tgt).predict(apply.loc[ma, FEATS]) for s in seeds], axis=0)
    return c
