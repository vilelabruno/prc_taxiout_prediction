import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adsb_corrector_champion import frame_base, BASE, ADSB, P, APTS, STABLE

P2 = dict(P, n_jobs=8)


def add_blocks(d):
    day = d.MVT_TIME_UTC_mvt.dt.day
    d['mes'] = d.MVT_TIME_UTC_mvt.dt.month
    d['bloco'] = d.mes * 10 + np.minimum((day - 1) // 5, 5)
    d['dia'] = d.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
    return d


def obs_mask(d):
    ma = d.mvt_aobt
    return (d.AOBT_3_flt.notna() & (ma > 0) & (ma <= 3600) & (d.pred <= 4000)).to_numpy()


def mask_features(d, feats_adsb, cov_col='adsb_cov'):
    d = d.copy()
    if feats_adsb is None:
        return d
    for f in ADSB:
        ok = d.ADEP_mvt.map(lambda a: f in feats_adsb.get(a, []))
        d.loc[~ok.to_numpy(), f] = np.nan
    return d


def oof_correction(d, feats, train_mask, y):
    c = pd.Series(0.0, index=d.index)
    tgt = np.clip(y, 0, 3600) - d.pred
    for mes, g in d.groupby('mes'):
        for b in sorted(g.bloco.unique()):
            te = (d.bloco == b).to_numpy() & train_mask
            tr = (d.mes == mes).to_numpy() & (d.bloco != b).to_numpy() & train_mask
            m = lgb.LGBMRegressor(**P2).fit(d.loc[tr, feats], tgt[tr])
            c[te] = m.predict(d.loc[te, feats])
    return c


def fit_apply(d_train, y, feats, train_mask, d_apply, apply_mask):
    m = lgb.LGBMRegressor(**P2).fit(d_train.loc[train_mask, feats], (np.clip(y, 0, 3600) - d_train.pred)[train_mask])
    c = pd.Series(0.0, index=d_apply.index)
    if apply_mask.any():
        c[apply_mask] = m.predict(d_apply.loc[apply_mask, feats])
    return c
