import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
EP = pd.Timestamp('1970-01-01', tz='UTC')
STATE3 = ['ds_n30', 'ds_mean30', 'ds_bk']
STATE_ALL = ['ds_n30', 'ds_mean30', 'ds_f15_30', 'ds_n60', 'ds_mean60', 'ds_fut_mean30', 'ds_bk', 'ds_bk_cum']
BASE9 = ['m', 'mvt_aobt', 'mvt_lobt', 'sem_aobt', 'hour', 'dow', 'apt', 'car', 'pred']
P = dict(objective='regression', n_estimators=400, learning_rate=0.04, num_leaves=32, min_child_samples=60, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.8, reg_lambda=3.0, verbose=-1, n_jobs=8, random_state=3)


def delay_state_features(df, center=False):
    f = df[['MVT_ID_mvt', 'ADEP_mvt', 'EOBT_1_flt', 'AOBT_3_flt', 'MVT_TIME_UTC_mvt']].copy()
    f['a'] = (f.AOBT_3_flt - EP).dt.total_seconds(); f['e'] = (f.EOBT_1_flt - EP).dt.total_seconds()
    f['mv'] = (f.MVT_TIME_UTC_mvt - EP).dt.total_seconds()
    f['dl'] = f.a - f.e
    if center:
        f['ym'] = f.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m')
        f['e'] = f.e + f.groupby(['ADEP_mvt', 'ym']).dl.transform('median').fillna(0)
        f['dl'] = f.a - f.e
    f['t'] = f.a.fillna(f.e).fillna(f.mv - 900)
    out = []
    for ap, g in f.groupby('ADEP_mvt'):
        g = g.sort_values('a'); ok = g.dropna(subset=['a', 'dl'])
        A = ok.a.to_numpy(); D = ok.dl.to_numpy(); cs = np.r_[0, np.cumsum(D)]; c15 = np.r_[0, np.cumsum(D > 900)]
        t = g.t.to_numpy(); r = {'MVT_ID_mvt': g.MVT_ID_mvt.to_numpy()}
        for w, nm in ((1800, '30'), (3600, '60')):
            lo = np.searchsorted(A, t - w, 'left'); hi = np.searchsorted(A, t, 'left'); n = hi - lo
            r['ds_n' + nm] = n; r['ds_mean' + nm] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
            if nm == '30': r['ds_f15_30'] = np.where(n > 0, (c15[hi] - c15[lo]) / np.maximum(n, 1), np.nan)
        lo = np.searchsorted(A, t, 'right'); hi = np.searchsorted(A, t + 1800, 'right'); n = hi - lo
        r['ds_fut_mean30'] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
        Eok = np.sort(ok.e.to_numpy())
        r['ds_bk'] = np.searchsorted(Eok, t, 'right') - np.searchsorted(A, t, 'right')
        Eall = np.sort(g.e.dropna().to_numpy())
        r['ds_bk_cum'] = np.searchsorted(Eall, t, 'right') - np.searchsorted(A, t, 'right')
        out.append(pd.DataFrame(r))
    return pd.concat(out, ignore_index=True)


def fit_apply(train, apply, feats, mask_train, mask_apply, y_train, seeds=(3, 11, 29), shrink=0.5, params=None, save=None):
    p = dict(P, **(params or {}))
    tgt = (np.clip(y_train, 0, 3600) - train.pred)[mask_train]
    c = np.zeros(len(apply))
    if mask_apply.any():
        ms = [lgb.LGBMRegressor(**dict(p, random_state=s)).fit(train.loc[mask_train, feats], tgt) for s in seeds]
        if save is not None:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            for s, m in zip(seeds, ms):
                m.booster_.save_model(f'{save}_seed{s}.txt')
        c[mask_apply] = np.mean([m.predict(apply.loc[mask_apply, feats]) for m in ms], axis=0)
    return shrink * c


def cold_snow_flag(df, root, year):
    import ltfm_snow_layer as SL
    d = df[['MVT_TIME_UTC_mvt']].copy(); d['_ix'] = np.arange(len(d)); d = d.sort_values('MVT_TIME_UTC_mvt')
    d = pd.merge_asof(d, SL._metar(str(root), year), left_on='MVT_TIME_UTC_mvt', right_on='t', direction='backward',
                      tolerance=pd.Timedelta('90min'))
    flag = ((d.sn_r6 == 1) & (d.tc <= 3.5)).fillna(False).to_numpy()
    out = np.zeros(len(df), bool); out[d._ix.to_numpy()] = flag
    return pd.Series(out, index=df.index)
