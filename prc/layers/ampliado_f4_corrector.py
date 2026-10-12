import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adsb'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd, lightgbm as lgb
from adsb_corrector_champion import BASE, ADSB, P
from adsb_corrector_callsign import obs_mask
from adsb_position_corrector import STAB_ALL, mask_airport_feats
from pos_features import POS_FULL

RTF = ['rt0', 'rt0_n', 'rt0_iqr', 'rt0_lvl', 'rtfm', 'rtfm_n', 'rtfm_iqr', 'rtfm_lvl', 'tx0', 'txfm', 'tx0_m_pred', 'txfm_m_pred', 'tx0_m_aobt', 'txfm_m_aobt']
LGF = ['lg_ngaps', 'lg_has_prev', 'lg_prev_end_off', 'lg_n', 'lg_first_off', 'lg_dur', 'lg_gs0', 'lg_first_moving_off', 'lg_dist', 'lg_stat_time',
       'lg_mov_time', 'lg_disp', 'lg_n_stop_runs', 'lg_gs_mean_mov', 'lg_first_is_stat']
LRT = ['lrt0', 'lrt0_n', 'lrt0_iqr', 'lrt0_lvl', 'lrtfm', 'lrtfm_n', 'lrtfm_iqr', 'lrtfm_lvl', 'ltx0', 'ltxfm']
LDER = ['ltx0_m_pred', 'ltxfm_m_pred', 'ltx0_m_aobt', 'ltxfm_m_aobt']
NEW_FEATS = RTF + LGF + LRT + LDER
SEEDS = (3, 11, 29)
THR = 0.15


def derived(d):
    d = d.copy()
    d['tx0_m_pred'] = d.tx0 - d.pred; d['txfm_m_pred'] = d.txfm - d.pred; d['tx0_m_aobt'] = d.tx0 - d.mvt_aobt; d['txfm_m_aobt'] = d.txfm - d.mvt_aobt
    d['ltx0_m_pred'] = d.ltx0 - d.pred; d['ltxfm_m_pred'] = d.ltxfm - d.pred; d['ltx0_m_aobt'] = d.ltx0 - d.mvt_aobt; d['ltxfm_m_aobt'] = d.ltxfm - d.mvt_aobt
    return d


def keep_stable(drift, feats=NEW_FEATS, thr=THR):
    keep = {}
    for ap, g in drift.groupby('apt'):
        mx = g.groupby('feat').dev.max()
        keep[ap] = [f for f in feats if f in mx.index and mx[f] <= thr]
    return keep


def prep(d, pos_keep, new_keep):
    d = mask_airport_feats(d, ADSB, STAB_ALL)
    d = mask_airport_feats(d, POS_FULL, pos_keep)
    return mask_airport_feats(d, NEW_FEATS, new_keep)


def feature_list():
    return BASE + ADSB + ['adsb_cov'] + POS_FULL + NEW_FEATS


def oof(d, feats, mask, y, seeds=SEEDS, n_jobs=12, params=None):
    c = np.zeros(len(d)); tgt = np.clip(y, 0, 3600) - d.pred.to_numpy()
    mes = d.mes.to_numpy(); blk = d.bloco.to_numpy(); X = d[feats]
    for m in (1, 7):
        for b in sorted(np.unique(blk[mes == m])):
            te = (blk == b) & mask; tr = (mes == m) & (blk != b) & mask
            c[te] = np.mean([lgb.LGBMRegressor(**dict(P, **(params or {}), n_jobs=n_jobs, random_state=s)).fit(X[tr], tgt[tr]).predict(X[te]) for s in seeds], axis=0)
    return c


def fit_apply(dtr, y, feats, trmask, dap, apmask, seeds=SEEDS, n_jobs=12, params=None):
    tgt = np.clip(y, 0, 3600) - dtr.pred.to_numpy(); c = np.zeros(len(dap))
    if apmask.any():
        c[apmask] = np.mean([lgb.LGBMRegressor(**dict(P, **(params or {}), n_jobs=n_jobs, random_state=s)).fit(dtr.loc[trmask, feats], tgt[trmask]).predict(dap.loc[apmask, feats]) for s in seeds], axis=0)
    return c


def drift_zero_neighbors(ext, ref26, feats, apts, thr=THR):
    from sklearn.metrics import roc_auc_score
    ext = ext.copy(); dropped = {}
    for ap in apts:
        a = ext[(ext.ADEP_mvt == ap) & (ext.adsb_cov == 1)]; b = ref26[ref26.ADEP_mvt == ap]; bad = []
        for f in feats:
            if f in BASE:
                continue
            u, v = a[f].dropna(), b[f].dropna()
            if len(u) < 300 or len(v) < 300:
                continue
            u = u.sample(min(len(u), 20000), random_state=0); v = v.sample(min(len(v), 20000), random_state=0)
            if abs(roc_auc_score(np.r_[np.zeros(len(u)), np.ones(len(v))], np.r_[u.values, v.values]) - 0.5) > thr:
                bad.append(f)
        for f in bad:
            ext.loc[ext.ADEP_mvt == ap, f] = np.nan
        dropped[ap] = bad
    return ext, dropped


def fit_apply_ext(dtr, y, feats, trmask, dext, yext, wext, dap, apmask, seeds=SEEDS, n_jobs=12, params=None):
    tgt = np.clip(y, 0, 3600) - dtr.pred.to_numpy(); c = np.zeros(len(dap))
    te = np.clip(yext, 0, 3600) - dext.pred.to_numpy()
    X = pd.concat([dtr.loc[trmask, feats], dext[feats]]); t = np.r_[tgt[trmask], te]; w = np.r_[np.ones(int(trmask.sum())), np.full(len(te), wext)]
    if apmask.any():
        c[apmask] = np.mean([lgb.LGBMRegressor(**dict(P, **(params or {}), n_jobs=n_jobs, random_state=s)).fit(X, t, sample_weight=w).predict(dap.loc[apmask, feats]) for s in seeds], axis=0)
    return c
