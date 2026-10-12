import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import numpy as np, pandas as pd, lightgbm as lgb
from prc.paths import ARTIFACTS, DATA_ROOT, MODELS, RANKING

TRAINING_ALL = DATA_ROOT / 'processed' / 'training_all_2025.parquet'
LO_H, HI_H = 1.0, 14.0
APPLY_HI_S = 4 * 3600
DOSE_MIX = 0.75
SEEDS = range(5)
BASE = ['lm', 'prefc', 'hour', 'dow']
SIB = ['arr_gap', 'arr_late', 'arr_copy', 'arr_same_pref', 'arr_same_type', 'arr_has',
       'nxt_after_mvt', 'nxt_after_sched', 'nxt_after_sched_rel', 'arr_rate', 'arr_n']
DENS = ['na3', 'na6', 'sh6', 'n3']
FEATS = BASE + SIB + DENS + ['te_m_pref', 'te_m_prefmon', 'dst2c', 'te_m_prefdst']


def _pref(s):
    return s.astype(str).str.extract(r'^([A-Z]{2,3})')[0].fillna('NA')


def dep_features(dep):
    L = dep[dep.ADEP_mvt == 'LIRF'].copy()
    L['m'] = (L.MVT_TIME_UTC_mvt - L.SCHED_TIME_UTC_mvt).dt.total_seconds()
    L['noaobt'] = L.AOBT_3_flt.isna()
    L['mon'] = L.SCHED_TIME_UTC_mvt.dt.month
    L['hour'] = L.SCHED_TIME_UTC_mvt.dt.hour
    L['dow'] = L.SCHED_TIME_UTC_mvt.dt.dayofweek
    L['pref'] = _pref(L.FLIGHT_mvt)
    L['lm'] = np.log(L.m.clip(lower=1))
    A = L.sort_values('MVT_TIME_UTC_mvt')
    ts = A.MVT_TIME_UTC_mvt.astype('int64').to_numpy() // 10**6
    na = A.noaobt.to_numpy().astype(int); cs = np.concatenate([[0], np.cumsum(na)])
    for h in (3, 6):
        lo = np.searchsorted(ts, ts - h * 3600, 'left'); hi = np.searchsorted(ts, ts + h * 3600, 'right')
        A[f'na{h}'] = cs[hi] - cs[lo] - na; A[f'n{h}'] = hi - lo - 1
    A['sh6'] = A.na6 / np.maximum(A.n6, 1)
    return A


def sib_feats(dep, arr):
    A = arr[['STAND_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'FLIGHT_mvt', 'AIRCRAFT_TYPE_mvt']] \
        .dropna(subset=['STAND_mvt', 'BLOCK_TIME_UTC_mvt']).copy()
    A.columns = ['st', 'ab', 'as_', 'am', 'af', 'aty']; A = A.sort_values('ab')
    D = dep[['STAND_mvt', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'FLIGHT_mvt', 'AIRCRAFT_TYPE_mvt']].copy()
    D.columns = ['st', 'ds', 'dm', 'df', 'dty']; D['i'] = D.index; D = D.dropna(subset=['st', 'ds', 'dm'])
    M = pd.merge_asof(D.sort_values('ds'), A, left_on='ds', right_on='ab', by='st', direction='backward',
                      tolerance=pd.Timedelta('14h')).set_index('i')
    o = pd.DataFrame(index=dep.index)
    o['arr_gap'] = (M.ds - M.ab).dt.total_seconds()
    o['arr_copy'] = (((M.ab - M.as_).dt.total_seconds().abs() <= 10) & ((M.am - M.as_).dt.total_seconds().abs() > 900)).astype(float)
    o['arr_late'] = (M.am - M.as_).dt.total_seconds()
    o['arr_has'] = M.ab.notna().astype(float)
    o['arr_same_pref'] = (M.af.astype(str).str[:3] == M.df.astype(str).str[:3]).astype(float)
    o['arr_same_type'] = (M['aty'] == M['dty']).astype(float)
    N1 = pd.merge_asof(D.sort_values('dm'), A[['st', 'ab']], left_on='dm', right_on='ab', by='st', direction='forward',
                       tolerance=pd.Timedelta('24h')).set_index('i')
    N2 = pd.merge_asof(D.sort_values('ds'), A[['st', 'ab']], left_on='ds', right_on='ab', by='st', direction='forward',
                       tolerance=pd.Timedelta('24h')).set_index('i')
    o['nxt_after_mvt'] = (N1.ab - N1.dm).dt.total_seconds()
    o['nxt_after_sched'] = (N2.ab - N2.dm).dt.total_seconds()
    o['nxt_after_sched_rel'] = (N2.ab - N2.ds).dt.total_seconds()
    return o


def arr_rate_table(arr, copy_scale=None):
    a = arr.copy(); a['m'] = (a.MVT_TIME_UTC_mvt - a.SCHED_TIME_UTC_mvt).dt.total_seconds()
    a['cp'] = ((a.BLOCK_TIME_UTC_mvt - a.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 10)
    a = a[a.m.abs() > 900]; a['pref'] = _pref(a.FLIGHT_mvt); a['mon'] = a.SCHED_TIME_UTC_mvt.dt.month
    return a.groupby(['pref', 'mon']).cp.agg(['sum', 'count']), a.cp.mean()


def apply_rate(C, tab, glob, k=10, drop_rate=False):
    s = tab['sum'].to_dict(); c = tab['count'].to_dict(); r = []; n = []
    for p, m in zip(C.pref, C.mon):
        ss = s.get((p, m), 0); cc = c.get((p, m), 0); r.append((ss + k * glob) / (cc + k)); n.append(cc)
    C['arr_rate'] = r; C['arr_n'] = n
    return C


def build_cell(dep, arr, tab, glob, lo=LO_H, hi=HI_H):
    f = dep_features(dep).join(sib_feats(dep[dep.ADEP_mvt == 'LIRF'], arr))
    f['SCHED'] = f.SCHED_TIME_UTC_mvt; f['MVT'] = f.MVT_TIME_UTC_mvt
    f['day'] = f.SCHED.dt.dayofyear; f['blk'] = f.day // 5
    f = apply_rate(f, tab, glob)
    f = f[f.noaobt & (f.m >= lo * 3600) & (f.m < hi * 3600)].copy()
    f['dst2'] = f.ADES_mvt.astype(str).str[:2].fillna('NA')
    return f


def te_blockout(M, keys, C, k=10):
    out = np.zeros(len(C)); base = M.e.mean()
    for b in np.unique(C.blk):
        mm = M[M.blk != b]; g = mm.groupby(keys).e.agg(['sum', 'count'])
        te = C.blk.values == b
        idx = pd.MultiIndex.from_frame(C.loc[te, keys]) if len(keys) > 1 else C.loc[te, keys[0]]
        s = np.nan_to_num(g['sum'].reindex(idx).to_numpy()); c = g['count'].reindex(idx).fillna(0).to_numpy()
        out[te] = (s + k * base) / (c + k)
    return out


def te_full(M, keys, Z, k=10):
    base = M.e.mean(); g = M.groupby(keys).e.agg(['sum', 'count'])
    idx = pd.MultiIndex.from_frame(Z[keys]) if len(keys) > 1 else Z[keys[0]]
    s = g['sum'].reindex(idx).fillna(0).to_numpy(); c = g['count'].reindex(idx).fillna(0).to_numpy()
    return (s + k * base) / (c + k)


def clf(seed=1, n_jobs=4):
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.04, num_leaves=6, min_child_samples=15, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=seed, n_jobs=n_jobs)


RANK_COLS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt',
             'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'BLOCK_TIME_UTC_mvt']


def load_inputs(rank_path=None):
    cols = RANK_COLS
    t = pd.read_parquet(TRAINING_ALL, columns=cols + ['TAXITIME_SEC_mvt'])
    rk = pd.read_parquet(rank_path or RANKING, columns=cols)
    rk.loc[rk.PHASE_mvt == 'DEP', 'BLOCK_TIME_UTC_mvt'] = pd.NaT
    return t, rk


def build_train(t):
    dep = t[(t.PHASE_mvt == 'DEP') & t.TAXITIME_SEC_mvt.notna()]
    arr = t[(t.PHASE_mvt == 'ARR') & (t.ADES_mvt == 'LIRF')]
    tab, glob = arr_rate_table(arr)
    C = build_cell(dep, arr, tab, glob)
    for c in ('arr_copy', 'arr_has', 'arr_same_pref', 'arr_same_type'):
        C[c] = C[c].fillna(0.0)
    C['y'] = C.TAXITIME_SEC_mvt; C['eco'] = ((C.y - C.m).abs() <= 120).astype(int)
    L = dep[dep.ADEP_mvt == 'LIRF'].copy()
    L['m'] = (L.MVT_TIME_UTC_mvt - L.SCHED_TIME_UTC_mvt).dt.total_seconds()
    M = L[L.AOBT_3_flt.notna() & (L.m > 1800)].copy()
    M['e'] = ((M.TAXITIME_SEC_mvt - M.m).abs() <= 120).astype(int)
    M['pref'] = _pref(M.FLIGHT_mvt); M['mon'] = M.SCHED_TIME_UTC_mvt.dt.month
    M['blk'] = M.SCHED_TIME_UTC_mvt.dt.dayofyear // 5
    M['dst2'] = M.ADES_mvt.astype(str).str[:2]; M['pm'] = M.pref + '_' + M.mon.astype(str)
    return C, M


def predict_p(C, M, Z, nseeds=len(SEEDS), save_dir=None):
    C = C.copy(); Z = Z.copy()
    C['te_m_pref'] = te_blockout(M, ['pref'], C)
    C['pm'] = C.pref + '_' + C.mon.astype(str)
    C['te_m_prefmon'] = te_blockout(M, ['pm'], C)
    C['te_m_prefdst'] = te_blockout(M, ['pref', 'dst2'], C)
    Z['te_m_pref'] = te_full(M, ['pref'], Z)
    Z['pm'] = Z.pref + '_' + Z.mon.astype(str)
    Z['te_m_prefmon'] = te_full(M, ['pm'], Z)
    Z['te_m_prefdst'] = te_full(M, ['pref', 'dst2'], Z)
    top = C.pref.value_counts().index[:25].tolist(); cats_p = top + ['oth']
    for D_ in (C, Z):
        D_['lm'] = np.log(D_.m.clip(lower=1))
        D_['prefc'] = pd.Categorical(np.where(D_.pref.isin(top), D_.pref, 'oth'), categories=cats_p)
    cats = sorted(set(C.dst2.unique()) | set(Z.dst2.unique()))
    for D_ in (C, Z):
        D_['dst2c'] = pd.Categorical(D_.dst2, categories=cats)
        for c in SIB + DENS: D_[c] = D_[c].astype(float)
    ps = []
    for s in range(nseeds):
        m = clf(s).fit(C[FEATS], C.eco)
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            m.booster_.save_model(str(Path(save_dir) / f'seed{s}.txt'))
        ps.append(m.predict_proba(Z[FEATS])[:, 1])
    Z['p'] = np.mean(ps, 0)
    mu = C[(C.eco == 0) & (C.y <= 3600)].y.mean()
    return Z, mu


def build_layer(pos2, t=None, rk=None, dose_mix=DOSE_MIX, save_dir=None):
    if t is None or rk is None: t, rk = load_inputs()
    C, M = build_train(t)
    dep26 = rk[rk.PHASE_mvt == 'DEP']; arr26 = rk[(rk.PHASE_mvt == 'ARR') & (rk.ADES_mvt == 'LIRF')]
    tab26, g26 = arr_rate_table(arr26)
    Z = build_cell(dep26, arr26, tab26, g26)
    Z, mu = predict_p(C, M, Z, save_dir=save_dir)
    mix = Z.p * Z.m + (1 - Z.p) * mu
    mix = pd.Series(mix.to_numpy(), index=Z.MVT_ID_mvt.astype('int64').to_numpy())
    mrow = pd.Series(Z.m.to_numpy(), index=mix.index)
    out = pos2[['MVT_ID_mvt', 'TAXITIME_SEC_mvt']].copy()
    ids = out.MVT_ID_mvt.astype('int64')
    sel = ids.isin(mix.index).to_numpy() & (mrow.reindex(ids).to_numpy() < APPLY_HI_S)
    v = out.TAXITIME_SEC_mvt.to_numpy(float).copy()
    v[sel] = (1 - dose_mix) * v[sel] + dose_mix * mix.reindex(ids[sel]).to_numpy()
    out['TAXITIME_SEC_mvt'] = np.maximum(v, 0.0)
    return out, dict(mu=mu, n_cell=len(Z), n_changed=int(sel.sum()), p_mean=float(Z.p.mean()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos2', required=True)
    ap.add_argument('--out', default=str(ARTIFACTS / 'layers' / 'sem_match.parquet'))
    ap.add_argument('--rank', default=None)
    a = ap.parse_args()
    pos2 = pd.read_parquet(a.pos2)
    t, rk = load_inputs(a.rank)
    out, info = build_layer(pos2, t, rk, save_dir=MODELS / 'layers' / 'sem_match')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); out.to_parquet(a.out, index=False)
    print(info)
