import numpy as np
import pandas as pd
from prc.espera_rot.common import HIST_K, HKEYS, HSUF, KEY, RAW_COLS, ROT, ROT_ORIGIN, TRAINING_ALL, epoch


def _id(raw):
    return raw[KEY].astype('int64').to_numpy()


def sl_features(dep, arr):
    d2 = pd.DataFrame({KEY: _id(dep), 'apt': dep.ADEP_mvt.to_numpy(), 'STAND_mvt': dep.STAND_mvt.to_numpy(), 'T': epoch(dep.MVT_TIME_UTC_mvt).to_numpy(), 'SCH': epoch(dep.SCHED_TIME_UTC_mvt).to_numpy()})
    a3 = pd.DataFrame({'apt': arr.ADES_mvt.to_numpy(), 'STAND_mvt': arr.STAND_mvt.to_numpy(), 's_block': epoch(arr.BLOCK_TIME_UTC_mvt).to_numpy(), 's_sched': epoch(arr.SCHED_TIME_UTC_mvt).to_numpy()})
    a3 = a3.dropna(subset=['s_block']).sort_values('s_block')
    d2['ref'] = d2['T'] - 600
    d2 = d2[d2.ref.notna()].sort_values('ref')
    sl = pd.merge_asof(d2, a3, left_on='ref', right_on='s_block', by=['apt', 'STAND_mvt'], direction='backward', tolerance=16 * 3600)
    sl['sl_ub'] = sl['T'] - sl.s_block
    sl['sl_gap_sched'] = sl.SCH - sl.s_block
    sl['sl_delay'] = sl.s_block - sl.s_sched
    return sl[[KEY, 'sl_ub', 'sl_gap_sched', 'sl_delay']]


def _carrier(raw):
    s = raw.FLIGHT_mvt.astype(str)
    c3 = s.str[:3]
    return c3.where(~c3.str[2].str.isdigit().fillna(False), s.str[:2]).to_numpy()


def rot_features(dep, arr):
    dep = dep.reset_index(drop=True)
    arr = arr.reset_index(drop=True)
    d = pd.DataFrame({KEY: _id(dep), 'ap': dep.ADEP_mvt.astype(str).to_numpy(), 'stand': dep.STAND_mvt.astype(str).to_numpy(), 'S': epoch(dep.SCHED_TIME_UTC_mvt, ROT_ORIGIN).to_numpy(), 'car': _carrier(dep), 'type': dep.AIRCRAFT_TYPE_mvt.astype(str).to_numpy()})
    a = pd.DataFrame({'ap': arr.ADES_mvt.astype(str).to_numpy(), 'stand': arr.STAND_mvt.astype(str).to_numpy(), 'B': epoch(arr.BLOCK_TIME_UTC_mvt, ROT_ORIGIN).to_numpy(), 'S': epoch(arr.SCHED_TIME_UTC_mvt, ROT_ORIGIN).to_numpy(), 'mvt': epoch(arr.MVT_TIME_UTC_mvt, ROT_ORIGIN).to_numpy(), 'car': _carrier(arr), 'type': arr.AIRCRAFT_TYPE_mvt.astype(str).to_numpy()})
    a = a[np.isfinite(a.B)].reset_index(drop=True)
    a['taxi_in'] = a.B - a.mvt
    left = d[[KEY, 'ap', 'stand', 'S', 'car', 'type']].copy()
    left['key'] = left.S + 3600.0
    left = left[left.key.notna()]
    right = a[['ap', 'stand', 'B', 'S', 'mvt', 'car', 'type', 'taxi_in']].rename(columns={'S': 'S_f', 'mvt': 'mvt_f', 'car': 'car_f', 'type': 'type_f'})
    right['key'] = right.B
    m = pd.merge_asof(left.sort_values('key'), right.sort_values('key'), on='key', by=['ap', 'stand'], direction='backward', tolerance=36000.0)
    m = m.sort_values(KEY).reset_index(drop=True)
    f = pd.DataFrame({KEY: m[KEY]})
    f['rot_linked'] = m.B.notna().astype(float)
    same_car = np.where(m.B.notna(), (m.car == m.car_f).astype(float), np.nan)
    same_type = np.where(m.B.notna(), (m.type == m.type_f).astype(float), np.nan)
    f['rot_same_car'] = same_car
    f['rot_gap'] = m.S - m.B
    f['rot_good'] = ((same_car == 1) & (same_type == 1) & (f.rot_gap >= -3600) & (f.rot_gap <= 21600)).astype(float)
    f.loc[m.B.isna(), 'rot_good'] = np.nan
    f['rot_arr_delay'] = m.mvt_f - m.S_f
    f['rot_taxi_in'] = m.taxi_in
    ak = a.groupby(['ap', 'stand']).indices
    dk = d.groupby(['ap', 'stand']).indices
    nprev = np.full(len(d), np.nan)
    for key, idx in dk.items():
        if key not in ak:
            nprev[idx] = 0
            continue
        bs = np.sort(a.B.to_numpy()[ak[key]])
        S = d.S.to_numpy()[idx]
        nprev[idx] = np.searchsorted(bs, S, 'right') - np.searchsorted(bs, S - 21600, 'left')
    npv = pd.DataFrame({KEY: d[KEY].to_numpy(), 'rot_n_prev6h': nprev})
    f = f.merge(npv, on=KEY, how='left')
    return f[[KEY] + ROT]


def core_features(dep, arr_sl, arr_rot):
    dep = dep.reset_index(drop=True)
    T = epoch(dep.MVT_TIME_UTC_mvt)
    aobt = epoch(dep.AOBT_3_flt).where(dep.AOBT_3_flt.notna())
    out = pd.DataFrame({KEY: _id(dep), 'apt_name': dep.ADEP_mvt.to_numpy(), 'ctx_mvt_aobt': (T - aobt).to_numpy(), 'ctx_sem_aobt': dep.AOBT_3_flt.isna().astype(float).to_numpy()})
    return out.merge(sl_features(dep, arr_sl), on=KEY, how='left').merge(rot_features(dep, arr_rot), on=KEY, how='left')


def _keys(d):
    d = d.copy()
    d['hr'] = d.MVT_TIME_UTC_mvt.dt.hour
    d['apt'] = d.ADEP_mvt
    d['wk'] = d.WK_TBL_CAT_flt.fillna('?')
    d['stand'] = d.STAND_mvt.fillna('?')
    d['rwy'] = d.RUNWAY_mvt.fillna('?')
    return d


def hist_frame(x, exclude_months=()):
    h = x[x.PHASE_mvt == 'DEP'].copy()
    h['y'] = (h.MVT_TIME_UTC_mvt - h.BLOCK_TIME_UTC_mvt).dt.total_seconds()
    h = h[(h.y > 60) & (h.y < 5000)]
    h = h[~h.MVT_TIME_UTC_mvt.dt.month.isin(exclude_months)].copy()
    h = _keys(h)
    h['aobt_off'] = (h.AOBT_3_flt - h.BLOCK_TIME_UTC_mvt).dt.total_seconds().clip(-1800, 1800)
    b1 = h.groupby(['apt', 'hr', 'wk']).y.mean().rename('b1').reset_index()
    b2 = h.groupby(['apt', 'hr']).y.mean().rename('b2').reset_index()
    h = h.merge(b1, on=['apt', 'hr', 'wk'], how='left').merge(b2, on=['apt', 'hr'], how='left')
    h['ya'] = h.y.clip(upper=3000) - h.b1.fillna(h.b2)
    return h


def hist_features(h, target):
    t = _keys(target)
    t[KEY] = _id(t)
    out = pd.DataFrame({KEY: t[KEY].to_numpy()})
    for keys, name in HKEYS:
        g = h.groupby(keys).agg(n=('ya', 'size'), m=('ya', 'mean'), sd=('ya', 'std'), tl=('ya', lambda v: (v > 300).mean()), ao=('aobt_off', 'mean'), aos=('aobt_off', 'std')).reset_index()
        g[name + '_eff'] = g.m * g.n / (g.n + HIST_K)
        g[name + '_n'] = g.n
        g[name + '_std'] = g.sd
        g[name + '_tail'] = g.tl
        g[name + '_aoff'] = g.ao
        g[name + '_aostd'] = g.aos
        j = t[[KEY] + keys].merge(g[keys + [name + s for s in HSUF]], on=keys, how='left')
        out = out.merge(j.drop(columns=keys), on=KEY, how='left')
    return out


def arr_rot_pool(arr_rank):
    prev = pd.read_parquet(TRAINING_ALL, columns=RAW_COLS, filters=[('MVT_TIME_UTC_mvt', '>=', pd.Timestamp('2025-12-20', tz='UTC'))])
    prev = prev[prev.PHASE_mvt == 'ARR']
    if len(prev) == 0:
        return arr_rank
    return pd.concat([prev, arr_rank[RAW_COLS]], ignore_index=True)


def rank_features(ranking, ids, hist, months=None):
    dep = ranking[(ranking.PHASE_mvt == 'DEP') & ranking[KEY].astype('int64').isin(ids)]
    if months is not None:
        dep = dep[dep.MVT_TIME_UTC_mvt.dt.month.isin(months)]
    arr = ranking[ranking.PHASE_mvt == 'ARR']
    return core_features(dep, arr, arr_rot_pool(arr)).merge(hist_features(hist, dep), on=KEY, how='left')
