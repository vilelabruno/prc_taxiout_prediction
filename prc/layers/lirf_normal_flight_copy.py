import numpy as np, pandas as pd
WINDOWS_2026 = {1: [11, 12, 1, 2], 7: [4, 5, 6, 7, 8, 9, 10],
                2: [12, 1, 2, 3], 6: [4, 5, 6, 7, 8]}


def label_frame(t: pd.DataFrame) -> pd.DataFrame:
    x = t[(t.PHASE_mvt == 'DEP') & (t.ADEP_mvt == 'LIRF') & t.TAXITIME_SEC_mvt.notna() & t.AOBT_3_flt.notna()].copy()
    x['y'] = x.TAXITIME_SEC_mvt.astype(float)
    x['m'] = (x.MVT_TIME_UTC_mvt - x.SCHED_TIME_UTC_mvt).dt.total_seconds()
    x['lob'] = (x.LOBT_flt - x.SCHED_TIME_UTC_mvt).dt.total_seconds()
    x['mon'] = x.MVT_TIME_UTC_mvt.dt.month
    x['cp'] = ((x.y - x.m).abs() <= 10).astype(int)
    x['fl'] = x.FLIGHT_mvt.astype(str)
    return x[x.lob.notna() & (x.lob <= 3600)]


def flag_set(x: pd.DataFrame, months, thr=0.9, nmin=8, kmin=2, thr_m=0.8, nmin_m=3) -> set:
    xs = x[x.mon.isin(months)]
    s = xs.groupby('fl').cp.agg(['mean', 'count']); ok = set(s[(s['mean'] >= thr) & (s['count'] >= nmin)].index)
    if kmin > 0:
        pm = xs.groupby(['fl', 'mon']).cp.agg(['mean', 'count']); pm = pm[(pm['count'] >= nmin_m) & (pm['mean'] >= thr_m)]
        k = pm.groupby(level=0).size(); ok &= set(k[k >= kmin].index)
    return ok


def apply_layer(rows: pd.DataFrame, base: pd.Series, flags_by_month: dict, w=0.75):
    m = (rows.MVT_TIME_UTC_mvt - rows.SCHED_TIME_UTC_mvt).dt.total_seconds()
    lob = (rows.LOBT_flt - rows.SCHED_TIME_UTC_mvt).dt.total_seconds()
    mon = rows.MVT_TIME_UTC_mvt.dt.month; fl = rows.FLIGHT_mvt.astype(str)
    ch = rows.AOBT_3_flt.notna() & (m > 0) & lob.notna() & (lob <= 3600)
    hit = pd.Series(False, index=rows.index)
    for mth, fs in flags_by_month.items(): hit |= (mon == mth) & fl.isin(fs)
    ch &= hit
    new = base.copy(); new[ch] = (base[ch] + w * (m[ch] - base[ch])).clip(lower=0)
    return new, ch
