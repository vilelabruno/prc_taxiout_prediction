import argparse
import pandas as pd
from prc.paths import ARTIFACTS, RANKING

KEY = 'MVT_ID_mvt'
TARGET = 'TAXITIME_SEC_mvt'


def rows(ranking, months, start, end, gap=3600.0):
    r = pd.read_parquet(ranking, columns=[KEY, 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt'])
    r = r[(r.PHASE_mvt == 'DEP') & (r.ADEP_mvt == 'LIRF')].drop_duplicates(KEY).copy()
    r[KEY] = r[KEY].astype('int64')
    mvt = pd.to_datetime(r.MVT_TIME_UTC_mvt, utc=True)
    sch = pd.to_datetime(r.SCHED_TIME_UTC_mvt, utc=True)
    loc = sch.dt.tz_convert('Europe/Rome')
    hm = loc.dt.hour * 60 + loc.dt.minute
    y = (mvt - sch).dt.total_seconds()
    ok = r.AOBT_3_flt.isna() & (y > gap) & hm.between(start, end) & mvt.dt.month.isin(months)
    return pd.DataFrame({KEY: r[KEY][ok].to_numpy(), 'mes': mvt.dt.month[ok].to_numpy(), 'y': y[ok].to_numpy()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--months', default='2,6')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=240)
    ap.add_argument('--dose', type=float, default=0.5)
    a = ap.parse_args()
    p = pd.read_parquet(a.pred)
    p[KEY] = p[KEY].astype('int64')
    sel = rows(RANKING, [int(x) for x in a.months.split(',')], a.start, a.end)
    p = p.set_index(KEY)
    before = p.loc[sel[KEY], TARGET].to_numpy()
    p.loc[sel[KEY], TARGET] = (1.0 - a.dose) * before + a.dose * sel.y.to_numpy()
    p.reset_index().to_parquet(a.out, index=False)
    sel['antes'] = before
    sel['final'] = (1.0 - a.dose) * before + a.dose * sel.y.to_numpy()
    print(sel.groupby('mes').size().to_dict())
    print(sel.to_string(index=False))


if __name__ == '__main__':
    main()
