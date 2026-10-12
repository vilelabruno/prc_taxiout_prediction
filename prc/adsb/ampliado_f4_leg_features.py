import os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import own_flight_features as O
from prc.paths import RANKING
if os.environ.get('SURF_DIR'):
    O.SURF = Path(os.environ['SURF_DIR'])

GAP = float(os.environ.get('LEG_GAP', 1200.0))


def hav(la1, lo1, la2, lo2):
    return O.hav(la1, lo1, la2, lo2)


def leg_summary(t, gs, lat, lon, alt, mvt):
    g = np.nan_to_num(gs, nan=-1.0); mov = g >= 2.0; stat = (g >= 0) & (g < 1.0)
    brk = np.flatnonzero(np.diff(t) > GAP) + 1
    s0 = int(brk[-1]) if len(brk) else 0
    o = {'lg_ngaps': float(len(brk)), 'lg_has_prev': float(s0 > 0)}
    o['lg_prev_end_off'] = float(mvt - t[s0 - 1]) if s0 > 0 else np.nan
    o['lg_prev_n'] = float(s0)
    T, G, LA, LO, MV, ST = t[s0:], gs[s0:], lat[s0:], lon[s0:], mov[s0:], stat[s0:]
    o['lg_n'] = float(len(T)); o['lg_first_off'] = float(mvt - T[0]); o['lg_dur'] = float(T[-1] - T[0])
    o['lg_t0'] = float(T[0]); o['lg_lat0'] = float(LA[0]); o['lg_lon0'] = float(LO[0]); o['lg_gs0'] = float(G[0]) if np.isfinite(G[0]) else np.nan
    if MV.any():
        i = int(np.flatnonzero(MV)[0]); o['lg_first_moving_off'] = float(mvt - T[i]); o['lg_latfm'] = float(LA[i]); o['lg_lonfm'] = float(LO[i]); o['lg_tfm'] = float(T[i])
    else:
        o['lg_first_moving_off'] = np.nan; o['lg_latfm'] = np.nan; o['lg_lonfm'] = np.nan; o['lg_tfm'] = np.nan
    dt = np.diff(T)
    if len(T) > 1:
        o['lg_dist'] = float(np.sum(hav(LA[:-1], LO[:-1], LA[1:], LO[1:])))
        o['lg_stat_time'] = float(np.sum(dt[(ST[:-1] & ST[1:]) & (dt <= 120)])); o['lg_mov_time'] = float(np.sum(dt[(MV[:-1] | MV[1:]) & (dt <= 120)]))
        o['lg_disp'] = float(hav(LA[0], LO[0], LA[-1], LO[-1]))
    else:
        o['lg_dist'] = o['lg_stat_time'] = o['lg_mov_time'] = o['lg_disp'] = 0.0
    o['lg_n_stop_runs'] = float(np.sum(np.diff(ST.astype(int)) == 1) + (1 if ST[0] else 0))
    o['lg_gs_mean_mov'] = float(np.nanmean(G[MV])) if MV.any() else np.nan
    o['lg_first_is_stat'] = float(ST[0])
    return o


def process(args):
    day, df = args
    s = O.load_day(day); out = []
    if s is None:
        return out
    cs = s.dropna(subset=['callsign']).groupby(['apt', 'callsign']).icao24.unique().to_dict()
    by = {k: v for k, v in s.groupby('icao24')}
    for r in df.itertuples():
        ics = cs.get((r.ADEP_mvt, r.key))
        if ics is None:
            continue
        best = None
        for ic in ics:
            g = by[ic]; w = g[(g.t >= r.mvt - O.WIN_BEFORE) & (g.t <= r.mvt + O.WIN_AFTER) & (g.apt == r.ADEP_mvt)]
            if len(w) >= 2 and (best is None or len(w) > len(best)):
                best = w
        if best is None:
            continue
        o = leg_summary(best.t.to_numpy(), best.gs.to_numpy(float), best.lat.to_numpy(), best.lon.to_numpy(), best.alt.to_numpy(float), r.mvt)
        o['MVT_ID_mvt'] = r.MVT_ID_mvt; o['lg_icao'] = best.icao24.iloc[0]
        out.append(o)
    return out


def run(label, df, out_path, nproc=6):
    df = df.copy(); df['mvt'] = (df.MVT_TIME_UTC_mvt - O.EP).dt.total_seconds()
    c = df.CALLSIGN_flt.astype('string').str.strip().replace('', pd.NA); f = df.FLIGHT_mvt.astype('string').str.strip()
    df['key'] = c.fillna(f); df['day'] = df.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
    tasks = [(d, g[['MVT_ID_mvt', 'ADEP_mvt', 'key', 'mvt']]) for d, g in df.groupby('day')]
    rows = []
    with Pool(nproc) as p:
        for k, res in enumerate(p.imap_unordered(process, tasks)):
            rows += res
            if k % 10 == 0:
                print(label, k, '/', len(tasks), flush=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False); print(label, 'pernas', len(rows))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('label'); ap.add_argument('outp')
    ap.add_argument('--rank', default=None, help='ranking alternativo (fase final, ex.: data/final/final_ranking.parquet); ignora o rotulo na leitura')
    ap.add_argument('--months', default=None, help="meses a manter do --rank, ex. '2,6'")
    ap.add_argument('--nproc', type=int, default=6)
    a = ap.parse_args(); label, outp = a.label, a.outp
    cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt', 'MVT_TIME_UTC_mvt']
    if label.startswith('m') and not a.rank:
        d = pd.read_parquet(O.TRAINING_ALL, columns=cols); d = d[(d.PHASE_mvt == 'DEP') & (d.MVT_TIME_UTC_mvt.dt.month == int(label[1:]))]
    elif label == '2025' and not a.rank:
        d = pd.read_parquet(O.TRAINING_ALL, columns=cols); d = d[(d.PHASE_mvt == 'DEP') & d.MVT_TIME_UTC_mvt.dt.month.isin([1, 7])]
    else:
        d = O.read_rank(a.rank or RANKING, cols, 'DEP', a.months)
    run(label, d, outp, a.nproc)
