import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from own_flight_features import load_day, hav, EP, WIN_BEFORE, WIN_AFTER, TRAINING_ALL
from prc.paths import FEATURES, RANKING
GAP, DW = 600.0, 240.0


def chain_feats(w, mvt):
    w = w[w.t <= mvt + 1.0]
    if len(w) < 2:
        return None
    t = w.t.to_numpy(); g = np.nan_to_num(w.gs.to_numpy(dtype=float), nan=-1.0)
    lat, lon = w.lat.to_numpy(), w.lon.to_numpy()
    n = len(t)
    i = n - 1
    while i > 0 and t[i] - t[i - 1] <= GAP:
        i -= 1
    o = {'ob_A_off': mvt - t[i], 'ob_A_n': n - i, 'ob_A_gs0': g[i], 'ob_A_gap': (t[i] - t[i - 1]) if i > 0 else np.nan}
    o['ob_A_dist'] = float(np.sum(hav(lat[i:-1], lon[i:-1], lat[i + 1:], lon[i + 1:]))) if n - i > 1 else 0.0
    j = n - 1; dwell = 0.0; cut = None
    stat_end = None
    k = n - 1
    while k > 0 and t[k] - t[k - 1] <= GAP:
        k -= 1
        if g[k] >= 0 and g[k] < 1.0:
            e = k
            while e > 0 and t[e] - t[e - 1] <= GAP and g[e - 1] >= 0 and g[e - 1] < 1.0:
                e -= 1
            if t[k] - t[e] > DW:
                cut = k; dwell = t[k] - t[e]; break
            k = e
    if cut is None:
        jb = k
    else:
        jb = cut + 1 if cut + 1 < n else cut
    o['ob_B_off'] = mvt - t[jb]; o['ob_B_dwell'] = dwell; o['ob_B_cut'] = float(cut is not None)
    mv = np.where(g[jb:] >= 2.0)[0]
    o['ob_B_mv_off'] = mvt - t[jb + mv[0]] if len(mv) else np.nan
    o['ob_B_gs0'] = g[jb]
    o['ob_B_dist'] = float(np.sum(hav(lat[jb:-1], lon[jb:-1], lat[jb + 1:], lon[jb + 1:]))) if n - jb > 1 else 0.0
    o['ob_B_pos0_lat'] = lat[jb]; o['ob_B_pos0_lon'] = lon[jb]
    o['ob_n_win'] = n
    return o


def process(args):
    day, df = args
    s = load_day(day); out = []
    if s is None:
        return out
    cs = s.dropna(subset=['callsign']).groupby(['apt', 'callsign']).icao24.unique().to_dict()
    by_ic = {k: v for k, v in s.groupby('icao24')}
    for r in df.itertuples():
        ics = cs.get(('LEBL', r.key))
        if ics is None:
            continue
        best = None
        for ic in ics:
            g = by_ic[ic]
            w = g[(g.t >= r.mvt - WIN_BEFORE) & (g.t <= r.mvt + WIN_AFTER) & (g.apt == 'LEBL')]
            if len(w) >= 2 and (best is None or len(w) > len(best)):
                best = w
        if best is not None:
            o = chain_feats(best, r.mvt)
            if o is not None:
                o['MVT_ID_mvt'] = r.MVT_ID_mvt; out.append(o)
    return out


def run(df, nproc=8):
    df = df[(df.PHASE_mvt == 'DEP') & (df.ADEP_mvt == 'LEBL')].copy()
    df['mvt'] = (df.MVT_TIME_UTC_mvt - EP).dt.total_seconds()
    c = df.CALLSIGN_flt.astype('string').str.strip().replace('', pd.NA)
    df['key'] = c.fillna(df.FLIGHT_mvt.astype('string').str.strip())
    df['day'] = df.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
    tasks = [(d, g[['MVT_ID_mvt', 'key', 'mvt']]) for d, g in df.groupby('day')]
    rows = []
    with Pool(nproc) as p:
        for res in p.imap_unordered(process, tasks):
            rows += res
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    from own_flight_features import read_rank
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('which', nargs='?', default='all', choices=['all', '2025', 'final'])
    ap.add_argument('--rank', default=None)
    ap.add_argument('--months', default=None)
    ap.add_argument('--label', default='final'); ap.add_argument('--out', default=None)
    ap.add_argument('--nproc', type=int, default=8); ap.add_argument('--ano', type=int, default=2026)
    a = ap.parse_args()
    cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt', 'MVT_TIME_UTC_mvt']
    FEATURES.mkdir(parents=True, exist_ok=True)
    if a.which in ('all', '2025'):
        tr = pd.read_parquet(TRAINING_ALL, columns=cols)
        tr = tr[tr.MVT_TIME_UTC_mvt.dt.month.isin([1, 7])]
        b = run(tr, a.nproc); b['ano'] = 2025
        b.to_parquet(FEATURES / 'lebl_offblock_2025.parquet', index=False); print('2025', len(b))
    if a.which in ('all', 'final'):
        b = run(read_rank(a.rank or RANKING, cols, 'DEP', a.months), a.nproc); b['ano'] = a.ano
        b.to_parquet(a.out or FEATURES / f'lebl_offblock_{a.label}.parquet', index=False); print('final', len(b))
