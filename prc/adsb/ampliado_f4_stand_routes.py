import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import own_flight_features as O

REF_DS, RADIUS, CL = 80.0, 60.0, 150.0
LAT0 = {"EDDF": 50.03, "EDDM": 48.35, "EGLL": 51.47, "EHAM": 52.31, "LEBL": 41.30, "LEMD": 40.47, "LFPG": 49.01, "LIRF": 41.80, "LTFM": 41.28, "LSZH": 47.46}


def xy(lat, lon, apt):
    la0 = LAT0[apt]
    return np.c_[(np.asarray(lon) - 0) * 111320 * np.cos(np.radians(la0)), (np.asarray(lat)) * 110574]


def _proc(args):
    day, df = args
    s = O.load_day(day)
    out = []
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
        t = best.t.to_numpy(); gs = np.nan_to_num(best.gs.to_numpy(float), nan=-1.0); lat = best.lat.to_numpy(); lon = best.lon.to_numpy()
        if gs[0] >= 3 or O.hav(lat[0], lon[0], r.sla, r.slo) > REF_DS:
            continue
        mv = np.flatnonzero(gs >= 2.0)
        if len(mv) == 0 or t[mv[0]] - t[0] > 600:
            continue
        i0 = mv[0]
        for j in range(i0, len(t)):
            out.append((r.MVT_ID_mvt, r.ADEP_mvt, r.STAND_mvt, t[j] - t[i0], lat[j], lon[j], gs[j]))
    return out


def build_refs(rows, out_path, nproc=6):
    tasks = [(d, g) for d, g in rows.groupby('day')]
    res = []
    with Pool(nproc) as p:
        for k, r in enumerate(p.imap_unordered(_proc, tasks)):
            res += r
            if k % 20 == 0:
                print('refs', k, '/', len(tasks), flush=True)
    d = pd.DataFrame(res, columns=['ref', 'apt', 'stand', 'tau', 'lat', 'lon', 'gs'])
    d.to_parquet(out_path, index=False); print('refs pontos', len(d), 'voos', d.ref.nunique())
    return d


class Router:
    def __init__(self, refs, stands):
        self.t = {}; self.stand_nb = {}
        for apt, g in refs.groupby('apt'):
            P = xy(g.lat.to_numpy(), g.lon.to_numpy(), apt)
            self.t[apt] = (cKDTree(P), g.stand.to_numpy().astype(str), g.tau.to_numpy(), g.ref.to_numpy())
        for apt, g in stands.groupby('apt'):
            Pst = xy(g.sla.to_numpy(), g.slo.to_numpy(), apt); tr = cKDTree(Pst); st = g.stand.to_numpy().astype(str)
            nb = {}
            for i, s in enumerate(st):
                nb[s] = set(st[j] for j in tr.query_ball_point(Pst[i], CL))
            self.stand_nb[apt] = nb

    def query(self, apt, stand, lat, lon, own=None):
        if apt not in self.t or not np.isfinite(lat) or not np.isfinite(lon):
            return (np.nan, 0, np.nan, np.nan)
        tree, st, tau, ref = self.t[apt]
        idx = np.asarray(tree.query_ball_point(xy([lat], [lon], apt)[0], RADIUS), dtype=int)
        if len(idx) == 0:
            return (np.nan, 0, np.nan, np.nan)
        if own is not None:
            idx = idx[ref[idx] != own]
        for lvl, keep in ((1, lambda s: s == str(stand)), (2, lambda s: s in self.stand_nb.get(apt, {}).get(str(stand), ())), (3, lambda s: True)):
            m = np.array([keep(s) for s in st[idx]], dtype=bool) if lvl < 3 else np.ones(len(idx), bool)
            if m.any():
                ii = idx[m]
                df = pd.DataFrame({'r': ref[ii], 't': tau[ii]}).groupby('r').t.median()
                if len(df) >= (3 if lvl == 1 else 2) or lvl == 3:
                    return (float(df.median()), int(len(df)), float(df.quantile(.75) - df.quantile(.25)) if len(df) > 1 else np.nan, lvl)
        return (np.nan, 0, np.nan, np.nan)
