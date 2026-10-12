import os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prc.paths import FEATURES, RANKING, SURFACE_DIR
import own_flight_features as O
O.SURF = Path(os.environ.get('SURF_DIR', str(SURFACE_DIR)))
GAP = 1200.0
APTS = ['EDDF', 'EDDM', 'EHAM', 'LSZH', 'LEBL', 'EGLL', 'LEMD']
NAN = np.nan


def bearing(la1, lo1, la2, lo2):
    p = np.pi / 180
    return (np.degrees(np.arctan2(np.sin((lo2 - lo1) * p) * np.cos(la2 * p), np.cos(la1 * p) * np.sin(la2 * p) - np.sin(la1 * p) * np.cos(la2 * p) * np.cos((lo2 - lo1) * p))) + 360) % 360


def leg_feats(T, G, LA, LO, mvt):
    g = np.nan_to_num(G, nan=-1.0); o = {}
    k = int(np.argmin(np.abs(T - mvt)))
    if abs(T[k] - mvt) > 150: k = len(T) - 1
    pe = (LA[k], LO[k]); o['ox_end_gap'] = float(mvt - T[-1])
    de = O.hav(LA, LO, pe[0], pe[1]); o['ox_dend0'] = float(de[0])
    seg = O.hav(LA[:-1], LO[:-1], LA[1:], LO[1:]) if len(T) > 1 else np.zeros(0); cum = np.r_[0.0, np.cumsum(seg)]; tot = cum[-1]
    o['ox_tort'] = float(tot / max(de[0], 30.0)) if tot > 0 else NAN
    dt = np.diff(T) if len(T) > 1 else np.zeros(0); near = de < 500
    stat = (g >= 0) & (g < 2.0)
    if len(dt): o['ox_hold_t'] = float(np.sum(dt[(stat[:-1] & stat[1:] & near[:-1] & near[1:]) & (dt <= 120)]))
    else: o['ox_hold_t'] = 0.0
    o['ox_thr_off'] = float(mvt - T[np.argmax(near)]) if near.any() else NAN
    rem = tot - cum; j = np.flatnonzero(rem >= 1000)
    o['ox_t1000'] = float(mvt - T[j[-1]]) if len(j) else NAN
    if tot > 200:
        for q in range(4):
            m = (cum >= tot * q / 4) & (cum <= tot * (q + 1) / 4 + 1e-9) & (g >= 0)
            o[f'ox_v{q+1}'] = float(np.mean(G[m])) if m.any() else NAN
    else:
        for q in range(4): o[f'ox_v{q+1}'] = NAN
    mid = stat & ~near; i0 = int(np.argmax(~stat)) if (~stat).any() else len(T)
    mid[:i0] = False
    if len(dt):
        o['ox_stop_t'] = float(np.sum(dt[(mid[:-1] & mid[1:]) & (dt <= 120)])); o['ox_stops'] = float(np.sum(np.diff(mid.astype(int)) == 1))
    else: o['ox_stop_t'] = 0.0; o['ox_stops'] = 0.0
    mv = np.flatnonzero(g >= 1.0)
    if len(mv):
        i = mv[0]; f8 = np.flatnonzero(g[i:] >= 8.0); o['ox_t8'] = float(T[i + f8[0]] - T[i]) if len(f8) else NAN
        a = min(i + 1, len(T) - 1); b = np.flatnonzero(T - T[i] >= 60)
        if len(b) and len(T) > 4:
            b0 = b[0]; c = np.flatnonzero(T - T[i] >= 150); c0 = c[0] if len(c) else len(T) - 1
            if b0 > i and c0 > b0 and O.hav(LA[i], LO[i], LA[b0], LO[b0]) > 8 and O.hav(LA[b0], LO[b0], LA[c0], LO[c0]) > 15:
                d = abs((bearing(LA[i], LO[i], LA[b0], LO[b0]) - bearing(LA[b0], LO[b0], LA[c0], LO[c0]) + 180) % 360 - 180); o['ox_pb'] = float(d)
            else: o['ox_pb'] = NAN
        else: o['ox_pb'] = NAN
    else: o['ox_t8'] = NAN; o['ox_pb'] = NAN
    o['ox_gsmax'] = float(np.nanmax(G)) if np.isfinite(G).any() else NAN
    return o, pe


def process(args):
    day, df = args; s = O.load_day(day); out = []
    if s is None: return out
    cs = s.dropna(subset=['callsign']).groupby(['apt', 'callsign']).icao24.unique().to_dict()
    by = {k: (v.t.to_numpy(), v.gs.to_numpy(float), v.lat.to_numpy(), v.lon.to_numpy(), v.apt.to_numpy()) for k, v in s.groupby('icao24')}
    ap = {a: g.sort_values('t') for a, g in s.groupby('apt')}
    cache = {}
    for r in df.itertuples():
        ics = cs.get((r.ADEP_mvt, r.key))
        if ics is None: continue
        best = None; bic = None
        for ic in ics:
            t_, g_, la_, lo_, ap_ = by[ic]; i0, i1 = np.searchsorted(t_, [r.mvt - O.WIN_BEFORE, r.mvt + O.WIN_AFTER], side='left'), None
            i0 = np.searchsorted(t_, r.mvt - O.WIN_BEFORE, 'left'); i1 = np.searchsorted(t_, r.mvt + O.WIN_AFTER, 'right')
            mk = ap_[i0:i1] == r.ADEP_mvt
            if mk.sum() >= 2 and (best is None or mk.sum() > len(best[0])):
                best = (t_[i0:i1][mk], g_[i0:i1][mk], la_[i0:i1][mk], lo_[i0:i1][mk]); bic = ic
        if best is None: continue
        t = best[0]; brk = np.flatnonzero(np.diff(t) > GAP) + 1; s0 = int(brk[-1]) if len(brk) else 0
        T = t[s0:]
        if len(T) < 2: continue
        o, pe = leg_feats(T, best[1][s0:], best[2][s0:], best[3][s0:], r.mvt)
        a = ap[r.ADEP_mvt]
        if r.ADEP_mvt not in cache:
            cache[r.ADEP_mvt] = (a.t.to_numpy(), a.lat.to_numpy(), a.lon.to_numpy(), a.gs.to_numpy(float), pd.factorize(a.icao24)[0], a.icao24.to_numpy())
        at, alat, alon, ags, aic, aico = cache[r.ADEP_mvt]; own = bic
        lo, hi = np.searchsorted(at, [r.mvt - 600, r.mvt])
        la_, lo_, gs_, ic_, t_ = alat[lo:hi], alon[lo:hi], ags[lo:hi], aic[lo:hi], at[lo:hi]
        dd = O.hav(la_, lo_, pe[0], pe[1]); gsn = np.nan_to_num(gs_, nan=99.0)
        okown = aico[lo:hi] != own
        m1 = okown & (dd < 450) & (gsn < 5) & (t_ <= r.mvt - 30)
        o['ox_q_ahead'] = float(len(np.unique(ic_[m1])))
        m2 = okown & (gsn >= 5) & (t_ >= r.mvt - 300)
        o['ox_q_moving'] = float(len(np.unique(ic_[m2])))
        m3 = okown & (dd < 450) & (gsn < 2) & (t_ >= r.mvt - 300) & (t_ <= r.mvt - 30)
        o['ox_q_hold'] = float(len(np.unique(ic_[m3])))
        o['MVT_ID_mvt'] = r.MVT_ID_mvt; out.append(o)
    return out


def run(df, nproc=5, label=''):
    df = df[df.ADEP_mvt.isin(APTS)].copy(); df['mvt'] = (df.MVT_TIME_UTC_mvt - O.EP).dt.total_seconds()
    c = df.CALLSIGN_flt.astype('string').str.strip().replace('', pd.NA); f = df.FLIGHT_mvt.astype('string').str.strip()
    df['key'] = c.fillna(f); df['day'] = df.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
    tasks = [(d, g[['MVT_ID_mvt', 'ADEP_mvt', 'key', 'mvt']]) for d, g in df.groupby('day')]; rows = []
    with Pool(nproc) as p:
        for k, res in enumerate(p.imap_unordered(process, tasks)):
            rows += res
            if k % 10 == 0: print(label, k, '/', len(tasks), flush=True)
    return pd.DataFrame(rows)


def parse_cli(default_nproc=5):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('label'); ap.add_argument('outp', nargs='?', default=None)
    ap.add_argument('--rank', default=None, help='ranking alternativo (ex.: data/final/final_ranking.parquet)')
    ap.add_argument('--months', default=None, help="meses a manter do --rank, ex. '2,6'")
    ap.add_argument('--nproc', type=int, default=default_nproc)
    return ap.parse_args()


if __name__ == '__main__':
    a = parse_cli(); label = a.label; outp = a.outp or FEATURES / f'ox_{label}.parquet'
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt', 'MVT_TIME_UTC_mvt']
    if label.startswith('m') and not a.rank:
        d = pd.read_parquet(O.TRAINING_ALL, columns=cols); d = d[(d.PHASE_mvt == 'DEP') & (d.MVT_TIME_UTC_mvt.dt.month == int(label[1:]))]
    else:
        d = O.read_rank(a.rank or RANKING, cols, 'DEP', a.months)
    res = run(d, nproc=a.nproc, label=label); res['MVT_ID_mvt'] = res.MVT_ID_mvt.astype('int64'); res.to_parquet(outp, index=False); print(label, 'pernas', len(res))
