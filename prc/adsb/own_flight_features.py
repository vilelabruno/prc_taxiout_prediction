import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prc.paths import DATA_ROOT, FEATURES, RANKING, SURFACE_DIR

TRAINING_ALL = DATA_ROOT / 'processed' / 'training_all_2025.parquet'
SURF = Path(os.environ.get('SURF_DIR') or SURFACE_DIR)
EP = pd.Timestamp("1970-01-01", tz="UTC")
WIN_BEFORE, WIN_AFTER = 5400.0, 60.0


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742000 * np.arcsin(np.sqrt(a))


def load_day(day):
    d0 = pd.Timestamp(day, tz="UTC")
    frames = []
    for off in (-1, 0, 1):
        f = SURF / f"{(d0 + pd.Timedelta(days=off)).strftime('%Y-%m-%d')}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            lo, hi = d0.timestamp() - 10800, d0.timestamp() + 86400 + 600
            frames.append(df[(df.t >= lo) & (df.t <= hi)])
    if not frames:
        return None
    s = pd.concat(frames).sort_values(["icao24", "t"])
    s["callsign"] = s.groupby("icao24").callsign.transform(lambda x: x.ffill().bfill())
    s["callsign"] = s.callsign.astype("string").str.strip()
    return s


def summarize(w, mvt):
    t, gs, lat, lon, alt = w.t.to_numpy(), w.gs.to_numpy(dtype=float), w.lat.to_numpy(), w.lon.to_numpy(), w.alt.to_numpy(dtype=float)
    g = np.nan_to_num(gs, nan=-1.0)
    mov, stat = g >= 2.0, (g >= 0) & (g < 1.0)
    o = {"adsb_cov": 1, "adsb_first_off": mvt - t[0], "adsb_last_off": mvt - t[-1]}
    o["adsb_first_moving_off"] = mvt - t[mov][0] if mov.any() else np.nan
    o["adsb_last_stat_off"] = mvt - t[stat][-1] if stat.any() else np.nan
    o["adsb_first_is_stat"] = float(stat[0])
    dt = np.diff(t)
    o["adsb_stat_time"] = float(np.sum(dt[(stat[:-1] & stat[1:]) & (dt <= 120)])) if len(t) > 1 else 0.0
    o["adsb_mov_time"] = float(np.sum(dt[(mov[:-1] | mov[1:]) & (dt <= 120)])) if len(t) > 1 else 0.0
    o["adsb_dist"] = float(np.sum(hav(lat[:-1], lon[:-1], lat[1:], lon[1:]))) if len(t) > 1 else 0.0
    o["adsb_disp"] = float(hav(lat[0], lon[0], lat[-1], lon[-1]))
    o["adsb_gs_max"] = float(np.nanmax(gs)) if np.isfinite(gs).any() else np.nan
    o["adsb_gs_mean_mov"] = float(np.nanmean(gs[mov])) if mov.any() else np.nan
    o["adsb_n_air"] = int(np.sum(alt > 0))
    o["adsb_n_stop_runs"] = int(np.sum(np.diff(stat.astype(int)) == 1) + (1 if stat[0] else 0))
    o["adsb_gap_max"] = float(dt.max()) if len(dt) else 0.0
    return o


def process(args):
    day, df = args
    s = load_day(day)
    out = []
    if s is None:
        return out
    cs = s.dropna(subset=["callsign"]).groupby(["apt", "callsign"]).icao24.unique().to_dict()
    by_ic = {k: v for k, v in s.groupby("icao24")}
    for r in df.itertuples():
        ics = cs.get((r.ADEP_mvt, r.fl))
        if ics is None:
            continue
        best = None
        for ic in ics:
            g = by_ic[ic]
            w = g[(g.t >= r.mvt - WIN_BEFORE) & (g.t <= r.mvt + WIN_AFTER) & (g.apt == r.ADEP_mvt)]
            if len(w) >= 2 and (best is None or len(w) > len(best)):
                best = w
        if best is not None:
            o = summarize(best, r.mvt)
            o["MVT_ID_mvt"] = r.MVT_ID_mvt
            out.append(o)
    return out


def run(year_label, df, out_path=None, nproc=6):
    df = df.copy()
    df["mvt"] = (df.MVT_TIME_UTC_mvt - EP).dt.total_seconds()
    df["fl"] = df.FLIGHT_mvt.astype("string").str.strip()
    df["day"] = df.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    tasks = [(d, g[["MVT_ID_mvt", "ADEP_mvt", "fl", "mvt"]]) for d, g in df.groupby("day")]
    rows = []
    with Pool(nproc) as p:
        for k, res in enumerate(p.imap_unordered(process, tasks)):
            rows += res
            if k % 10 == 0:
                print(year_label, k, "/", len(tasks), flush=True)
    feats = pd.DataFrame(rows)
    ids = df[["MVT_ID_mvt"]].drop_duplicates()
    feats = ids.merge(feats, on="MVT_ID_mvt", how="left")
    feats["adsb_cov"] = feats.adsb_cov.fillna(0)
    out_path = Path(out_path) if out_path else FEATURES / f"adsb_own_{year_label}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    print(year_label, "linhas", len(feats), "| cobertas", int(feats.adsb_cov.sum()), f"({feats.adsb_cov.mean()*100:.1f}%)")


def read_rank(path, cols, phase="DEP", months=None):
    d = pd.read_parquet(path, columns=cols)
    d = d[d.PHASE_mvt == phase]
    if months:
        mm = [int(m) for m in (months.split(",") if isinstance(months, str) else months)]
        d = d[d.MVT_TIME_UTC_mvt.dt.month.isin(mm)]
    return d


def rank_args(ap, label_default="final"):
    ap.add_argument("--rank", default=None, help="ranking alternativo (ex.: data/final/final_ranking.parquet); ignora o modo original")
    ap.add_argument("--months", default=None, help="meses a manter do --rank, ex. '2,6' (padrao: todos)")
    ap.add_argument("--label", default=label_default, help="rotulo de saida usado com --rank")
    ap.add_argument("--out", default=None, help="caminho de saida usado com --rank (padrao: nome com o rotulo)")
    return ap


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", default="all", choices=["all", "2025", "final"])
    rank_args(ap); ap.add_argument("--nproc", type=int, default=6)
    a = ap.parse_args()
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt"]
    if a.which in ("all", "2025"):
        tr = pd.read_parquet(TRAINING_ALL, columns=cols)
        tr = tr[(tr.PHASE_mvt == "DEP") & tr.MVT_TIME_UTC_mvt.dt.month.isin([1, 7])]
        run("2025", tr, nproc=a.nproc)
    if a.which in ("all", "final"):
        run(a.label, read_rank(a.rank or RANKING, cols, "DEP", a.months), a.out, a.nproc)
