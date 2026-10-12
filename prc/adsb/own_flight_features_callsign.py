import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from own_flight_features import load_day, summarize, EP, WIN_BEFORE, WIN_AFTER, TRAINING_ALL
from prc.paths import FEATURES, RANKING


def process(args):
    day, df = args
    s = load_day(day)
    out = []
    if s is None:
        return out
    cs = s.dropna(subset=["callsign"]).groupby(["apt", "callsign"]).icao24.unique().to_dict()
    by_ic = {k: v for k, v in s.groupby("icao24")}
    for r in df.itertuples():
        ics = cs.get((r.ADEP_mvt, r.key))
        if ics is None:
            continue
        best, ncand = None, 0
        for ic in ics:
            g = by_ic[ic]
            w = g[(g.t >= r.mvt - WIN_BEFORE) & (g.t <= r.mvt + WIN_AFTER) & (g.apt == r.ADEP_mvt)]
            if len(w) >= 2:
                ncand += 1
                if best is None or len(w) > len(best):
                    best = w
        if best is not None:
            o = summarize(best, r.mvt)
            o["adsb_ncand"] = ncand
            o["cs_src"] = r.cs_src
            o["MVT_ID_mvt"] = r.MVT_ID_mvt
            out.append(o)
    return out


def run(label, df, out_path=None, nproc=6):
    df = df.copy()
    df["mvt"] = (df.MVT_TIME_UTC_mvt - EP).dt.total_seconds()
    c = df.CALLSIGN_flt.astype("string").str.strip().replace("", pd.NA)
    f = df.FLIGHT_mvt.astype("string").str.strip()
    df["key"] = c.fillna(f)
    df["cs_src"] = c.notna().astype(int)
    df["day"] = df.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    tasks = [(d, g[["MVT_ID_mvt", "ADEP_mvt", "key", "cs_src", "mvt"]]) for d, g in df.groupby("day")]
    rows = []
    with Pool(nproc) as p:
        for k, res in enumerate(p.imap_unordered(process, tasks)):
            rows += res
            if k % 10 == 0:
                print(label, k, "/", len(tasks), flush=True)
    feats = pd.DataFrame(rows)
    ids = df[["MVT_ID_mvt", "cs_src"]].drop_duplicates("MVT_ID_mvt")
    feats = ids.drop(columns="cs_src").merge(feats, on="MVT_ID_mvt", how="left")
    feats["adsb_cov"] = feats.adsb_cov.fillna(0)
    feats["adsb_ncand"] = feats.adsb_ncand.fillna(0)
    feats["cs_src"] = feats.MVT_ID_mvt.map(ids.set_index("MVT_ID_mvt").cs_src)
    out_path = Path(out_path) if out_path else FEATURES / f"adsb_own_cs_{label}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    print(label, "linhas", len(feats), "| cobertas", int(feats.adsb_cov.sum()), f"({feats.adsb_cov.mean()*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    from own_flight_features import read_rank, rank_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", default="all", choices=["all", "2025", "final"])
    rank_args(ap); ap.add_argument("--nproc", type=int, default=6)
    a = ap.parse_args()
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "MVT_TIME_UTC_mvt"]
    if a.which in ("all", "2025"):
        tr = pd.read_parquet(TRAINING_ALL, columns=cols)
        tr = tr[(tr.PHASE_mvt == "DEP") & tr.MVT_TIME_UTC_mvt.dt.month.isin([1, 7])]
        run("2025", tr, nproc=a.nproc)
    if a.which in ("all", "final"):
        run(a.label, read_rank(a.rank or RANKING, cols, "DEP", a.months), a.out, a.nproc)
