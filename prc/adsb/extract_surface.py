import glob, gzip, json, subprocess, sys, tarfile
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prc.paths import DATA_ROOT, SURFACE_DIR
RAW_DIR = DATA_ROOT / 'external' / 'adsblol' / 'raw'
AIRPORTS = {"EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4706, -0.4619), "EHAM": (52.3086, 4.7639),
            "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626), "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389),
            "LTFM": (41.2753, 28.7519), "LSZH": (47.4647, 8.5492)}
DLAT, DLON = 0.06, 0.09


def airport_of(lat, lon):
    for a, (la, lo) in AIRPORTS.items():
        if abs(lat - la) <= DLAT and abs(lon - lo) <= DLON:
            return a
    return None


def main(date, raw_dir=None, out_dir=None):
    raw_dir = raw_dir or RAW_DIR
    out_dir = out_dir or SURFACE_DIR
    parts = sorted(glob.glob(str(Path(raw_dir) / date / "*.tar*")))
    proc = subprocess.Popen(["cat"] + parts, stdout=subprocess.PIPE)
    tf = tarfile.open(fileobj=proc.stdout, mode="r|")
    rows, n = [], 0
    for m in tf:
        if not (m.isfile() and "/traces/" in m.name and m.name.endswith(".json")):
            continue
        b = tf.extractfile(m).read()
        try:
            b = gzip.decompress(b)
        except Exception:
            pass
        try:
            d = json.loads(b)
        except Exception:
            continue
        tr = d.get("trace") or []
        base = d.get("timestamp", 0.0)
        call = None
        for pt in tr:
            if len(pt) > 8 and isinstance(pt[8], dict) and pt[8].get("flight"):
                call = pt[8]["flight"].strip()
            alt = pt[3]
            if alt == "ground":
                alt = 0
            elif not isinstance(alt, (int, float)) or alt > 1500:
                continue
            ap = airport_of(pt[1], pt[2])
            if ap:
                rows.append((d.get("icao"), call, base + pt[0], pt[1], pt[2], alt, pt[4] if len(pt) > 4 else None, ap))
        n += 1
    proc.kill()
    df = pd.DataFrame(rows, columns=["icao24", "callsign", "t", "lat", "lon", "alt", "gs", "apt"])
    out = Path(out_dir) / f"{date}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(date, "traces", n, "pontos de superfície", len(df), "->", out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('dates', nargs='+')
    ap.add_argument('--raw', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    for d in a.dates:
        main(d, a.raw, a.out)
