import argparse
import pandas as pd
from prc.paths import ARTIFACTS

KEY = 'MVT_ID_mvt'
TARGET = 'TAXITIME_SEC_mvt'


def read_series(path):
    d = pd.read_parquet(path)
    if 'pred' in d.columns and TARGET not in d.columns:
        d = d.reset_index().rename(columns={'index': KEY, 'pred': TARGET}) if KEY not in d.columns else d.rename(columns={'pred': TARGET})
    d[KEY] = d[KEY].astype('int64')
    return d.set_index(KEY)[TARGET].astype(float)


def v36(a, b, wa=0.5, wb=0.5):
    idx = a.index.intersection(b.index)
    out = wa * a.reindex(idx) + wb * b.reindex(idx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default=str(ARTIFACTS / 'base_a_full_final' / 'base_a_full_final.parquet'))
    ap.add_argument('--b', default=str(ARTIFACTS / 'baseb' / 'base_b_v35.parquet'))
    ap.add_argument('--out', default=str(ARTIFACTS / 'v36_final.parquet'))
    a = ap.parse_args()
    s = v36(read_series(a.a), read_series(a.b))
    out = s.rename(TARGET).reset_index()
    out.to_parquet(a.out, index=False)
    print(len(out))


if __name__ == '__main__':
    main()
