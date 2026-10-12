import argparse
import pandas as pd

KEY = 'MVT_ID_mvt'


def delta(new, old, col):
    a = new[[KEY, col]].copy()
    b = old[[KEY, col]].copy()
    a[KEY] = a[KEY].astype('int64')
    b[KEY] = b[KEY].astype('int64')
    a = a.groupby(KEY)[col].sum()
    b = b.groupby(KEY)[col].sum()
    ids = a.index.union(b.index)
    return (a.reindex(ids).fillna(0.0) - b.reindex(ids).fillna(0.0)).rename('delta')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True)
    ap.add_argument('--old', required=True)
    ap.add_argument('--col', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    d = delta(pd.read_parquet(a.new), pd.read_parquet(a.old), a.col)
    d.reset_index().to_parquet(a.out, index=False)
    print(len(d), float((d ** 2).mean() ** 0.5))


if __name__ == '__main__':
    main()
