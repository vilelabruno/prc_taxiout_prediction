import argparse
import numpy as np
import pandas as pd
from prc.paths import ARTIFACTS, SUBMITTING, RANKING
from prc.blend import read_series, KEY, TARGET

FLOOR = 60.0


def month_of(ranking_path):
    r = pd.read_parquet(ranking_path, columns=[KEY, 'MVT_TIME_UTC_mvt']).drop_duplicates(KEY)
    r[KEY] = r[KEY].astype('int64')
    return pd.to_datetime(r.set_index(KEY)['MVT_TIME_UTC_mvt']).dt.month


def compose(stack, v36_new, ref_final, ref_v36, months, janjul=(1, 7), delta=None, overrides=None):
    ids = stack.index
    out = stack.copy()
    mjj = months.reindex(ids).isin(janjul).to_numpy()
    jj = ids[mjj]
    base = ref_final.reindex(jj)
    delta = v36_new.reindex(jj) - ref_v36.reindex(jj) if delta is None else delta.reindex(jj)
    ok = base.notna() & delta.notna()
    val = base.copy()
    changed = ok & (delta.abs() > 1e-9)
    val[changed] = np.maximum(base[changed] + delta[changed], FLOOR)
    val[~ok] = stack.reindex(jj)[~ok]
    out.loc[jj] = val.to_numpy()
    if overrides is not None:
        idx = overrides.index.intersection(out.index)
        out.loc[idx] = overrides.loc[idx].to_numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stack', default=str(ARTIFACTS / 'stack_espera.parquet'))
    ap.add_argument('--v36', default=str(ARTIFACTS / 'v36_final.parquet'))
    ap.add_argument('--ref-final', default=str(ARTIFACTS / 'reference' / 'v85_janjul.parquet'))
    ap.add_argument('--ref-v36', default=str(ARTIFACTS / 'reference' / 'v36_janjul.parquet'))
    ap.add_argument('--jj-new', default=None)
    ap.add_argument('--jj-old', default=None)
    ap.add_argument('--jj-add', nargs='*', default=[])
    ap.add_argument('--jj-set', nargs='*', default=[])
    ap.add_argument('--out', default=str(ARTIFACTS / 'submission_final.parquet'))
    a = ap.parse_args()
    months = month_of(RANKING)
    delta = None if a.jj_new is None else read_series(a.jj_new) - read_series(a.jj_old).reindex(read_series(a.jj_new).index)
    for f in a.jj_add:
        t = pd.read_parquet(f)
        t[KEY] = t[KEY].astype('int64')
        t = t.set_index(KEY)['delta']
        delta = t if delta is None else delta.add(t, fill_value=0.0)
    ov = None
    for f in a.jj_set:
        t = pd.read_parquet(f)
        t[KEY] = t[KEY].astype('int64')
        t = t.set_index(KEY)['value']
        ov = t if ov is None else pd.concat([ov, t])
    out = compose(read_series(a.stack), read_series(a.v36), read_series(a.ref_final), read_series(a.ref_v36), months, delta=delta, overrides=ov)
    tmpl = pd.read_parquet(SUBMITTING, columns=[KEY])
    tmpl[KEY] = tmpl[KEY].astype('int64')
    sub = tmpl.merge(out.rename(TARGET).reset_index(), on=KEY, how='left')
    assert sub[TARGET].notna().all(), int(sub[TARGET].isna().sum())
    assert sub[KEY].is_unique and len(sub) == len(tmpl)
    sub[TARGET] = sub[TARGET].clip(lower=0.0)
    sub.to_parquet(a.out, index=False)
    print(len(sub))


if __name__ == '__main__':
    main()
