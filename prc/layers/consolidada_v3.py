import argparse
import numpy as np, pandas as pd
MIN_CHANGE = 0.5


def compose(pos2, layers, min_change=MIN_CHANGE):
    base = pos2.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
    tot = base.copy(); touched = pd.Series(0, index=base.index)
    for x in layers:
        d = x.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float).reindex(base.index) - base
        d = d.where(d.abs() >= min_change, 0.0)
        touched += (d != 0).astype(int); tot += d
    assert touched.max() <= 1, 'camadas nao disjuntas'
    out = pos2[['MVT_ID_mvt']].copy(); out['TAXITIME_SEC_mvt'] = tot.reindex(pos2.MVT_ID_mvt).to_numpy().astype(pos2.TAXITIME_SEC_mvt.dtype)
    assert out.MVT_ID_mvt.is_unique and len(out) == len(pos2) and out.TAXITIME_SEC_mvt.notna().all() and (out.TAXITIME_SEC_mvt >= 0).all()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('combinado'); ap.add_argument('sem_match'); ap.add_argument('out')
    ap.add_argument('--pos2', required=True); a = ap.parse_args()
    out = compose(pd.read_parquet(a.pos2), [pd.read_parquet(a.combinado), pd.read_parquet(a.sem_match)])
    out.to_parquet(a.out, index=False)
    print('linhas alteradas', int((np.abs(out.TAXITIME_SEC_mvt.to_numpy() - pd.read_parquet(a.pos2).TAXITIME_SEC_mvt.to_numpy()) > 0).sum()), 'minimo', float(out.TAXITIME_SEC_mvt.min()))
