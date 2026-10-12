import argparse
import numpy as np
import pandas as pd
from prc.paths import ARTIFACTS

KEY = 'MVT_ID_mvt'
TARGET = 'TAXITIME_SEC_mvt'
POOL = ('LEBL_50_50', 'EGLL_x0.5', 'LEMD_jan_like_x0.5', 'LEMD_jul_like_x0.5')


def series(path):
    d = pd.read_parquet(path)
    d[KEY] = d[KEY].astype('int64')
    return d.set_index(KEY)[TARGET].astype(float)


def compose(v36, e6, d):
    v53 = e6.reindex(v36.index).fillna(v36)
    d = d.copy()
    d[KEY] = d[KEY].astype('int64')
    d = d.set_index(KEY)
    use = d.elig.to_numpy(bool) & (d.camada.to_numpy() != '')
    v53.loc[d.index[use]] = d.novo.to_numpy(float)[use]
    c_s4 = pd.Series(np.where(d.camada.to_numpy() == 'S4', d.c.to_numpy(float), 0.0), index=d.index)
    c_pool = pd.Series(np.where(np.isin(d.camada.to_numpy(), POOL), d.c.to_numpy(float), 0.0), index=d.index)
    return v53, c_s4, c_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v36', default=str(ARTIFACTS / 'v36_final.parquet'))
    ap.add_argument('--e6', required=True)
    ap.add_argument('--camada-d', required=True)
    ap.add_argument('--out-v53', default=str(ARTIFACTS / 'stack_inputs' / 'v53.parquet'))
    ap.add_argument('--out-layers', default=str(ARTIFACTS / 'stack_inputs' / 'layers_s4_pool.parquet'))
    a = ap.parse_args()
    v36 = series(a.v36)
    v53, c_s4, c_pool = compose(v36, series(a.e6), pd.read_parquet(a.camada_d))
    ARTIFACTS.joinpath('stack_inputs').mkdir(parents=True, exist_ok=True)
    v53.rename(TARGET).reset_index().to_parquet(a.out_v53, index=False)
    pd.DataFrame({'c_s4': c_s4, 'c_pool': c_pool}).reset_index().to_parquet(a.out_layers, index=False)
    print(len(v53), int((v53 != v36).sum()))


if __name__ == '__main__':
    main()
