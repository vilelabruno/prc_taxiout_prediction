import argparse
import lightgbm as lgb
import numpy as np
import pandas as pd
from prc.espera_rot.common import ESP_OOF, LIRF_FEATS, LIRF_FRAME, LIRF_OOF, ROUNDS, SEEDS, lirf_params
from prc.espera_rot.lirf_dataset import build

DOSES = (0.25, 0.5, 0.75)


def rmse(v):
    return float(np.sqrt((v ** 2).mean()))


def cross(x, threads):
    eo = np.full(len(x), np.nan)
    for a, c in [(1, 7), (7, 1)]:
        k = (x.mes == a).values
        t = (x.mes == c).values
        eo[t] = np.mean([lgb.train(lirf_params(s, threads), lgb.Dataset(x.loc[k, LIRF_FEATS], x.loc[k, 'yt']), ROUNDS).predict(x.loc[t, LIRF_FEATS]) for s in SEEDS], axis=0)
    return eo


def report(x, eo):
    e = x.e.to_numpy()
    print('n', len(x), 'RMSE antes', round(rmse(e), 3))
    for dose in DOSES:
        e1 = e - dose * eo
        per = {m: round(rmse(e1[(x.mes == m).values]) - rmse(e[(x.mes == m).values]), 3) for m in (1, 7)}
        print('dose', dose, 'RMSE depois', round(rmse(e1), 3), 'delta', round(rmse(e1) - rmse(e), 3), per, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--threads', type=int, default=12)
    a = ap.parse_args()
    x = build() if (a.build or not LIRF_FRAME.exists()) else pd.read_parquet(LIRF_FRAME)
    x = x.reset_index(drop=True)
    eo = cross(x, a.threads)
    oof = x[['MVT_ID_mvt', 'mes', 'y', 'e']].copy()
    oof['pred'] = x.y - x.e
    oof['eo'] = eo
    ESP_OOF.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(LIRF_OOF, index=False)
    report(x, eo)
