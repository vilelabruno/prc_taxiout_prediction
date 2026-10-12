import argparse
import lightgbm as lgb
import numpy as np
import pandas as pd
from prc.espera_rot.common import ESP_OOF, FEATS, ROUNDS, TRAIN_FRAME, train_params
from prc.espera_rot.dataset import build

DOSES = (0.25, 0.5, 0.75)
COL = 'L1+H_r400'


def boot(e0, e1, dk, n=400):
    u = np.unique(dk)
    ix = {d: np.where(dk == d)[0] for d in u}
    s0 = np.array([(e0[ix[d]] ** 2).sum() for d in u])
    s1 = np.array([(e1[ix[d]] ** 2).sum() for d in u])
    nn = np.array([len(ix[d]) for d in u])
    r = np.random.default_rng(0)
    o = []
    for _ in range(n):
        q = r.integers(0, len(u), len(u))
        o.append(np.sqrt(s1[q].sum() / nn[q].sum()) - np.sqrt(s0[q].sum() / nn[q].sum()))
    return round(float(np.percentile(o, 5)), 3), round(float(np.percentile(o, 95)), 3)


def rmse(v):
    return float(np.sqrt((v ** 2).mean()))


def train_params_cv(threads):
    p = train_params(1, threads)
    for k in ('seed', 'bagging_seed', 'feature_fraction_seed'):
        p.pop(k)
    return p


def cross(x, threads):
    eg = np.full(len(x), np.nan)
    for a, c in [(1, 7), (7, 1)]:
        tr = (x.mes == a).values
        te = (x.mes == c).values
        eg[te] = lgb.train(train_params_cv(threads), lgb.Dataset(x.loc[tr, FEATS], x.loc[tr, 'yt']), ROUNDS).predict(x.loc[te, FEATS])
    return eg


def report(x, eg):
    e = x.e.to_numpy()
    days = x.dia_key.to_numpy()
    print('n', len(x), 'RMSE antes (9 aeroportos)', round(rmse(e), 3))
    for dose in DOSES:
        e1 = e - dose * eg
        per = {m: round(rmse(e1[(x.mes == m).values]) - rmse(e[(x.mes == m).values]), 3) for m in (1, 7)}
        print('dose', dose, 'RMSE depois', round(rmse(e1), 3), 'delta', round(rmse(e1) - rmse(e), 3), per, boot(e, e1, days), flush=True)
    e1 = e - 0.5 * eg
    print('por aeroporto dose 0.5', {ap: round(rmse(e1[(x.apt_name == ap).values]) - rmse(e[(x.apt_name == ap).values]), 2) for ap in sorted(x.apt_name.unique())})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--threads', type=int, default=30)
    a = ap.parse_args()
    x = build() if (a.build or not TRAIN_FRAME.exists()) else pd.read_parquet(TRAIN_FRAME)
    x = x.reset_index(drop=True)
    eg = cross(x, a.threads)
    oof = x[['MVT_ID_mvt', 'apt_name', 'mes', 'y', 'e']].copy()
    oof['pred'] = x.y - x.e
    oof[COL] = eg
    ESP_OOF.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(ESP_OOF / 'oof_espera_rot_hist.parquet', index=False)
    report(x, eg)
