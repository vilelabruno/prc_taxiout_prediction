import argparse
import lightgbm as lgb
import pandas as pd
from prc.espera_rot.common import ESP_MODELS, LIRF_FEATS, LIRF_FRAME, ROUNDS, SEEDS, lirf_params
from prc.espera_rot.lirf_dataset import build


def fit(tr, threads):
    return [lgb.train(lirf_params(sd, threads), lgb.Dataset(tr[LIRF_FEATS], tr.yt), ROUNDS) for sd in SEEDS]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--threads', type=int, default=12)
    a = ap.parse_args()
    tr = build() if (a.build or not LIRF_FRAME.exists()) else pd.read_parquet(LIRF_FRAME)
    ESP_MODELS.mkdir(parents=True, exist_ok=True)
    for sd, m in zip(SEEDS, fit(tr, a.threads)):
        m.save_model(str(ESP_MODELS / f'lirf_seed{sd}.txt'))
    print('treino', len(tr), 'modelos em', ESP_MODELS)
