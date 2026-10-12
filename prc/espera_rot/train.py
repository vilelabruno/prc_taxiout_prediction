import argparse
import lightgbm as lgb
import pandas as pd
from prc.espera_rot.common import ESP_MODELS, FEATS, ROUNDS, SEEDS, TRAIN_FRAME, train_params
from prc.espera_rot.dataset import build


def fit(tr, threads):
    return [lgb.train(train_params(sd, threads), lgb.Dataset(tr[FEATS], tr.yt), ROUNDS) for sd in SEEDS]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--threads', type=int, default=12)
    a = ap.parse_args()
    tr = build() if (a.build or not TRAIN_FRAME.exists()) else pd.read_parquet(TRAIN_FRAME)
    ESP_MODELS.mkdir(parents=True, exist_ok=True)
    for sd, m in zip(SEEDS, fit(tr, a.threads)):
        m.save_model(str(ESP_MODELS / f'seed{sd}.txt'))
    print('treino', len(tr), 'modelos em', ESP_MODELS)
