import argparse
import lightgbm as lgb
import numpy as np
import pandas as pd
from prc.paths import RANKING
from prc.espera_rot.common import DOSE, ESP_MODELS, FEATS, FLOOR, GATED, KEY, RAW_COLS, SEEDS, TARGET, TRAINING_ALL, apt_code
from prc.espera_rot.features import hist_frame, rank_features


def load_models():
    return [lgb.Booster(model_file=str(ESP_MODELS / f'seed{sd}.txt')) for sd in SEEDS]


def build_frame(pred, ranking, months=None):
    p = pred[[KEY, TARGET]].copy()
    p[KEY] = p[KEY].astype('int64')
    hist = hist_frame(pd.read_parquet(TRAINING_ALL, columns=RAW_COLS))
    f = rank_features(ranking, set(p[KEY]), hist, months)
    f = f.merge(p.rename(columns={TARGET: 'pred'}), on=KEY, how='inner')
    f['sl_slack'] = f.sl_ub - f.pred
    f['apt'] = apt_code(f.apt_name)
    return f


def ehat(f, models):
    X = f[FEATS].to_numpy(float)
    return sum(m.predict(X) for m in models) / len(models)


def correct(pred, ranking, dose=DOSE, months=None, models=None):
    f = build_frame(pred, ranking, months)
    f['ehat'] = ehat(f, models or load_models())
    f['gate'] = (f.apt_name != GATED).astype(float)
    f['new'] = np.maximum(FLOOR, f.pred + dose * f.ehat * f.gate)
    out = pred[[KEY, TARGET]].copy()
    new = pd.Series(f.new.to_numpy(), index=f[KEY].to_numpy())
    k = out[KEY].astype('int64')
    mapped = k.map(new)
    out[TARGET] = mapped.where(mapped.notna(), out[TARGET]).to_numpy()
    return out, f


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ranking', default=str(RANKING))
    ap.add_argument('--months', default=None)
    ap.add_argument('--dose', type=float, default=DOSE)
    a = ap.parse_args()
    months = [int(m) for m in a.months.split(',')] if a.months else None
    rk = pd.read_parquet(a.ranking, columns=RAW_COLS)
    pred = pd.read_parquet(a.pred, columns=[KEY, TARGET])
    out, f = correct(pred, rk, a.dose, months)
    out.to_parquet(a.out, index=False)
    d = out[TARGET].to_numpy() - pred[TARGET].to_numpy()
    print('linhas', len(out), 'alteradas', int((np.abs(d) > 1e-9).sum()), 'RMS da correcao', float(np.sqrt(np.mean(d ** 2))))
