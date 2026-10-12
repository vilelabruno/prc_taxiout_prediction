import argparse
import lightgbm as lgb
import numpy as np
import pandas as pd
from prc.paths import RANKING
from prc.espera_rot.common import DOSE, ESP_MODELS, FLOOR, KEY, LIRF, LIRF_FEATS, RAW_COLS, SEEDS, TARGET, TRAINING_ALL
from prc.espera_rot.features import hist_frame, rank_features


def load_models():
    return [lgb.Booster(model_file=str(ESP_MODELS / f'lirf_seed{sd}.txt')) for sd in SEEDS]


def build_frame(pred, ranking, months=None):
    p = pred[[KEY, TARGET]].copy()
    p[KEY] = p[KEY].astype('int64')
    rk = ranking[(ranking.PHASE_mvt == 'ARR') | (ranking.ADEP_mvt == LIRF)]
    hist = hist_frame(pd.read_parquet(TRAINING_ALL, columns=RAW_COLS))
    f = rank_features(rk, set(p[KEY]), hist, months)
    f = f[f.apt_name == LIRF]
    f = f.merge(p.rename(columns={TARGET: 'pred'}), on=KEY, how='inner')
    f['sl_slack'] = f.sl_ub - f.pred
    return f


def ehat(f, models):
    X = f[LIRF_FEATS].to_numpy(float)
    return sum(m.predict(X) for m in models) / len(models)


def correct(pred, ranking, dose=DOSE, months=None, models=None):
    f = build_frame(pred, ranking, months)
    f['ehat'] = ehat(f, models or load_models())
    f['new'] = np.maximum(FLOOR, f.pred + dose * f.ehat)
    out = pred[[KEY, TARGET]].copy()
    mapped = out[KEY].astype('int64').map(pd.Series(f.new.to_numpy(), index=f[KEY].to_numpy()))
    out[TARGET] = mapped.where(mapped.notna(), out[TARGET]).to_numpy()
    return out, f


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ranking', default=str(RANKING))
    ap.add_argument('--months', default=None)
    ap.add_argument('--dose', type=float, default=DOSE)
    ap.add_argument('--ehat-out', default=None)
    a = ap.parse_args()
    months = [int(m) for m in a.months.split(',')] if a.months else None
    rk = pd.read_parquet(a.ranking, columns=RAW_COLS)
    pred = pd.read_parquet(a.pred, columns=[KEY, TARGET])
    out, f = correct(pred, rk, a.dose, months)
    out.to_parquet(a.out, index=False)
    if a.ehat_out:
        f[[KEY, 'pred', 'ehat']].to_parquet(a.ehat_out, index=False)
    d = out[TARGET].to_numpy() - pred[TARGET].to_numpy()
    print('linhas', len(out), 'LIRF corrigidas', len(f), 'alteradas', int((np.abs(d) > 1e-9).sum()), 'RMS da correcao', float(np.sqrt(np.mean(d ** 2))), 'media', float(d.mean()))
