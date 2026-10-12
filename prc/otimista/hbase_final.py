import argparse, json, os, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from prc.paths import RANKING
from prc.otimista import frames as OF
from prc.otimista import final_ctx as FC
from prc.otimista.inputs import FINAL_DIR, HBASE_WORK, HBASE_MODELS
from prc.otimista.lib import load_holdout, holdout_cats
from prc.otimista.build import WDIST, dist
from prc.otimista.hbase_prep import fill_ox, read_frames

PR = dict(objective='regression', num_leaves=255, min_child_samples=40, n_estimators=500, learning_rate=0.06, colsample_bytree=0.7, subsample=0.8, subsample_freq=1, reg_lambda=3.0, force_row_wise=True, verbose=-1)


def masks(df):
    y = df.y.to_numpy(float)
    eco = (((df.y - df.m).abs() <= 120) & (df.m > 3600)).to_numpy()
    inc = ((df.sem_aobt == 0) & ((df.y - df.mvt_aobt).abs() > 1200) & (df.y < 3600)).to_numpy()
    day = (df.y > 80000).to_numpy()
    return y, eco, inc, day


VARS = {'raw': lambda y, eco, inc, day: (np.ones(len(y), bool), y), 'drop80k': lambda y, eco, inc, day: (y <= 80000, y), 'drop20k': lambda y, eco, inc, day: (y <= 20000, y),
        'clip3600': lambda y, eco, inc, day: (np.ones(len(y), bool), np.minimum(y, 3600)), 'defects': lambda y, eco, inc, day: (~(eco | inc | day), y)}


def weights(T, e, h):
    w_e = e.mm.map(lambda m: WDIST[dist(int(m), T)]).to_numpy(float)
    hs = [h[h.mes == m] for m in (1, 7)]
    return [w_e] + [np.full(len(x), WDIST[dist(m, T)]) for x, m in zip(hs, (1, 7))], [e] + hs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--vars', default=','.join(VARS))
    ap.add_argument('--months', default='2,6')
    ap.add_argument('--pred', required=True)
    ap.add_argument('--jobs', type=int, default=int(os.environ.get('JOBS', 12)))
    ap.add_argument('--work', default=str(HBASE_WORK))
    ap.add_argument('--model-dir', default=str(HBASE_MODELS))
    ap.add_argument('--rank', default=str(RANKING))
    ap.add_argument('--own', default=str(FINAL_DIR / 'adsb_own_cs_final.parquet'))
    ap.add_argument('--pts', default=str(FINAL_DIR / 'e2_pts_final.parquet'))
    ap.add_argument('--arr', default=str(FINAL_DIR / 'e2_arr_final.parquet'))
    ap.add_argument('--rt', default=str(FINAL_DIR / 'rt_final.parquet'))
    ap.add_argument('--legrt', default=str(FINAL_DIR / 'leg_rt_final.parquet'))
    ap.add_argument('--ox', default=str(FINAL_DIR / 'ox_final.parquet'))
    ap.add_argument('--lirf-ox', default=str(FINAL_DIR / 'ox_LIRF_final.parquet'))
    a = ap.parse_args()
    months = [int(x) for x in a.months.split(',')]
    assert all(T in (2, 6) for T in months)
    h, e, F = read_frames(a.work)
    FB = [f for f in F if f != 'pred']
    pred = pd.read_parquet(a.pred)
    pred['MVT_ID_mvt'] = pred.MVT_ID_mvt.astype('int64')
    pred = pred.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
    hold = load_holdout()
    rk = OF.build_final(a.rank, a.own, a.pts, a.arr, months, pred, holdout_cats(hold), a.rt, a.legrt)
    lx = pd.read_parquet(a.lirf_ox)
    lx['MVT_ID_mvt'] = lx.MVT_ID_mvt.astype('int64')
    out = Path(a.work) / 'final'
    out.mkdir(parents=True, exist_ok=True)
    md = Path(a.model_dir)
    md.mkdir(parents=True, exist_ok=True)
    log = []
    for T in months:
        cx = FC.CtxFinal(T, rk, hold=hold, ox_final=a.ox)
        r = fill_ox(cx.rkp, lx)
        r = r.assign(MVT_ID_mvt=r.MVT_ID_mvt.astype('int64'))
        miss = [f for f in FB if f not in r.columns]
        assert not miss, miss
        e_T = cx.prep(e)
        h_T = cx.prep(h)
        np.save(out / f'ids_T{T}.npy', r.MVT_ID_mvt.astype('int64').to_numpy())
        ws, fr = weights(T, e_T, h_T)
        X = pd.concat([f[FB] for f in fr])
        ms = [masks(f) for f in fr]
        y_ = np.concatenate([m[0] for m in ms])
        eco = np.concatenate([m[1] for m in ms])
        inc = np.concatenate([m[2] for m in ms])
        day = np.concatenate([m[3] for m in ms])
        w = np.concatenate(ws)
        for nm in a.vars.split(','):
            t0 = time.time()
            keep, yy = VARS[nm](y_, eco, inc, day)
            m = lgb.LGBMRegressor(**dict(PR, n_jobs=a.jobs, random_state=3)).fit(X[keep], yy[keep], sample_weight=w[keep])
            m.booster_.save_model(str(md / f'hbase_{nm}_T{T}.txt'))
            p = m.predict(r[FB])
            np.save(out / f'base_{nm}_T{T}.npy', p)
            row = dict(T=T, var=nm, n_treino=int(keep.sum()), n_rank=int(len(r)), media=round(float(p.mean()), 1), nan=int(np.isnan(p).sum()), seg=round(time.time() - t0))
            log.append(row)
            print(json.dumps(row), flush=True)
    json.dump({'features': FB, 'params': PR, 'log': log}, open(out / 'meta.json', 'w'))
