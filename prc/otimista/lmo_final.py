import argparse, time
from pathlib import Path
import numpy as np, pandas as pd
from prc.paths import RANKING
from prc.otimista.inputs import FINAL_DIR, OTIMISTA_MODELS
from prc.otimista.lib import load_holdout, holdout_cats
from prc.otimista import frames as OF
from prc.otimista.final_ctx import CtxFinal, FIN, POOL, POOL_APPLY, PARAMS_S4, PARAMS_POOL
from prc.otimista.aplicar_final import read_pred

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--pred', required=True); ap.add_argument('--out', required=True); ap.add_argument('--months', default='2,6'); ap.add_argument('--jobs', type=int, default=3); ap.add_argument('--seeds', default='3,11,29')
    ap.add_argument('--rank', default=str(RANKING)); ap.add_argument('--own', default=str(FINAL_DIR / 'adsb_own_cs_final.parquet')); ap.add_argument('--pts', default=str(FINAL_DIR / 'e2_pts_final.parquet'))
    ap.add_argument('--arr', default=str(FINAL_DIR / 'e2_arr_final.parquet')); ap.add_argument('--rt', default=str(FINAL_DIR / 'rt_final.parquet')); ap.add_argument('--legrt', default=str(FINAL_DIR / 'leg_rt_final.parquet')); ap.add_argument('--ox', default=str(FINAL_DIR / 'ox_final.parquet'))
    ap.add_argument('--gate-eddf', type=float, default=120.0); ap.add_argument('--zero-days', default='2026-06-11'); ap.add_argument('--thr', type=float, default=0.15)
    ap.add_argument('--model-dir', default=str(OTIMISTA_MODELS)); ap.add_argument('--s4-only', action='store_true'); ap.add_argument('--pool-only', action='store_true'); ap.add_argument('--save-c', default=None, help='diretorio para gravar c por linha de 2025-T (parquet)')
    a = ap.parse_args(); t0 = time.time(); seeds = tuple(int(x) for x in a.seeds.split(',')); months = [int(x) for x in a.months.split(',')]; log = lambda *x, **k: print(*x, **{**k, 'flush': True})
    out_path = Path(a.out); assert not out_path.exists(), f'{out_path} ja existe'
    pred = read_pred(a.pred); hold = load_holdout(); cats = holdout_cats(hold)
    rk = OF.build_final(a.rank, a.own, a.pts, a.arr, months, pred, cats, a.rt, a.legrt); rows = []
    for T in months:
        cx = CtxFinal(T, rk, hold=hold, ox_final=a.ox, thr=a.thr, gate_eddf=(a.gate_eddf if a.gate_eddf >= 0 else None), zero_days=[d for d in a.zero_days.split(',') if d], log=log)
        if not a.pool_only:
            bs, key, info, ntr, cached = cx.fit_or_load('s4_lmo', FIN, False, PARAMS_S4, seeds, a.jobs, a.model_dir, True, include_own=False)
            tb, c, e = cx.lmo_eval(bs, FIN, FIN, False, doses=(1.0,)); tb['modelo'] = 'S4'; tb['n_treino'] = ntr; rows.append(tb); log(tb.to_string())
            if a.save_c: Path(a.save_c).mkdir(parents=True, exist_ok=True); pd.DataFrame({'MVT_ID_mvt': e.MVT_ID_mvt.to_numpy(), 'apt': e.ADEP_mvt.to_numpy(), 'c': c, 'pred': e.pred.to_numpy(), 'y': e.TAXITIME_SEC_mvt.to_numpy()}).to_parquet(Path(a.save_c) / f'lmo_c_s4_T{T}.parquet', index=False)
            cx.clear()
        if not a.s4_only:
            bs, key, info, ntr, cached = cx.fit_or_load('pool_lmo', POOL, True, PARAMS_POOL, seeds, a.jobs, a.model_dir, True, include_own=False)
            tb, c, e = cx.lmo_eval(bs, POOL_APPLY, POOL, True, doses=(1.0, 0.5)); tb['modelo'] = 'POOL'; tb['n_treino'] = ntr; rows.append(tb); log(tb.to_string())
            if a.save_c: pd.DataFrame({'MVT_ID_mvt': e.MVT_ID_mvt.to_numpy(), 'apt': e.ADEP_mvt.to_numpy(), 'c': c, 'pred': e.pred.to_numpy(), 'y': e.TAXITIME_SEC_mvt.to_numpy()}).to_parquet(Path(a.save_c) / f'lmo_c_pool_T{T}.parquet', index=False)
            cx.clear()
        pd.concat(rows, ignore_index=True).to_csv(out_path, index=False); log(f'[T={T}] LMO ok {time.time()-t0:.0f}s')
    log('OK ->', out_path)
