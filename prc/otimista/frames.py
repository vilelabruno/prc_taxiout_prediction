import argparse, pickle, time
from pathlib import Path
import pandas as pd
from prc.layers.adsb_corrector_champion import frame_base
from prc.layers.adsb_corrector_callsign import add_blocks
from prc.adsb import pos_features as pf
from prc.layers import ampliado_f4_corrector as C
from prc.otimista.inputs import DER, PROC, RANKING, HOLDOUT, FRAMES
from prc.otimista.lib import holdout_cats

COLS = ['MVT_ID_mvt', 'ADES_mvt', 'STAND_mvt']


def stands_cons(arr26=None, rank26=None):
    a25 = pd.read_parquet(DER / 'e2_arr_2025.parquet').merge(pd.read_parquet(PROC / 'training_all_2025.parquet', columns=COLS + ['MVT_TIME_UTC_mvt']), on='MVT_ID_mvt')
    a26 = pd.read_parquet(arr26 or DER / 'e2_arr_2026.parquet').merge(pd.read_parquet(rank26 or RANKING, columns=COLS + ['MVT_TIME_UTC_mvt']), on='MVT_ID_mvt')
    return pf.stand_tables(a25, a26, 'cons')


def load_h(t25):
    h = pd.read_parquet(HOLDOUT)
    idx = h.index
    t = pd.read_parquet(PROC / 'training_all_2025.parquet', columns=['MVT_ID_mvt', 'CALLSIGN_flt'])
    h['MVT_ID_mvt'] = h.MVT_ID_mvt.astype('int64')
    t['MVT_ID_mvt'] = t.MVT_ID_mvt.astype('int64')
    h = h.merge(t, on='MVT_ID_mvt', how='left')
    h.index = idx
    own = pd.read_parquet(PROC / 'adsb_own_cs_2025.parquet')
    own['MVT_ID_mvt'] = own.MVT_ID_mvt.astype('int64')
    h = h.merge(own, on='MVT_ID_mvt', how='left')
    h.index = idx
    h['adsb_cov'] = h.adsb_cov.fillna(0)
    h = pf.add_pos_features(h, pd.read_parquet(DER / 'e2_pts_2025.parquet'), t25)
    h = frame_base(h)
    h = add_blocks(h)
    return h


def load_rk(t26, cats, rank, own, pts, months, pred):
    rk = pd.read_parquet(rank)
    rk = rk[rk.PHASE_mvt == 'DEP'].reset_index(drop=True)
    rk['MVT_ID_mvt'] = rk.MVT_ID_mvt.astype('int64')
    if months is not None:
        rk = rk[rk.MVT_TIME_UTC_mvt.dt.month.isin(list(months))].reset_index(drop=True)
    o = pd.read_parquet(own)
    o['MVT_ID_mvt'] = o.MVT_ID_mvt.astype('int64')
    rk = rk.merge(o, on='MVT_ID_mvt', how='left')
    rk['adsb_cov'] = rk.adsb_cov.fillna(0)
    rk = pf.add_pos_features(rk, pd.read_parquet(pts), t26)
    rk['pred'] = rk.MVT_ID_mvt.map(pred)
    rk['pred'] = rk.pred.astype(float)
    rk = frame_base(rk, cats)
    rk = add_blocks(rk)
    return rk


def build(out=FRAMES):
    t0 = time.time()
    t25, t26 = stands_cons()
    h = load_h(t25)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(h, open(out, 'wb'))
    print('holdout', len(h), f'{time.time()-t0:.0f}s ->', out, flush=True)


def build_final(rank, own, pts, arr, months, pred, cats, rt, legrt):
    t25, t26 = stands_cons(arr26=arr, rank26=rank)
    rk = load_rk(t26, cats, rank, own, pts, months, pred)
    r = pd.read_parquet(rt)
    r['MVT_ID_mvt'] = r.MVT_ID_mvt.astype('int64')
    lg = pd.read_parquet(legrt)
    lg['MVT_ID_mvt'] = lg.MVT_ID_mvt.astype('int64')
    idx = rk.index
    rk = rk.merge(r, on='MVT_ID_mvt', how='left').merge(lg[['MVT_ID_mvt'] + C.LGF + C.LRT], on='MVT_ID_mvt', how='left')
    rk.index = idx
    return C.derived(rk)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(FRAMES))
    a = ap.parse_args()
    build(Path(a.out))
