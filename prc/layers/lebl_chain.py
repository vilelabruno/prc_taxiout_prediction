import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adsb'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prc.paths import FEATURES, OOF, RANKING
from own_flight_features import TRAINING_ALL
from adsb_corrector_champion import frame_base
from adsb_corrector_callsign import add_blocks
import lebl_adsb_chain_corrector as C

ID = 'MVT_ID_mvt'
TARGET = 'TAXITIME_SEC_mvt'
HOLDOUT = OOF / 'champion_h045_holdout.parquet'
SEEDS = (3, 11, 29)


def _i64(d):
    d = d.copy()
    d[ID] = d[ID].astype('int64')
    return d


def clocks(d):
    d = d.copy()
    for col, name in (('AOBT_3_flt', 'aob'), ('LOBT_flt', 'lob'), ('IOBT_flt', 'iob'), ('EOBT_1_flt', 'eob')):
        d[name] = (d[col] - d.SCHED_TIME_UTC_mvt).dt.total_seconds()
    return d


def hold_frame():
    h = _i64(pd.read_parquet(HOLDOUT))
    idx = h.index
    aux = _i64(pd.read_parquet(TRAINING_ALL, columns=[ID, 'CALLSIGN_flt', 'IOBT_flt', 'EOBT_1_flt']))
    h = h.merge(aux, on=ID, how='left')
    h.index = idx
    own = _i64(pd.read_parquet(FEATURES / 'adsb_own_cs_2025.parquet'))
    h = h.merge(own, on=ID, how='left')
    h.index = idx
    h['adsb_cov'] = h.adsb_cov.fillna(0)
    cats = sorted(h.FLIGHT_mvt.astype(str).str.extract(r'^([A-Z]{2,3})')[0].dropna().unique())
    h = add_blocks(frame_base(h, cats))
    h['y'] = h[TARGET].astype(float)
    return h, cats


def rank_frame(base, cats, label, months):
    rk = pd.read_parquet(RANKING)
    rk = rk[rk.PHASE_mvt == 'DEP']
    if months:
        rk = rk[rk.MVT_TIME_UTC_mvt.dt.month.isin([int(m) for m in str(months).split(',')])]
    rk = _i64(rk.reset_index(drop=True))
    own = _i64(pd.read_parquet(FEATURES / f'adsb_own_cs_{label}.parquet'))
    rk = rk.merge(own, on=ID, how='left')
    rk['adsb_cov'] = rk.adsb_cov.fillna(0)
    rk['pred'] = rk[ID].map(base.set_index(ID)[TARGET]).astype(float)
    return add_blocks(frame_base(rk, cats))


def with_chain(d, ob):
    idx = d.index
    d = d.merge(ob.drop(columns='ano'), on=ID, how='left')
    d.index = idx
    return C.add_chain_columns(clocks(d))


def build(base, label='final', months=None, seeds=SEEDS, airport='LEBL'):
    base = _i64(base)
    h, cats = hold_frame()
    rk = rank_frame(base, cats, label, months)
    ob25 = _i64(pd.read_parquet(FEATURES / 'lebl_offblock_2025.parquet'))
    obf = _i64(pd.read_parquet(FEATURES / f'lebl_offblock_{label}.parquet'))
    train = with_chain(h[h.ADEP_mvt == airport], ob25)
    apply = with_chain(rk[rk.ADEP_mvt == airport], obf)
    pos = pd.Series(np.arange(len(base)), index=base[ID].to_numpy())
    out = base[TARGET].to_numpy(dtype='float64').copy()
    changed = 0
    for mes in sorted(train.mes.unique()):
        tr = train[train.mes == mes]
        ap = apply[apply.mes == mes]
        ma = C.obs_mask(ap)
        if not ma.any():
            continue
        c = C.fit_apply(tr, ap, seeds=seeds)
        rows = pos.reindex(ap.loc[ma, ID].to_numpy()).to_numpy()
        out[rows] = np.clip(ap.pred.to_numpy()[ma] + c.to_numpy()[ma], 0, None)
        changed += int(ma.sum())
    res = base[[ID]].copy()
    res[TARGET] = out.astype(base[TARGET].dtype)
    return res, changed


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default='final')
    ap.add_argument('--months', default=None)
    a = ap.parse_args()
    res, changed = build(pd.read_parquet(a.base), a.label, a.months)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(a.out, index=False)
    print('linhas alteradas', changed)
