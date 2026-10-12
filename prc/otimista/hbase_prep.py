import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from prc.layers import ampliado_f4_corrector as C
from prc.layers.adsb_corrector_callsign import obs_mask
from prc.otimista.inputs import DER, OTI, HBASE_WORK
from prc.otimista.lib import load_holdout, _i64
from prc.otimista.data import nb_path, OX, OXQ
from prc.otimista.build import ALLM

OXALL = OX + OXQ


def lirf_ox(lab):
    return _i64(pd.read_parquet(OTI / 'ox' / f'ox_LIRF_{lab}.parquet'))


def fill_ox(fr, ox):
    fr = fr.copy()
    idx = fr.index
    ids = fr.MVT_ID_mvt.astype('int64')
    z = ox.set_index('MVT_ID_mvt').reindex(ids)
    m = (fr.ADEP_mvt == 'LIRF').to_numpy()
    for f in OXALL:
        if f not in fr.columns:
            fr[f] = np.nan
        fr.loc[m, f] = z[f].to_numpy()[m]
    fr.index = idx
    return fr


def read_frames(work=HBASE_WORK):
    work = Path(work)
    h = pd.read_parquet(work / 'hold.parquet')
    e = pd.read_parquet(work / 'ext.parquet')
    feats = pd.read_csv(work / 'feats.csv').iloc[:, 0].tolist()
    h['y'] = h.y.astype(float)
    e['y'] = e.TAXITIME_SEC_mvt.astype(float)
    return h, e, feats


def build(work=HBASE_WORK):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    pk = json.load(open(DER / 'f4/pos_keep.json'))
    pk.pop('LEBL', None)
    new_keep = C.keep_stable(pd.read_csv(DER / 'f4/drift_all.csv'))
    h = load_holdout()
    y = h.TAXITIME_SEC_mvt.astype(float).to_numpy()
    d = C.prep(h, pk, new_keep)
    idx = d.index
    oxh = pd.concat([_i64(pd.read_parquet(OTI / f'ox/ox_m{k}.parquet')) for k in (1, 7)])
    d = _i64(d).merge(oxh, on='MVT_ID_mvt', how='left')
    d.index = idx
    d = fill_ox(d, pd.concat([lirf_ox('m1'), lirf_ox('m7')]))
    feats = [f for f in C.feature_list() if f in d.columns] + [f for f in OXALL if f in d.columns]
    extra = ['MVT_ID_mvt', 'ADEP_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'CALLSIGN_flt', 'adsb_cov']
    cols = list(dict.fromkeys(extra + feats + ['ltx0', 'ltxfm', 'tx0', 'txfm', 'tfm_off', 'mes', 'dia']))
    hold = d[[c for c in cols if c in d.columns]].copy()
    hold['y'] = y
    hold['mes'] = h.mes.to_numpy()
    hold['dia'] = h.dia.to_numpy()
    hold.to_parquet(work / 'hold.parquet')
    ex = []
    for mm in ALLM:
        e = C.prep(pd.read_parquet(nb_path(mm)), pk, new_keep)
        e = _i64(e).merge(_i64(pd.read_parquet(OTI / f'ox/ox_m{mm}.parquet')), on='MVT_ID_mvt', how='left')
        e = fill_ox(e, lirf_ox(f'm{mm}'))
        t = e[[c for c in cols + ['TAXITIME_SEC_mvt'] if c in e.columns]].copy()
        t['mm'] = mm
        ex.append(t)
    ext = pd.concat(ex)
    ext.to_parquet(work / 'ext.parquet')
    pd.Series(feats).to_csv(work / 'feats.csv', index=False)
    print(hold.shape, ext.shape, len(feats), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=str(HBASE_WORK))
    a = ap.parse_args()
    build(a.work)
