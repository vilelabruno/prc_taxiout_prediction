import pickle
from pathlib import Path
import pandas as pd
from prc.layers import ampliado_f4_corrector as C
from prc.otimista.inputs import DER, FRAMES

S4 = ['EDDF', 'EDDM', 'EHAM', 'LSZH']


def _i64(d):
    d = d.copy()
    d['MVT_ID_mvt'] = d.MVT_ID_mvt.astype('int64')
    return d


def load_frames():
    if not Path(FRAMES).exists():
        from prc.otimista.frames import build
        build()
    return pickle.load(open(FRAMES, 'rb'))


def load_holdout():
    h = load_frames()
    idx = h.index
    h = _i64(h)
    rt = _i64(pd.read_parquet(DER / 'f4/rt_2025.parquet'))
    lg = _i64(pd.read_parquet(DER / 'f4/leg_rt_2025.parquet'))
    h = h.merge(rt, on='MVT_ID_mvt', how='left').merge(lg[['MVT_ID_mvt'] + C.LGF + C.LRT], on='MVT_ID_mvt', how='left')
    h.index = idx
    return C.derived(h)


def holdout_cats(h):
    return sorted(h.FLIGHT_mvt.astype(str).str.extract(r'^([A-Z]{2,3})')[0].dropna().unique())
