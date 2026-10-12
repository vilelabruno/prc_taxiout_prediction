import pandas as pd
from prc.layers import ampliado_f4_corrector as C
from prc.otimista.inputs import DER, OTI

APTS_OX = ['EDDF', 'EDDM', 'EHAM', 'LSZH', 'LEBL', 'EGLL', 'LEMD']
NB_DIRS = [DER / 'f4', OTI / 'nb']
OX = ['ox_end_gap', 'ox_dend0', 'ox_tort', 'ox_hold_t', 'ox_thr_off', 'ox_t1000', 'ox_v1', 'ox_v2', 'ox_v3', 'ox_v4', 'ox_stop_t', 'ox_stops', 'ox_t8', 'ox_pb', 'ox_gsmax']
OXQ = ['ox_q_ahead', 'ox_q_moving', 'ox_q_hold']
LEGACY_OFF = ['EGLL', 'LEMD', 'LIRF', 'LFPG']


def nb_path(M):
    for d in NB_DIRS:
        if (d / f'nbfeat_m{M}.parquet').exists():
            return d / f'nbfeat_m{M}.parquet'
    raise FileNotFoundError(M)


def legacy_off(d):
    d = d.copy()
    for a in LEGACY_OFF:
        m = (d.ADEP_mvt == a).to_numpy()
        for c in C.ADSB + C.POS_FULL:
            d.loc[m, c] = float('nan')
    return d
