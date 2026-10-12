import numpy as np
import pandas as pd
from prc.paths import DATA_ROOT, MODELS, OOF, RANKING, WORK

APTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
KEY = 'MVT_ID_mvt'
TARGET = 'TAXITIME_SEC_mvt'
DOSE = 0.5
FLOOR = 60.0
GATED = 'LIRF'
SEEDS = (1, 2, 3)
ROUNDS = 400
CLIP = 1200.0
Y_MAX = 50000
HIST_K = 30.0
HIST_EXCLUDE_MONTHS = (1, 7)

TRAINING_ALL = DATA_ROOT / 'processed' / 'training_all_2025.parquet'
STACK_DIR = WORK / 'stack'
ESP_WORK = WORK / 'espera_rot'
ESP_MODELS = MODELS / 'espera_rot'
ESP_OOF = OOF / 'espera_rot'
TRAIN_FRAME = ESP_WORK / 'train.parquet'

ROT = ['rot_linked', 'rot_same_car', 'rot_gap', 'rot_good', 'rot_arr_delay', 'rot_taxi_in', 'rot_n_prev6h']
SL = ['sl_ub', 'sl_gap_sched', 'sl_delay', 'sl_slack']
HKEYS = [(['apt', 'stand'], 'h_st'), (['apt', 'stand', 'rwy'], 'h_sr'), (['apt', 'rwy', 'hr'], 'h_rh')]
HSUF = ['_eff', '_n', '_std', '_tail', '_aoff', '_aostd']
HCOLS = [n + s for _, n in HKEYS for s in HSUF]
FEATS = ['pred', 'apt', 'ctx_mvt_aobt', 'ctx_sem_aobt'] + ROT + SL + HCOLS

PARAMS = dict(objective='l2', learning_rate=0.03, num_leaves=15, min_data_in_leaf=1000, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=50, verbose=-1)

EPOCH = pd.Timestamp('1970-01-01', tz='UTC')
ROT_ORIGIN = pd.Timestamp('2025-01-01', tz='UTC')

RAW_COLS = [KEY, 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
            'AIRCRAFT_TYPE_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'AOBT_3_flt', 'WK_TBL_CAT_flt']


def epoch(s, origin=EPOCH):
    return (pd.to_datetime(s, utc=True) - origin).dt.total_seconds()


def train_params(seed, threads):
    return dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed, num_threads=threads)


def apt_code(a):
    return a.map({x: i for i, x in enumerate(APTS)})

LIRF = 'LIRF'
LIRF_FEATS = [c for c in FEATS if c != 'apt']
LIRF_PARAMS = dict(PARAMS, min_data_in_leaf=300)
LIRF_FRAME = ESP_WORK / 'train_lirf.parquet'
LIRF_OOF = ESP_OOF / 'oof_lirf_receita_v85.parquet'


def lirf_params(seed, threads):
    return dict(LIRF_PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed, num_threads=threads)
