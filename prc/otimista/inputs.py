from prc.paths import DATA_ROOT, WORK, ARTIFACTS, MODELS, OOF, FEATURES, RANKING

DER = FEATURES / 'adsb_ampliado'
OTI = FEATURES / 'adsb_otimista'
PROC = DATA_ROOT / 'processed'
FINAL_DIR = ARTIFACTS / 'final_inputs'
HOLDOUT = OOF / 'champion_h045_holdout.parquet'
FRAMES = WORK / 'otimista' / 'holdout_frames.pkl'
HBASE_WORK = WORK / 'hbase'
OTIMISTA_MODELS = MODELS / 'otimista'
HBASE_MODELS = MODELS / 'hbase'
