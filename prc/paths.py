import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('PRC_DATA_ROOT', REPO / 'data'))
WORK = Path(os.environ.get('PRC_WORK', REPO / 'work'))
ARTIFACTS = Path(os.environ.get('PRC_ARTIFACTS', REPO / 'artifacts'))
MODELS = ARTIFACTS / 'models'
OOF = ARTIFACTS / 'oof'
FEATURES = ARTIFACTS / 'features'
RANKING = Path(os.environ.get('PRC_RANKING', DATA_ROOT / 'final' / 'final_ranking.parquet'))
SUBMITTING = Path(os.environ.get('PRC_SUBMITTING', DATA_ROOT / 'final' / 'final_submitting.parquet'))
TRAINING_DIR = Path(os.environ.get('PRC_TRAINING_DIR', DATA_ROOT / 'raw'))
SURFACE_DIR = Path(os.environ.get('PRC_SURFACE_DIR', DATA_ROOT / 'processed' / 'adsb_surface'))


def ensure(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
