from prc.paths import ARTIFACTS, DATA_ROOT, FEATURES, MODELS, OOF, RANKING, SUBMITTING, WORK, ensure

KEY = 'MVT_ID_mvt'
TARGET = 'TAXITIME_SEC_mvt'
DEPLOY = WORK / 'stack'
STACK_MODELS = MODELS / 'stack'
STACK_OOF = OOF / 'stack'
HBASE_DIR = OOF / 'hbase'
SEQ_OUT = OOF / 'h3_seq'
HARM_DIR = WORK / 'h3' / 'harm'
MLP_UNI = WORK / 'mlp_member' / 'uni'
MLP_OUT = OOF / 'mlp_member'
