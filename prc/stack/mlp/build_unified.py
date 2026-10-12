import argparse, json, re, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from prc.stack.cfg import MLP_UNI, WORK, ensure
ap = argparse.ArgumentParser()
ap.add_argument('--v13-dir', default=str(WORK / 'mlp_member' / 'vm' / 'v13'))
ap.add_argument('--fox-dir', default=str(WORK / 'h2h4h8'))
ap.add_argument('--marcus-dir', default=str(WORK / 'frenteA'))
ap.add_argument('--out', default=str(MLP_UNI))
a = ap.parse_args()
V = a.v13_dir.rstrip('/') + '/'
FOX = a.fox_dir.rstrip('/') + '/'
MAR = a.marcus_dir.rstrip('/') + '/'
U = a.out.rstrip('/') + '/'
ensure(U)
CATS = ['ADEP_mvt', 'ADES_mvt', 'AIRCRAFT_OPERATOR_flt', 'AIRCRAFT_TYPE_mvt', 'RUNWAY_mvt', 'STAND_mvt',
        'MARKET_SEGMENT_flt', 'FLIGHT_TYPE_flt', 'WK_TBL_CAT_flt', 'FLIGHT_RULE_mvt']
BAN = re.compile(r'pred|TAXITIME', re.I)

def numeric_cols(schema, skip=()):
    out = []
    for f in schema:
        if f.name in skip or BAN.search(f.name):
            continue
        t = f.type
        if pa.types.is_floating(t) or pa.types.is_integer(t) or pa.types.is_boolean(t):
            out.append(f.name)
    return out

def to_f32(col):
    return col.to_numpy(zero_copy_only=False).astype(np.float32)

dep_tr = pd.read_parquet(V + 'training/departures.parquet')
dep_rk = pd.read_parquet(V + 'ranking/departures.parquet')
ids_tr, ids_rk = dep_tr.MVT_ID_mvt.values, dep_rk.MVT_ID_mvt.values
y = dep_tr.TAXITIME_SEC_mvt.values.astype(np.float32)
mo = pd.to_datetime(dep_tr.MVT_TIME_UTC_mvt).dt.month.values.astype(np.int8)
print('treino', len(ids_tr), 'rank', len(ids_rk), flush=True)

names, A_tr, A_rk = [], [], []

def add(name, a, b):
    names.append(name); A_tr.append(a); A_rk.append(b)

s_tr = pq.ParquetFile(V + 'training/features.parquet').schema_arrow
s_rk = pq.ParquetFile(V + 'ranking/features.parquet').schema_arrow
v13 = [c for c in numeric_cols(s_tr) if c in s_rk.names]
for c in v13:
    add(c, to_f32(pq.read_table(V + 'training/features.parquet', columns=[c]).column(0)),
        to_f32(pq.read_table(V + 'ranking/features.parquet', columns=[c]).column(0)))
print('V13', len(v13), flush=True)

pos_tr = pd.Series(np.arange(len(ids_tr)), index=ids_tr)
pos_rk = pd.Series(np.arange(len(ids_rk)), index=ids_rk)
have = set(names)

def add_keyed(df_all, df_rk, tag):
    n = 0
    for c in df_all.columns:
        if c in have or c not in df_rk.columns or BAN.search(c) or c == 'MVT_ID_mvt':
            continue
        if not (pd.api.types.is_numeric_dtype(df_all[c]) or pd.api.types.is_bool_dtype(df_all[c])):
            continue
        a = np.full(len(ids_tr), np.nan, np.float32)
        a[pos_tr.loc[df_all.MVT_ID_mvt.values].values] = df_all[c].astype(np.float32).values
        b = np.full(len(ids_rk), np.nan, np.float32)
        b[pos_rk.loc[df_rk.MVT_ID_mvt.values].values] = df_rk[c].astype(np.float32).values
        add(tag + c if tag else c, a, b); have.add(c); n += 1
    return n

fx = pd.concat([pd.read_parquet(FOX + 'ext.parquet'), pd.read_parquet(FOX + 'hold.parquet')], ignore_index=True)
fr = pd.read_parquet(FOX + 'rk.parquet')
print('fox novas', add_keyed(fx, fr, ''), flush=True)
del fx, fr

mt = pd.concat([pd.concat([pd.read_parquet(MAR + 'X_train.parquet'), pd.read_parquet(MAR + 'meta_train.parquet')[['MVT_ID_mvt']]], axis=1),
                pd.concat([pd.read_parquet(MAR + 'X_valid.parquet'), pd.read_parquet(MAR + 'meta_valid.parquet')[['MVT_ID_mvt']]], axis=1)], ignore_index=True)
mr = pd.concat([pd.read_parquet(MAR + 'X_rank.parquet'), pd.read_parquet(MAR + 'meta_rank.parquet')[['MVT_ID_mvt']]], axis=1)
mt = mt[[c for c in mt.columns if not c.startswith('cat_')]]
mr = mr[[c for c in mr.columns if not c.startswith('cat_')]]
print('marcus novas', add_keyed(mt, mr, 'm_'), flush=True)

keep = [i for i, a in enumerate(A_tr) if np.nanstd(a[mo != 1][:400000]) > 0]
print('colunas', len(names), 'nao constantes', len(keep), flush=True)
names = [names[i] for i in keep]
X = np.lib.format.open_memmap(U + 'X_all.npy', 'w+', np.float32, (len(ids_tr), len(keep)))
XR = np.lib.format.open_memmap(U + 'X_rank.npy', 'w+', np.float32, (len(ids_rk), len(keep)))
for i0 in range(0, len(ids_tr), 50000):
    X[i0:i0 + 50000] = np.column_stack([A_tr[i][i0:i0 + 50000] for i in keep])
for i0 in range(0, len(ids_rk), 50000):
    XR[i0:i0 + 50000] = np.column_stack([A_rk[i][i0:i0 + 50000] for i in keep])
X.flush(); XR.flush()
json.dump(names, open(U + 'cols.json', 'w'))
np.save(U + 'y_all.npy', y); np.save(U + 'month_all.npy', mo)
np.save(U + 'ids_all.npy', ids_tr); np.save(U + 'ids_rank.npy', ids_rk)
ct = dep_tr[[c for c in CATS if c in dep_tr.columns]].copy(); cr = dep_rk[[c for c in CATS if c in dep_rk.columns]].copy()
ct['cs3'] = dep_tr.CALLSIGN_flt.astype(str).str[:3]; cr['cs3'] = dep_rk.CALLSIGN_flt.astype(str).str[:3]
ct.to_parquet(U + 'cat_all.parquet'); cr.to_parquet(U + 'cat_rank.parquet')
print('ok', X.shape, XR.shape)
