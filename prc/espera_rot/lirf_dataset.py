import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from prc.espera_rot.common import CLIP, HCOLS, HIST_EXCLUDE_MONTHS, KEY, LIRF, LIRF_FEATS, LIRF_FRAME, RAW_COLS, STACK_DIR, TRAINING_ALL, Y_MAX
from prc.espera_rot.features import core_features, hist_features, hist_frame


def build(stack_dir=STACK_DIR, out=LIRF_FRAME):
    d = pd.read_parquet(stack_dir / 'dep25.parquet', columns=[KEY, 'y', 'apt', 'mes', 'dia_key', 'hand'])
    d[KEY] = d[KEY].astype('int64')
    d = d.rename(columns={'apt': 'apt_name'})
    d = d[(d.apt_name == LIRF) & (d.y < Y_MAX)].reset_index(drop=True)
    x = pd.read_parquet(TRAINING_ALL, columns=RAW_COLS)
    dep_all = x[x.PHASE_mvt == 'DEP']
    dep = dep_all[dep_all[KEY].astype('int64').isin(set(d[KEY]))]
    arr = x[x.PHASE_mvt == 'ARR']
    core = core_features(dep, arr, arr).drop(columns=['apt_name'])
    hf = hist_features(hist_frame(x, HIST_EXCLUDE_MONTHS), dep)
    yy = (dep.MVT_TIME_UTC_mvt - dep.BLOCK_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    hf.loc[~((yy > 60) & (yy < 5000)), HCOLS] = np.nan
    tr = d.merge(core, on=KEY, how='left').merge(hf, on=KEY, how='left')
    tr['pred'] = tr.hand
    tr['e'] = tr.y - tr.pred
    tr['yt'] = tr.e.clip(-CLIP, CLIP)
    tr['sl_slack'] = tr.sl_ub - tr.pred
    tr = tr[[KEY, 'apt_name', 'mes', 'dia_key', 'y', 'e', 'yt'] + LIRF_FEATS]
    out.parent.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(out, index=False)
    return tr


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stack-dir', default=str(STACK_DIR))
    ap.add_argument('--out', default=str(LIRF_FRAME))
    a = ap.parse_args()
    t = build(Path(a.stack_dir), Path(a.out))
    print(len(t), t.mes.value_counts().to_dict())
