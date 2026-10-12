import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from prc.espera_rot.common import CLIP, FEATS, HCOLS, HIST_EXCLUDE_MONTHS, KEY, RAW_COLS, STACK_DIR, TRAIN_FRAME, TRAINING_ALL, Y_MAX, apt_code
from prc.espera_rot.features import core_features, hist_features, hist_frame


def proxy(stack_dir):
    d = pd.read_parquet(stack_dir / 'dep25.parquet', columns=[KEY, 'y', 'apt', 'mes', 'dia_key'])
    d[KEY] = d[KEY].astype('int64')
    po = pd.read_parquet(stack_dir / 'mlp_dep_oof.parquet')['populacao'].to_numpy()
    pm = np.load(stack_dir / 'perm_oof_topk_mlp1.npy')
    assert len(d) == len(po) == len(pm)
    d['pred'] = 0.5 * po + 0.5 * pm
    d['e'] = d.y - d.pred
    return d.rename(columns={'apt': 'apt_name'})


def build(stack_dir=STACK_DIR, out=TRAIN_FRAME):
    d = proxy(stack_dir)
    d = d[d.y < Y_MAX].reset_index(drop=True)
    x = pd.read_parquet(TRAINING_ALL, columns=RAW_COLS)
    dep_all = x[x.PHASE_mvt == 'DEP']
    ids = set(d[KEY])
    dep = dep_all[dep_all[KEY].astype('int64').isin(ids)]
    arr = x[x.PHASE_mvt == 'ARR']
    core = core_features(dep, arr, arr).drop(columns=['apt_name'])
    h = hist_frame(x, HIST_EXCLUDE_MONTHS)
    hf = hist_features(h, dep)
    yy = (dep.MVT_TIME_UTC_mvt - dep.BLOCK_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    hf.loc[~((yy > 60) & (yy < 5000)), HCOLS] = np.nan
    tr = d.merge(core, on=KEY, how='left').merge(hf, on=KEY, how='left')
    tr = tr[tr.apt_name != 'LIRF'].reset_index(drop=True)
    tr['sl_slack'] = tr.sl_ub - tr.pred
    tr['apt'] = apt_code(tr.apt_name)
    tr['yt'] = tr.e.clip(-CLIP, CLIP)
    cols = [KEY, 'apt_name', 'mes', 'dia_key', 'y', 'e', 'yt'] + FEATS
    tr = tr[cols]
    out.parent.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(out, index=False)
    return tr


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stack-dir', default=str(STACK_DIR))
    ap.add_argument('--out', default=str(TRAIN_FRAME))
    a = ap.parse_args()
    t = build(Path(a.stack_dir), Path(a.out))
    print(len(t), t.mes.value_counts().to_dict())
