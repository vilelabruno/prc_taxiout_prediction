import argparse
import numpy as np
import pandas as pd
from prc.paths import ARTIFACTS, WORK
from prc.basea import residual_chain as rc
from prc.basea.residual_chain_stack_adsb import clock_overlay

CACHE = ARTIFACTS / 'base_a_full_final'
MATRICES = WORK / 'basea' / 'matrices'
REQUIRED = ['light70', 'light123', 'light153', 'specialist_single', 'specialist_five', 'specialist_arrival',
            'airport_experts', 'cat70', 'cat93', 'cat153']


def assemble(cache, rdep, rx, upto='v12'):
    offset = rc.baseline(rx).to_numpy()
    lirf = rc.lirf_scope(rx).to_numpy()
    ordinary = ~lirf

    def load(name):
        return np.load(cache / f'sub_{name}.npy')

    def residual(name):
        return np.maximum(0.0, load(name) + offset)

    single = load('specialist_single')
    v4 = 0.25 * np.maximum(0.0, residual('light70')) + 0.75 * np.maximum(0.0, residual('cat70'))
    v4[lirf] = single[lirf]
    v5 = 0.25 * v4 + 0.75 * np.maximum(0.0, residual('cat93'))
    v5[lirf] = single[lirf]
    v6 = v5.copy()
    five = load('specialist_five')
    v6[lirf] = five[lirf]
    v7 = 0.25 * np.maximum(0.0, residual('light123')) + 0.75 * v6
    arrival = load('specialist_arrival')
    v7[lirf] = 0.5 * arrival[lirf] + 0.5 * v6[lirf]
    light153 = residual('light153')
    experts = load('airport_experts')
    experts = np.where(np.isnan(experts), np.maximum(0.0, light153), experts)
    combined = 0.5 * experts + 0.5 * np.maximum(0.0, light153)
    cat153 = residual('cat153')
    v10 = v7.copy()
    v10[ordinary] = (0.375 * v7 + 0.25 * combined + 0.375 * np.maximum(0.0, cat153))[ordinary]
    v11, triggered = clock_overlay(rdep, rx, v10)
    if upto == 'v11':
        return v11, triggered
    protected = lirf | (v11 != v10)
    v12 = v11.copy()
    change = 0.375 * (np.maximum(0.0, residual('cat183')) - np.maximum(0.0, cat153))
    v12[~protected] = (v11 + change)[~protected]
    return v12, triggered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=str(CACHE))
    ap.add_argument('--matrices', default=str(MATRICES))
    ap.add_argument('--upto', choices=['v11', 'v12'], default='v12')
    ap.add_argument('--out', default=str(CACHE / 'base_a_full_final.parquet'))
    a = ap.parse_args()
    cache = pd.io.common.stringify_path(a.cache)
    from pathlib import Path
    cache = Path(cache)
    mats = Path(a.matrices)
    rdep = pd.read_parquet(mats / 'rdep.parquet')
    rx = pd.read_parquet(mats / 'rx.parquet')
    pred, triggered = assemble(cache, rdep, rx, a.upto)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    out = pd.DataFrame({rc.ID: rdep[rc.ID].to_numpy(), rc.TARGET: pred})
    out[rc.ID] = out[rc.ID].astype('int64')
    assert out[rc.ID].is_unique
    out.to_parquet(a.out, index=False)
    print(len(out), triggered, a.upto, float(out[rc.TARGET].mean()))


if __name__ == '__main__':
    main()
