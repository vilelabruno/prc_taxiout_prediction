import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from prc.stack.cfg import ARTIFACTS, DEPLOY, KEY, TARGET, ensure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=str(DEPLOY))
    ap.add_argument('--perm-tag', default='_mlp1')
    ap.add_argument('--legacy-months', default='1,7')
    ap.add_argument('--ratio-cut', type=float, default=1.5)
    ap.add_argument('--lemd-cut-months', default='1')
    ap.add_argument('--out', default=str(ARTIFACTS / 'stack_full_pred.parquet'))
    a = ap.parse_args()
    W = Path(a.dir)
    A = pd.read_parquet(W / 'dep25.parquet')
    B = pd.read_parquet(W / 'dep26.parquet')
    po = pd.read_parquet(W / 'mlp_dep_oof.parquet').populacao.to_numpy()
    p26c = pd.read_parquet(W / 'mlp_dep_26.parquet').populacao26_corr.to_numpy()
    perm_oof = np.load(W / f'perm_oof_topk{a.perm_tag}.npy')
    perm26 = np.load(W / f'perm_26_topk{a.perm_tag}.npy')
    ok = ~A.gig.to_numpy(bool)
    hand = A.hand.to_numpy(float)
    oof = 0.5 * po + 0.5 * perm_oof
    v36 = B.v36.to_numpy(float)
    v53 = B.v53.to_numpy(float)
    apt = B.apt.to_numpy()
    mes = B.mes.to_numpy()
    S26 = 0.5 * (v36 + p26c) + 0.5 * perm26
    d26 = S26 - v53
    legacy = np.isin(mes, [int(x) for x in a.legacy_months.split(',') if x])
    safe = np.ones(len(B), bool)
    for ap_ in sorted(set(apt) - {'LIRF'}):
        s25 = (A.apt.to_numpy() == ap_) & ok
        r25 = float(np.sqrt(np.mean((oof - hand)[s25] ** 2)))
        for m_ in sorted(set(mes[~legacy])):
            s = (apt == ap_) & (mes == m_)
            if not s.any():
                continue
            ratio = float(np.sqrt(np.mean(d26[s] ** 2))) / r25
            if ratio > a.ratio_cut:
                safe[s] = False
                print('corta', ap_, m_, round(ratio, 2))
    safe &= ~((apt == 'LEMD') & np.isin(mes, [int(x) for x in a.lemd_cut_months.split(',') if x]))
    use = (apt != 'LIRF') & safe
    final = np.where(use, S26, v53)
    d = final - v53
    print('linhas alteradas %d (%.1f%%) | RMS da mudanca %.2f s | NaN %d' % (int((np.abs(d) > 1e-9).sum()), 100 * np.mean(np.abs(d) > 1e-9), np.sqrt(np.mean(d ** 2)), int(np.isnan(final).sum())))
    out = pd.DataFrame({KEY: B[KEY].astype('int64').to_numpy(), TARGET: final})
    assert out[KEY].is_unique
    ensure(Path(a.out).parent)
    out.to_parquet(a.out, index=False)


if __name__ == '__main__':
    main()
