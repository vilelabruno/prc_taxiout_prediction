import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from prc.stack.cfg import ARTIFACTS, DEPLOY, KEY, TARGET, ensure

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default=str(DEPLOY))
ap.add_argument('--perm-tag', default='')
ap.add_argument('--out', default=str(ARTIFACTS / 'stack_pred.parquet'))
ap.add_argument('--ratio-cut', type=float, default=1.5)
ap.add_argument('--lemd-cut-months', default='1')
a = ap.parse_args()
W = Path(a.dir)
A = pd.read_parquet(W / 'dep25.parquet')
B = pd.read_parquet(W / 'dep26.parquet')
po = pd.read_parquet(W / 'mlp_dep_oof.parquet').populacao.to_numpy()
p26c = pd.read_parquet(W / 'mlp_dep_26.parquet').populacao26_corr.to_numpy()
perm_oof = np.load(W / f'perm_oof_topk{a.perm_tag}.npy')
perm26 = np.load(W / f'perm_26_topk{a.perm_tag}.npy')
y = A.y.to_numpy(float)
ok = ~A.gig.to_numpy(bool)
hand = A.hand.to_numpy(float)
lirf = A.apt.to_numpy() == 'LIRF'
oof = 0.5 * po + 0.5 * perm_oof


def r(v, s):
    return float(np.sqrt(np.mean((v[s] - y[s]) ** 2)))


print('OOF aninhado: fora do LIRF  mao %.3f | permutacoes %.3f | redes %.3f | MISTURA 50/50 %.3f' % (r(hand, ok & ~lirf), r(perm_oof, ok & ~lirf), r(po, ok & ~lirf), r(oof, ok & ~lirf)))
h = np.where(lirf, hand, oof)
print('total sem gig (LIRF = mao): mao %.3f -> mistura %.3f' % (r(hand, ok), r(h, ok)))
v36 = B.v36.to_numpy(float)
v53 = B.v53.to_numpy(float)
apt26 = B.apt.to_numpy()
mes26 = B.mes.to_numpy()
S26 = 0.5 * (v36 + p26c) + 0.5 * perm26
d26 = S26 - v53
safe = np.ones(len(B), bool)
for ap_ in sorted(set(apt26) - {'LIRF'}):
    s25 = (A.apt.to_numpy() == ap_) & ok
    s26 = apt26 == ap_
    d25 = (oof - hand)[s25]
    r25 = float(np.sqrt(np.mean(d25 ** 2)))
    r26 = float(np.sqrt(np.mean(d26[s26] ** 2)))
    ratio = r26 / r25
    cut = ratio > a.ratio_cut
    if cut:
        safe[s26] = False
    print('  %-5s %7.2f  %7.2f   %.2fx  %s' % (ap_, r25, r26, ratio, 'CORTA (mantem pilha a mao)' if cut else 'ok'))
cut_months = [int(x) for x in a.lemd_cut_months.split(',') if x]
safe &= ~((apt26 == 'LEMD') & np.isin(mes26, cut_months))
use26 = (apt26 != 'LIRF') & safe
final = np.where(use26, S26, v53)
d = final - v53
print('linhas alteradas %d (%.1f%%) | RMS da mudanca %.2f s | NaN %d | min %.1f' % (int((np.abs(d) > 1e-9).sum()), 100 * np.mean(np.abs(d) > 1e-9), np.sqrt(np.mean(d ** 2)), int(np.isnan(final).sum()), final.min()))
out = pd.DataFrame({KEY: B[KEY].to_numpy(), TARGET: final})
assert out[KEY].is_unique
dst = Path(a.out)
ensure(dst.parent)
out.to_parquet(dst, index=False)
print(dst.name, 'sha256', hashlib.sha256(dst.read_bytes()).hexdigest()[:16])
