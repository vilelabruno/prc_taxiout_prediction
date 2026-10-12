import argparse
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from prc.stack.cfg import DEPLOY, STACK_MODELS, ensure

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default=str(DEPLOY))
ap.add_argument('--mlp', action='store_true')
ap.add_argument('--no-nets', action='store_true')
ap.add_argument('--ncombo', type=int, default=60)
ap.add_argument('--ktop', type=int, default=15)
a = ap.parse_args()
W = Path(a.dir)
ensure(STACK_MODELS)
TAG = ('_mlp1' if a.mlp else '') + ('_nonets' if a.no_nets else '')
A = pd.read_parquet(W / 'dep25.parquet')
B = pd.read_parquet(W / 'dep26.parquet')
PREF = ('dA', 'dS4', 'dPool', 'dHand', 'H_base') if a.no_nets else ('dA', 'dS4', 'dPool', 'dHand', 'H_base', 'nn_', 'dCNN') + (('dMLP',) if a.mlp else ())
MEM = [c for c in A.columns if c.startswith(PREF)]
k = len(MEM)
idx = {c: i for i, c in enumerate(MEM)}
X = A[MEM].to_numpy(float)
X26 = B[MEM].to_numpy(float)
y = A.y.to_numpy(float)
b36 = A.b36.to_numpy(float)
v36 = B.v36.to_numpy(float)
gig = A.gig.to_numpy(bool)
ok = ~gig
dia_key = A.dia_key.to_numpy().astype(int)
blk = A.bloco.to_numpy().astype(int)
hand = A.hand.to_numpy(float)
cellA = np.where(A.apt.to_numpy() == 'LIRF', np.where(A.sem_aobt.to_numpy() == 1, 'LIRF_sem', 'LIRF_ok'), A.apt.to_numpy())
cells = sorted(set(cellA))
aidx = np.array([cells.index(c) for c in cellA])
nc = len(cells)
cell26 = np.where(B.apt.to_numpy() == 'LIRF', np.where(B.sem_aobt.to_numpy() == 1, 'LIRF_sem', 'LIRF_ok'), B.apt.to_numpy())
aidx26 = np.array([cells.index(c) for c in cell26])
lirf = A.apt.to_numpy() == 'LIRF'
t0 = time.time()
rng = np.random.RandomState(0)
NCOMBO = a.ncombo
KTOP = a.ktop
ESS = ['dA', 'dS4', 'dPool', 'dHand']
EXTRAS = [c for c in MEM if c not in ESS]
combos = [ESS + EXTRAS, ESS + [c for c in EXTRAS if c.startswith('H_base')], ESS + [c for c in EXTRAS if c.startswith('nn_')], ESS]
while len(combos) < NCOMBO:
    ne = rng.randint(1, len(EXTRAS) + 1)
    combos.append(ESS + list(rng.choice(EXTRAS, ne, replace=False)))
LEARN = [('r1e5', 'ridge', 1e5), ('r1e6', 'ridge', 1e6), ('r1e7', 'ridge', 1e7), ('l.003', 'lasso', 0.003), ('l.01', 'lasso', 0.01), ('l.03', 'lasso', 0.03)]
PRED = [(ci_, ln) for ci_ in range(len(combos)) for ln in LEARN]
nP = len(PRED)
days = np.unique(dia_key)
blocks = sorted(set(blk))
blk_days = {b: np.unique(dia_key[blk == b]) for b in blocks}
SS = {}
for dk in days:
    for c in range(nc):
        sel = ok & (dia_key == dk) & (aidx == c)
        if sel.any():
            SS[(dk, c)] = (X[sel].T @ X[sel], X[sel].T @ (y - b36)[sel])


def purge_set(b):
    dd = blk_days[b]
    mm = dd[0] // 100
    lo, hi = dd.min() - 1, dd.max() + 1
    return set(dk for dk in days if dk // 100 == mm and lo <= dk <= hi)


PS = {b: purge_set(b) for b in blocks}


def acc(tr, c):
    G = np.zeros((k, k))
    g = np.zeros(k)
    for dk in tr:
        if (dk, c) in SS:
            G += SS[(dk, c)][0]
            g += SS[(dk, c)][1]
    return G, g


def lasso_cd(G, g, a_rel, it=40):
    n = len(g)
    s = np.sqrt(np.diag(G)) + 1e-9
    lam = a_rel * np.max(np.abs(g) / s)
    w = np.zeros(n)
    for _ in range(it):
        for j in range(n):
            r = g[j] - G[j] @ w + G[j, j] * w[j]
            w[j] = np.sign(r) * max(abs(r) - lam * s[j], 0.0) / (G[j, j] + 1e-9)
    return w


def fitw(G, g, cols, kind, par):
    ix = [idx[c] for c in cols]
    Gs = G[np.ix_(ix, ix)]
    return (np.linalg.solve(Gs + par * np.eye(len(ix)), g[ix]) if kind == 'ridge' else lasso_cd(Gs, g[ix], par)), ix


def fit_apply(tr, Xr, base_r, ar):
    out = np.zeros((len(base_r), nP), np.float32)
    for c in range(nc):
        s = ar == c
        if not s.any():
            continue
        G, g = acc(tr, c)
        for j, (ci_, (ln, kind, par)) in enumerate(PRED):
            w, ix = fitw(G, g, combos[ci_], kind, par)
            out[s, j] = base_r[s] + Xr[s][:, ix] @ w
    return out


res_topk = np.zeros(len(y))
cvpred = np.zeros((len(y), nP), np.float32)
for b in blocks:
    otr = [dk for dk in days if dk not in PS[b]]
    inner = [bb for bb in blocks if bb != b and not (set(blk_days[bb]) & PS[b])]
    te = np.where(blk == b)[0]
    rows_o = np.concatenate([np.where(blk == bb)[0] for bb in inner])
    oof = np.zeros((len(rows_o), nP), np.float32)
    pos = 0
    for bb in inner:
        rr = np.where(blk == bb)[0]
        tr = [dk for dk in otr if dk not in PS[bb]]
        oof[pos:pos + len(rr)] = fit_apply(tr, X[rr], b36[rr], aidx[rr])
        pos += len(rr)
    pte = fit_apply(otr, X[te], b36[te], aidx[te])
    oo = ok[rows_o]
    inner_r = np.sqrt(np.mean((oof[oo] - y[rows_o][oo, None]) ** 2, axis=0))
    top = np.argsort(inner_r)[:KTOP]
    res_topk[te] = pte[:, top].mean(1)
    print('bloco', b, 'melhor interno %.2f' % inner_r[top[0]], f'{time.time() - t0:.0f}s', flush=True)
    cvpred[te] = pte
cv_r = np.sqrt(np.mean((cvpred[ok] - y[ok, None]) ** 2, axis=0))
topF = np.argsort(cv_r)[:KTOP]
print('FINAL: top-%d por CV em todos os blocos (RMSE %.3f a %.3f)' % (KTOP, cv_r[topF[0]], cv_r[topF[-1]]))
GT = {c: acc(days, c) for c in range(nc)}
p26 = np.zeros((len(B), nP), np.float32)
coefs = {}
for c in range(nc):
    s = aidx26 == c
    for j, (ci_, (ln, kind, par)) in enumerate(PRED):
        if j not in topF:
            continue
        w, ix = fitw(GT[c][0], GT[c][1], combos[ci_], kind, par)
        coefs[(cells[c], int(j))] = {'cols': [MEM[i] for i in ix], 'w': w, 'kind': kind, 'par': par}
        if s.any():
            p26[s, j] = v36[s] + X26[s][:, ix] @ w
top26 = p26[:, topF].mean(1)


def r(v, s=None):
    return float(np.sqrt(np.mean((v[s] - y[s]) ** 2))) if s is not None else float(np.sqrt(np.mean((v[ok] - y[ok]) ** 2)))


print('OOF aninhado, top-%d: sem_gig %.3f | fora do LIRF %.3f | LIRF %.3f   (mao: %.3f | %.3f | %.3f)' % (KTOP, r(res_topk), r(res_topk, ok & ~lirf), r(res_topk, ok & lirf), r(hand), r(hand, ok & ~lirf), r(hand, ok & lirf)))
np.save(W / f'perm_oof_topk{TAG}.npy', res_topk)
np.save(W / f'perm_26_topk{TAG}.npy', top26)
pickle.dump({'members': MEM, 'cells': cells, 'top': [int(j) for j in topF], 'pred_spec': [(int(ci_), ln) for ci_, ln in PRED], 'combos': combos, 'coefs': coefs}, open(STACK_MODELS / f'perm_coefs{TAG}.pkl', 'wb'))
print('2026: RMS da correcao do stack-permutacoes sobre v36 (fora do LIRF) %.1f | 2025 OOF %.1f' % (np.sqrt(np.mean((top26 - v36)[B.apt.to_numpy() != 'LIRF'] ** 2)), np.sqrt(np.mean((res_topk - b36)[ok & ~lirf] ** 2))))
