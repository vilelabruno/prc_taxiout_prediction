import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from prc.stack.cfg import DEPLOY, STACK_MODELS, ensure

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default=str(DEPLOY))
ap.add_argument('--device', default='cpu', choices=['cuda', 'mps', 'cpu'])
a = ap.parse_args()
dev = a.device
S = Path(a.dir)
ensure(STACK_MODELS)
A = pd.read_parquet(S / 'dep25.parquet')
B = pd.read_parquet(S / 'dep26.parquet')
MEM = [c for c in A.columns if c.startswith(('dA', 'dS4', 'dPool', 'dHand', 'H_base', 'nn_', 'dCNN'))]
k = len(MEM)
CTX = [c for c in A.columns if c.startswith('c_')]
nctx = len(CTX)
y = A.y.to_numpy(float)
b36 = A.b36.to_numpy(float)
gig = A.gig.to_numpy(bool)
ok = ~gig
hand = A.hand.to_numpy(float)
dia_key = A.dia_key.to_numpy().astype(int)
blk = A.bloco.to_numpy().astype(int)
lirf = A.apt.to_numpy() == 'LIRF'
cell = np.where(A.apt.to_numpy() == 'LIRF', np.where(A.sem_aobt.to_numpy() == 1, 'LIRF_sem', 'LIRF_ok'), A.apt.to_numpy())
cells = sorted(set(cell))
cid = np.array([cells.index(c) for c in cell])
nc = len(cells)
cell26 = np.where(B.apt.to_numpy() == 'LIRF', np.where(B.sem_aobt.to_numpy() == 1, 'LIRF_sem', 'LIRF_ok'), B.apt.to_numpy())
cid26 = np.array([cells.index(c) for c in cell26])
Xm = np.concatenate([A[MEM].to_numpy(np.float32), B[MEM].to_numpy(np.float32)]) / 100.0
mu = A[CTX].mean().to_numpy()
sd_ = A[CTX].std().to_numpy() + 1e-6
ctxf = np.clip((np.concatenate([A[CTX].to_numpy(float), B[CTX].to_numpy(float)]) - mu) / sd_, -6, 6).astype(np.float32)
CID = np.concatenate([cid, cid26])
N25 = len(A)
N26 = len(B)
tgt = np.concatenate([(np.clip(y - b36, -3000, 3000) / 100.0), np.zeros(N26)]).astype(np.float32)
days = np.unique(dia_key)
blocks = sorted(set(blk))
blk_days = {b: np.unique(dia_key[blk == b]) for b in blocks}


def purge_set(b):
    dd = blk_days[b]
    mm = dd[0] // 100
    lo, hi = dd.min() - 1, dd.max() + 1
    return set(dk for dk in days if dk // 100 == mm and lo <= dk <= hi)


class Net(nn.Module):
    def __init__(s, eps, add, sigma, drop):
        super().__init__()
        s.eps = eps
        s.add = add
        s.sigma = sigma
        s.g = nn.Parameter(torch.zeros(k))
        s.W = nn.Parameter(torch.zeros(nc, k))
        s.emb = nn.Embedding(nc, 8)
        s.dw = nn.Sequential(nn.Linear(nctx + 8, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, k))
        nn.init.zeros_(s.dw[-1].weight)
        nn.init.zeros_(s.dw[-1].bias)
        s.mlp = nn.Sequential(nn.Linear(k + nctx + 8, 192), nn.GELU(), nn.Dropout(drop), nn.Linear(192, 96), nn.GELU(), nn.Dropout(drop), nn.Linear(96, 1))
        nn.init.zeros_(s.mlp[-1].weight)
        nn.init.zeros_(s.mlp[-1].bias)

    def forward(s, xm, xc, ci):
        e = s.emb(ci)
        z = xm
        if s.training and s.sigma > 0:
            z = z + s.sigma * torch.randn_like(z) * (xm.abs() > 0).float()
        out = (z * (s.g + s.W[ci])).sum(1)
        if s.eps > 0:
            out = out + (z * s.eps * torch.tanh(s.dw(torch.cat([xc, e], 1)))).sum(1)
        if s.add:
            out = out + s.mlp(torch.cat([z, xc, e], 1)).squeeze(1)
        return out


def fit(tr, va, te, cfg, seed, lam_lin=3.3e-4, ep=40, pat=5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(cfg['eps'], cfg['add'], cfg['sigma'], cfg['drop']).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)

    def f(arr, ix):
        return torch.from_numpy(arr[ix]).to(dev)

    Xt, Ct, It, Yt = f(Xm, tr), f(ctxf, tr), torch.from_numpy(CID[tr]).to(dev), f(tgt, tr)
    Xv, Cv, Iv, Yv = f(Xm, va), f(ctxf, va), torch.from_numpy(CID[va]).to(dev), f(tgt, va)
    best, bst, bad = 1e18, None, 0
    n = len(tr)
    for e in range(ep):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 4096):
            ix = perm[i:i + 4096]
            p = net(Xt[ix], Ct[ix], It[ix])
            loss = ((p - Yt[ix]) ** 2).mean() + lam_lin * (net.W ** 2).sum() + 1e-5 * (net.g ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(((net(Xv, Cv, Iv) - Yv) ** 2).mean().sqrt())
        if v < best - 1e-4:
            best, bst, bad = v, {kk: vv.clone() for kk, vv in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= pat:
                break
    net.load_state_dict(bst)
    net.eval()
    with torch.no_grad():
        pred = net(f(Xm, te), f(ctxf, te), torch.from_numpy(CID[te]).to(dev)).cpu().numpy() * 100.0
    return pred, e + 1, {kk: vv.cpu() for kk, vv in bst.items()}


VARS = [('V1 dw(ctx) eps 0.1, ruido 0.1', dict(eps=0.1, add=False, sigma=0.1, drop=0.1), 2),
        ('V2 dw(ctx) eps 0.3, ruido 0.1', dict(eps=0.3, add=False, sigma=0.1, drop=0.1), 2),
        ('V3 dw eps 0.1 + MLP aditivo, ruido 0.2', dict(eps=0.1, add=True, sigma=0.2, drop=0.3), 2),
        ('V4 so MLP aditivo, ruido 0.3, dropout 0.5', dict(eps=0.0, add=True, sigma=0.3, drop=0.5), 2)]
t0 = time.time()
preds = {v[0]: np.zeros(N25) for v in VARS}
p26 = {v[0]: np.zeros(N26) for v in VARS}
print('dispositivo', dev, '| membros', k, '| contexto', nctx, flush=True)
for b in blocks:
    ps = purge_set(b)
    te = np.where(blk == b)[0]
    pm = np.array([dk in ps for dk in dia_key])
    inner = [bb for bb in blocks if bb != b and not (set(blk_days[bb]) & ps)]
    vb = set(inner[::4])
    trmask = ~(blk == b) & ~pm & ~gig
    isv = np.isin(blk, list(vb)) & trmask
    tr = np.where(trmask & ~isv)[0]
    va = np.where(isv)[0]
    for nm, cfg, ns in VARS:
        pp = [fit(tr, va, te, cfg, sd)[0] for sd in range(ns)]
        preds[nm][te] = b36[te] + np.mean(pp, 0)
    print('bloco', b, f'{time.time() - t0:.0f}s', flush=True)
vb = [11, 14, 72]
isv = np.isin(blk, vb) & ~gig
tr = np.where(~np.isin(blk, vb) & ~gig)[0]
va = np.where(isv)[0]
te26 = np.arange(N25, N25 + N26)
states = {}
for vi, (nm, cfg, ns) in enumerate(VARS):
    pp = []
    for sd in range(3):
        p, _, st = fit(tr, va, te26, cfg, sd)
        pp.append(p)
        states[f'v{vi}_s{sd}'] = st
    p26[nm] = np.mean(pp, 0)
    print('final', nm, 'rms da correcao 2026 (fora LIRF): %.1f' % np.sqrt(np.mean(p26[nm][B.apt.to_numpy() != 'LIRF'] ** 2)), flush=True)
torch.save({'states': states, 'vars': [(v[0], v[1]) for v in VARS], 'members': MEM, 'ctx': CTX, 'cells': cells, 'ctx_mu': mu, 'ctx_sd': sd_}, STACK_MODELS / 'mlp_dep_final.pt')
pop = np.mean([preds[v[0]] for v in VARS], axis=0)
pop26 = np.mean([p26[v[0]] for v in VARS], axis=0)


def r(v, s=None):
    return float(np.sqrt(np.mean((v[s] - y[s]) ** 2))) if s is not None else float(np.sqrt(np.mean((v[ok] - y[ok]) ** 2)))


for nm, v in list(preds.items()) + [('POPULACAO', pop)]:
    print('%-44s sem_gig %.3f | fora LIRF %.3f | LIRF %.3f' % (nm, r(v), r(v, ok & ~lirf), r(v, ok & lirf)))
print('pilha a mao: sem_gig %.3f | fora LIRF %.3f' % (r(hand), r(hand, ok & ~lirf)))
pd.DataFrame({**{v: preds[v] for v in preds}, 'populacao': pop}).to_parquet(S / 'mlp_dep_oof.parquet', index=False)
pd.DataFrame({'populacao26_corr': pop26}).to_parquet(S / 'mlp_dep_26.parquet', index=False)
