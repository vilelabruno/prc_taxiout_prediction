import sys, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from prc.stack.cfg import MLP_OUT, MLP_UNI, STACK_MODELS, ensure
A = dict(a.split('=') for a in sys.argv[1:])
mode = A.get('mode', 'oof'); loss_k = A.get('loss', 'huber'); E = int(A.get('epochs', 12)); seeds = int(A.get('seeds', 5))
W = int(A.get('width', 1024)); drop = float(A.get('drop', 0.2)); lr = float(A.get('lr', 6e-4)); wd = float(A.get('wd', 1e-2))
bs = int(A.get('bs', 4096)); delta = float(A.get('delta', 4.0)); U = A.get('dir', str(MLP_UNI)).rstrip('/') + '/'; tag = A.get('tag', 'final'); ensure(A.get('out', MLP_OUT), STACK_MODELS)
OUT = A.get('out', str(MLP_OUT)).rstrip('/') + '/' + tag
dev = A.get('device', 'cuda'); dtype_dev = dev.split(':')[0]
X = np.load(U + 'X_all.npy', mmap_mode='r'); XR = np.load(U + 'X_rank.npy', mmap_mode='r')
y = np.load(U + 'y_all.npy'); mo = np.load(U + 'month_all.npy')
ids = np.load(U + 'ids_all.npy', allow_pickle=True); idr = np.load(U + 'ids_rank.npy', allow_pickle=True)
ca = pd.read_parquet(U + 'cat_all.parquet'); cr = pd.read_parquet(U + 'cat_rank.parquet')
excl = (1, 7) if mode == 'oof' else (1, 7, 4, 10)
tr = ~np.isin(mo, excl); ev = np.isin(mo, (4, 10)) if mode == 'val' else np.isin(mo, (1, 7))
print(mode, loss_k, 'train', tr.sum(), 'eval', ev.sum(), flush=True)

rng = np.random.RandomState(0); sidx = np.sort(rng.choice(np.where(tr)[0], 300000, replace=False))
S = torch.tensor(np.asarray(X[sidx]))
lo = torch.nanquantile(S, 0.005, dim=0); hi = torch.nanquantile(S, 0.995, dim=0)
Sc = torch.minimum(torch.maximum(S, lo), hi)
mu = torch.nanmean(Sc, 0); sd = torch.sqrt(torch.nanmean((Sc - mu) ** 2, 0)).clamp_min(1e-6)
mr = torch.isnan(S).float().mean(0); flag = ((mr > 0.005) & (mr < 0.995)).nonzero().squeeze(1)
del S, Sc
print('numericas', X.shape[1], 'flags', len(flag), flush=True)

def prep(arr, n):
    outs = []
    for i in range(0, n, 200000):
        c = torch.tensor(np.asarray(arr[i:i + 200000])).to(dev)
        m = torch.isnan(c)
        z = ((torch.minimum(torch.maximum(c, lo.to(dev)), hi.to(dev)) - mu.to(dev)) / sd.to(dev))
        z = torch.where(m, torch.zeros_like(z), z)
        outs.append(torch.cat([z, m[:, flag.to(dev)].float()], 1).half())
    return torch.cat(outs)

XT = prep(X, len(y)); XRT = prep(XR, len(idr)); nin = XT.shape[1]
cn = list(ca.columns); vocab = {}
CT, CRT = [], []
for c in cn:
    vc = ca[c][tr].astype(str).value_counts(); keep = vc[vc >= 20].index
    vocab[c] = {k: i + 1 for i, k in enumerate(keep)}
    CT.append(torch.tensor(ca[c].astype(str).map(vocab[c]).fillna(0).astype(int).values))
    CRT.append(torch.tensor(cr[c].astype(str).map(vocab[c]).fillna(0).astype(int).values))
CT = torch.stack(CT, 1).to(dev); CRT = torch.stack(CRT, 1).to(dev)
card = [len(vocab[c]) + 1 for c in cn]; edim = [min(48, max(4, int(round(1.6 * k ** 0.56)))) for k in card]
iapt = cn.index('ADEP_mvt'); napt = card[iapt]
print('cats', dict(zip(cn, card)), flush=True)
mo_t = torch.tensor(mo.astype(np.int64)).to(dev)
SC = 600.0
yc = np.minimum(y, 3600.0) if loss_k == 'mse' else y
YT = torch.tensor(yc / SC, dtype=torch.float32).to(dev); Y = y

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.ModuleList([nn.Embedding(k, d) for k, d in zip(card, edim)])
        s.inp = nn.Sequential(nn.Linear(nin + sum(edim), W), nn.LayerNorm(W), nn.SiLU())
        s.blocks = nn.ModuleList([nn.Sequential(nn.Linear(W, W), nn.LayerNorm(W), nn.SiLU(), nn.Dropout(drop), nn.Linear(W, W)) for _ in range(3)])
        s.norm = nn.LayerNorm(W); s.out = nn.Linear(W, 1); s.aph = nn.Linear(W, napt)
    def forward(s, x, c):
        h = torch.cat([x.float()] + [e(c[:, i]) for i, e in enumerate(s.emb)], 1)
        h = s.inp(h)
        for b in s.blocks: h = h + b(F.silu(h))
        h = s.norm(h)
        return s.out(h).squeeze(1) + s.aph(h).gather(1, c[:, iapt:iapt + 1]).squeeze(1)

def predict(net, xt, ct, idx):
    net.eval(); o = []
    with torch.no_grad(), torch.autocast(dtype_dev, dtype=torch.bfloat16):
        for i in range(0, len(idx), 16384):
            j = idx[i:i + 16384]; o.append(net(xt[j], ct[j]).float())
    return torch.cat(o).cpu().numpy() * SC

def rm(p, yy):
    e = p - yy; g = yy <= 50000
    return float(np.sqrt((e ** 2).mean())), float(np.sqrt((e[g] ** 2).mean()))

tri = torch.tensor(np.where(tr)[0]).to(dev); evi = torch.tensor(np.where(ev)[0]).to(dev); evn = np.where(ev)[0]
res = {'ho': [], 'rk': []}; curves = []
for sd_ in range(seeds):
    torch.manual_seed(sd_); net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps = E * (len(tri) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    t0 = time.time(); cur = []
    for ep in range(E):
        net.train(); perm = tri[torch.randperm(len(tri), device=dev)]; tl = 0.0
        for k in range(len(tri) // bs):
            j = perm[k * bs:(k + 1) * bs]
            with torch.autocast(dtype_dev, dtype=torch.bfloat16):
                p = net(XT[j], CT[j]).float()
            l = F.huber_loss(p, YT[j], delta=delta) if loss_k == 'huber' else F.mse_loss(p, YT[j])
            opt.zero_grad(set_to_none=True); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sch.step(); tl += l.item() if k % 50 == 0 else 0
        pe = predict(net, XT, CT, evi); r = rm(pe, Y[evn]); cur.append(r)
        print(f'seed{sd_} ep{ep + 1} loss {tl:.1f} eval RMSE {r[0]:.2f} (sem gig {r[1]:.2f}) {time.time() - t0:.0f}s', flush=True)
    curves.append(cur); res['ho'].append(pe)
    res['rk'].append(predict(net, XRT, CRT, torch.arange(len(idr), device=dev)))
    torch.save({'state': {kk: vv.cpu() for kk, vv in net.state_dict().items()}, 'lo': lo, 'hi': hi, 'mu': mu, 'sd': sd, 'flag': flag, 'vocab': vocab, 'cols': cn, 'width': W, 'drop': drop, 'nin': nin, 'card': card, 'edim': edim}, f'{STACK_MODELS}/mlp_member_{tag}_seed{sd_}.pt')
ho = np.mean(res['ho'], 0); rk = np.mean(res['rk'], 0)
print('ENSEMBLE eval RMSE', rm(ho, Y[evn]), flush=True)
np.savez(OUT + '.npz', ids_ho=ids[evn], ho=ho, ids_rk=idr, rk=rk, ho_seeds=np.array(res['ho']), rk_seeds=np.array(res['rk']), curves=np.array(curves))
