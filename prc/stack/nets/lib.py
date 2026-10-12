import time
import numpy as np, pandas as pd
from prc.stack.cfg import HARM_DIR

OUT = HARM_DIR
AIRC = {"EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4706, -0.4619), "EHAM": (52.3086, 4.7639),
        "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626), "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389),
        "LTFM": (41.2753, 28.7519), "LSZH": (47.4647, 8.5492)}
S4A = ['EDDF', 'EDDM', 'EHAM', 'LSZH']
NMAX = 192
CH = ['trel', 'dtp', 'gs', 'air', 'dsx', 'dsy', 'dstand', 'dthr', 'sinh', 'cosh', 'spd', 'stat', 'valid']


def load_static(apts=S4A):
    d = pd.read_parquet(OUT / 'nn_static.parquet')
    d = d[d.ADEP_mvt.isin(apts) & (d.adsb_cov == 1) & d.mk].reset_index(drop=True)
    d['S4'] = d.ADEP_mvt.isin(S4A)
    return d


RC = None


def runway_centroids():
    import pickle
    return pickle.load(open(OUT / 'runway_centroids.pkl', 'rb'))[0]


def xy(ap, lat, lon):
    la0, lo0 = AIRC[ap]
    return (np.asarray(lon, float) - lo0) * np.cos(np.radians(la0)) * 111320.0, (np.asarray(lat, float) - la0) * 110540.0


def build_seq(d, year):
    z = np.load(OUT / f'seq_{year}.npz'); ids, off, pts = z['ids'], z['off'], z['pts']
    pos = {int(i): k for k, i in enumerate(ids)}
    cent = runway_centroids(); apt_c = {}
    for (ap, r), (cx_, cy_) in cent.items(): apt_c.setdefault(ap, []).append((cx_, cy_))
    n = len(d); X = np.zeros((n, NMAX, len(CH)), np.float16); has = np.zeros(n, bool)
    sx, sy = np.zeros(n), np.zeros(n)
    for ap in d.ADEP_mvt.unique():
        k = (d.ADEP_mvt == ap).to_numpy(); a, b = xy(ap, d.sla[k].to_numpy(float), d.slo[k].to_numpy(float)); sx[k], sy[k] = a, b
    ADEP = d.ADEP_mvt.to_numpy(); mid = d.MVT_ID_mvt.to_numpy()
    for i in range(n):
        j = pos.get(int(mid[i]))
        if j is None: continue
        p = pts[off[j]:off[j + 1]]; m = len(p)
        t, x, y, gs, alt = p[:, 0].astype(np.float64), p[:, 1].astype(np.float64), p[:, 2].astype(np.float64), np.nan_to_num(p[:, 3], nan=0.0), p[:, 4]
        dt = np.r_[0, np.diff(t)]; dxy = np.hypot(np.r_[0, np.diff(x)], np.r_[0, np.diff(y)]); spd = np.where(dt > 0, dxy / np.maximum(dt, 1e-3), 0)
        hd = np.arctan2(np.r_[0, np.diff(x)], np.r_[0, np.diff(y)])
        c = apt_c.get(ADEP[i], [(0, 0)]); dth = np.min([np.hypot(x - a, y - b) for a, b in c], axis=0)
        dsx, dsy = x - sx[i], y - sy[i]; dstd = np.hypot(dsx, dsy)
        f = np.c_[t / 600.0, np.log1p(dt) / 3, gs / 20.0, (alt > 0).astype(float), np.sign(dsx) * np.log1p(np.abs(dsx)) / 6, np.sign(dsy) * np.log1p(np.abs(dsy)) / 6, np.log1p(dstd) / 6, np.log1p(dth) / 6,
                  np.sin(hd), np.cos(hd), np.minimum(spd, 40) / 10, (gs < 1.0).astype(float), np.ones(m)]
        f = np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), -12, 12); f[:, -1] = 1.0; X[i, :m] = f.astype(np.float16); has[i] = True
    return X, has


STATIC_SKIP = {'MVT_ID_mvt', 'ADEP_mvt', 'mes', 'bloco', 'dia', 'TAXITIME_SEC_mvt', 'STAND_mvt', 'sla', 'slo', 'RUNWAY_mvt', 'year', 'src', 'mk', 'S4', 'qb2', 'v45', 'apt', 'car'}


def static_cols(d):
    return [c for c in d.columns if c not in STATIC_SKIP]


def prep_static(d, cols, ref):
    A = d[cols].to_numpy(np.float64); R = ref[cols].to_numpy(np.float64)
    med = np.nanmedian(R, 0); q1, q3 = np.nanpercentile(R, 25, 0), np.nanpercentile(R, 75, 0); sc = np.where((q3 - q1) > 1e-6, q3 - q1, np.nanstd(R, 0) + 1e-6)
    miss = np.isnan(R).mean(0) > 0.005
    Z = np.clip((np.where(np.isnan(A), med, A) - med) / sc, -6, 6)
    return np.c_[Z, np.isnan(A[:, miss]).astype(float)].astype(np.float32)


def get_torch():
    import torch, torch.nn as nn
    return torch, nn


def make_net(kind, ds, nch=len(CH), width=64, drop=0.1, n_apt=4, layers=2):
    torch, nn = get_torch()

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.kind = kind; s.st = nn.Sequential(nn.Linear(ds, 2 * width), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * width, 2 * width), nn.GELU())
            s.ap = nn.Embedding(n_apt, 8)
            if kind == 'cnn':
                L = [nn.Conv1d(nch, width, 5, padding=2), nn.GELU()]
                for _ in range(layers + 1): L += [nn.Conv1d(width, width, 5, padding=2, stride=2), nn.GroupNorm(8, width), nn.GELU(), nn.Dropout(drop)]
                s.enc = nn.Sequential(*L); so = 2 * width
            elif kind in ('cgru', 'ctr'):
                s.stem = nn.Sequential(nn.Conv1d(nch, width, 5, padding=2), nn.GELU(), nn.Conv1d(width, width, 5, padding=2, stride=2), nn.GroupNorm(8, width), nn.GELU(), nn.Conv1d(width, width, 5, padding=2, stride=2), nn.GroupNorm(8, width), nn.GELU())
                if kind == 'cgru': s.enc = nn.GRU(width, width, num_layers=layers, batch_first=True, bidirectional=True, dropout=drop if layers > 1 else 0.0); so = 4 * width
                else: s.pos = nn.Parameter(torch.zeros(1, 64, width)); el = nn.TransformerEncoderLayer(width, 4, 2 * width, drop, batch_first=True, norm_first=True, activation='gelu'); s.enc = nn.TransformerEncoder(el, layers); so = 2 * width
            elif kind == 'gru':
                s.proj = nn.Linear(nch, width); s.enc = nn.GRU(width, width, num_layers=layers, batch_first=True, bidirectional=True, dropout=drop if layers > 1 else 0.0); so = 4 * width
            elif kind == 'tr':
                s.proj = nn.Linear(nch, width); el = nn.TransformerEncoderLayer(width, 4, 2 * width, drop, batch_first=True, norm_first=True, activation='gelu'); s.enc = nn.TransformerEncoder(el, layers); so = 2 * width
            else: so = 0
            s.head = nn.Sequential(nn.Linear(2 * width + so + 8, 2 * width), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * width, 1))
            s.dom = nn.Sequential(nn.Linear(2 * width + so, 64), nn.GELU(), nn.Linear(64, 2))

        def forward(s, xs, xq, apt, ret_emb=False):
            h = [s.st(xs), s.ap(apt)]
            if s.kind != 'mlp':
                m = xq[:, :, -1] > 0
                if s.kind == 'cnn':
                    z = s.enc(xq.transpose(1, 2)); h.append(torch.cat([z.mean(2), z.amax(2)], 1))
                elif s.kind in ('cgru', 'ctr'):
                    z = s.stem(xq.transpose(1, 2)).transpose(1, 2)
                    m4 = torch.nn.functional.max_pool1d(m.float().unsqueeze(1), 4, 4).squeeze(1) > 0; mf = m4.unsqueeze(-1).float()
                    if s.kind == 'cgru': z, _ = s.enc(z)
                    else: z = s.enc(z + s.pos[:, :z.shape[1]], src_key_padding_mask=~m4)
                    h.append(torch.cat([(z * mf).sum(1) / mf.sum(1).clamp(min=1), (z + (mf - 1) * 1e4).amax(1)], 1))
                elif s.kind == 'gru':
                    z, _ = s.enc(torch.nn.functional.gelu(s.proj(xq))); mf = m.unsqueeze(-1).float(); h.append(torch.cat([(z * mf).sum(1) / mf.sum(1).clamp(min=1), (z + (mf - 1) * 1e4).amax(1)], 1))
                else:
                    z = s.enc(s.proj(xq), src_key_padding_mask=~m); mf = m.unsqueeze(-1).float(); h.append(torch.cat([(z * mf).sum(1) / mf.sum(1).clamp(min=1), (z + (mf - 1) * 1e4).amax(1)], 1))
            e = torch.cat(h, 1); o = s.head(e).squeeze(-1)
            return (o, torch.cat([h[0]] + h[2:], 1)) if ret_emb else o
    return Net()


def grl_fn():
    torch, nn = get_torch()
    class GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
        @staticmethod
        def backward(ctx, g): return -ctx.lam * g, None
    return GRL.apply


def train_nn(kind, Xs, Xq, apt, tgt, tr, va, te, soft=None, alpha=1.0, dann=0.0, tgt_idx=None, epochs=12, bs=512, lr=2e-3, wd=1e-2, drop=0.1, width=64, layers=2, huber=None, seed=0, aug=0.0, dev=None, verbose=True, threads=4, tq=None):
    torch, nn = get_torch(); torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(threads)
    dev = dev or 'cpu'
    net = make_net(kind, Xs.shape[1], width=width, drop=drop, layers=layers).to(dev); GRL = grl_fn()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd); steps = epochs * int(np.ceil(len(tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    Ts = torch.tensor(Xs).to(dev); Tq = tq if tq is not None else (torch.tensor(Xq).to(dev) if Xq is not None else None); Ta = torch.tensor(apt, dtype=torch.long).to(dev); Ty = torch.tensor((np.nan_to_num(tgt) / 100.0).astype(np.float32)).to(dev); Tm = torch.tensor((~np.isnan(tgt)).astype(np.float32)).to(dev)
    Tsf = torch.tensor((np.nan_to_num(soft) / 100.0).astype(np.float32)).to(dev) if soft is not None else None
    def batch(ix, train=False):
        q = None; ix = torch.as_tensor(ix, device=dev)
        if Tq is not None:
            q = Tq[ix].float()
            if train and aug > 0:
                keep = (torch.rand(q.shape[0], q.shape[1], device=dev) > aug).float().unsqueeze(-1); q = q * keep
        return Ts[ix], q, Ta[ix], Ty[ix]
    def sbatch(ix):
        ix = torch.as_tensor(ix, device=dev); return Tm[ix], (Tsf[ix] if Tsf is not None else None)
    def predict(ix):
        net.eval(); out = []
        with torch.no_grad():
            for k in range(0, len(ix), 2048):
                a, q, p, _ = batch(ix[k:k + 2048]); out.append(net(a, q, p).float().cpu().numpy())
        return np.concatenate(out) * 100.0
    best, best_state, hist = 1e18, None, []
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(tr); tl = 0; nb = 0; t0 = time.time()
        for k in range(0, len(perm), bs):
            ix = perm[k:k + bs]
            if len(ix) < 8: continue
            a, q, p, y = batch(ix, True)
            if dann > 0 and tgt_idx is not None:
                import math
                prog = (ep * len(perm) + k) / (epochs * len(perm)); lam = dann * (2.0 / (1.0 + math.exp(-10 * prog)) - 1.0)
                pr, e_s = net(a, q, p, ret_emb=True); jx = np.random.choice(tgt_idx, len(ix)); a2, q2, p2, _ = batch(jx); _, e_t = net(a2, q2, p2, ret_emb=True)
                dl = nn.functional.cross_entropy(net.dom(GRL(torch.cat([e_s, e_t]), lam)), torch.cat([torch.zeros(len(ix)), torch.ones(len(jx))]).long().to(dev))
            else: pr = net(a, q, p); dl = 0.0
            if soft is not None:
                mk_, sf_ = sbatch(ix); loss = (alpha * mk_ * (pr - y) ** 2 + (1 - alpha) * (pr - sf_) ** 2).mean() + (dl if dann > 0 and tgt_idx is not None else 0.0)
            else: loss = (nn.functional.huber_loss(pr, y, delta=huber) if huber else nn.functional.mse_loss(pr, y)) + (dl if dann > 0 and tgt_idx is not None else 0.0)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); tl += loss.item(); nb += 1
        pv = predict(va); rv = float(np.sqrt(np.mean((pv - tgt[va]) ** 2))); rt = float(np.sqrt(tl / nb) * 100)
        hist.append((ep, rt, rv))
        if verbose: print(f'  ep {ep} treino {rt:.1f} val {rv:.1f} {time.time() - t0:.0f}s', flush=True)
        if rv < best: best = rv; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return predict(te), hist, net


def prep_params(ref, cols):
    R = ref[cols].to_numpy(np.float64)
    med = np.nanmedian(R, 0); q1, q3 = np.nanpercentile(R, 25, 0), np.nanpercentile(R, 75, 0)
    sc = np.where((q3 - q1) > 1e-6, q3 - q1, np.nanstd(R, 0) + 1e-6); miss = np.isnan(R).mean(0) > 0.005
    return dict(cols=list(cols), med=med, sc=sc, miss=miss)


def prep_apply(d, params):
    A = d[params['cols']].to_numpy(np.float64); med, sc, miss = params['med'], params['sc'], params['miss']
    Z = np.clip((np.where(np.isnan(A), med, A) - med) / sc, -6, 6)
    return np.c_[Z, np.isnan(A[:, miss]).astype(float)].astype(np.float32)
