import argparse, json, time
import numpy as np, pandas as pd, torch
from prc.stack.nets import lib as N
from prc.stack.cfg import SEQ_OUT, STACK_MODELS, ensure

parser = argparse.ArgumentParser()
parser.add_argument('tag')
parser.add_argument('kind')
parser.add_argument('seeds', type=int)
parser.add_argument('kw', nargs='?', default='')
parser.add_argument('--device', default='cpu', choices=['cuda', 'mps', 'cpu'])
args = parser.parse_args()
tag, kind, seeds = args.tag, args.kind, args.seeds
kw = {k: float(v) if '.' in v else int(v) for k, v in (x.split('=') for x in args.kw.split(','))} if args.kw else {}
epochs = kw.pop('epochs', 12); threads = kw.pop('threads', 4)
ES = [4, 10]
d = N.load_static(N.S4A)
d25 = d[d.year == 2025].reset_index(drop=True); d26 = d[d.year == 2026].reset_index(drop=True)
z = np.load(N.OUT / 'seqX_2025_EDDFEDDMEHAMLSZH.npz'); Xq25, h25 = z['X'], z['has']; d25 = d25[h25].reset_index(drop=True); Xq25 = Xq25[h25]
z = np.load(N.OUT / 'seqX_2026.npz'); Xq26, h26 = z['X'], z['has']; d26 = d26[h26].reset_index(drop=True); Xq26 = Xq26[h26]
tgt = np.clip(d25.TAXITIME_SEC_mvt.to_numpy(float), 0, 3600) - d25.pred.to_numpy(float); mes = d25.mes.to_numpy()
cols = N.static_cols(d25)
tr = np.flatnonzero(~np.isin(mes, ES)); va = np.flatnonzero(np.isin(mes, ES))
params = N.prep_params(d25.iloc[tr], cols)
Xs = N.prep_apply(pd.concat([d25, d26], ignore_index=True), params)
import gc
dev = args.device
Tq = torch.empty((len(Xq25) + len(Xq26),) + Xq25.shape[1:], dtype=torch.float16, device=dev); Tq[:len(Xq25)] = torch.from_numpy(Xq25).to(dev); Tq[len(Xq25):] = torch.from_numpy(Xq26).to(dev); del Xq25, Xq26; gc.collect()
ap = np.r_[pd.Categorical(d25.ADEP_mvt, categories=N.S4A).codes, pd.Categorical(d26.ADEP_mvt, categories=N.S4A).codes].astype(np.int64)
tg = np.r_[tgt, np.zeros(len(d26))]; te = np.arange(len(d25), len(d25) + len(d26))
print('treino', len(tr), 'val', len(va), 'predicao 2026', len(te), 'colunas estaticas', Xs.shape[1], flush=True)
MD = STACK_MODELS / 'nets'; ensure(MD, SEQ_OUT)
np.savez(MD / f'prep_{tag}.npz', med=params['med'], sc=params['sc'], miss=params['miss']); json.dump({'cols': cols, 'kind': kind, 'kw': kw, 'epochs': epochs, 'apts': N.S4A, 'ds': int(Xs.shape[1]), 'es_months': ES, 'train_rows': int(len(tr))}, open(MD / f'meta_{tag}.json', 'w'))
out = pd.DataFrame({'MVT_ID_mvt': d26.MVT_ID_mvt.to_numpy(), 'ano': 2026, 'apt': d26.ADEP_mvt.to_numpy()}); ps = []; t0 = time.time()
for s in range(seeds):
    p, hist, net = N.train_nn(kind, Xs, None, ap, tg, tr, va, te, tq=Tq, epochs=epochs, seed=s, threads=threads, verbose=False, dev=dev, **kw)
    torch.save(net.state_dict(), MD / f'{tag}_seed{s}.pt'); ps.append(p); out[f'nn_c_s{s}'] = p; out['nn_c'] = np.mean(ps, 0)
    out.to_parquet(SEQ_OUT / f'nnfull_{tag}.parquet', index=False)
    print('semente', s, 'melhor val %.2f' % min(h[2] for h in hist), 'rms da correcao 2026 %.1f' % np.sqrt(np.mean(np.mean(ps, 0) ** 2)), f'{time.time()-t0:.0f}s', flush=True)
print('feito')
