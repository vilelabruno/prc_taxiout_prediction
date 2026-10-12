import argparse, json, os, time
import numpy as np, pandas as pd
import torch
from prc.stack.nets import lib as N
from prc.stack.cfg import SEQ_OUT, STACK_MODELS, ensure
parser = argparse.ArgumentParser()
parser.add_argument('cfgs')
parser.add_argument('--device', default='cpu', choices=['cuda', 'mps', 'cpu'])
args = parser.parse_args()
cfgs = json.load(open(args.cfgs)); OUT = str(SEQ_OUT); ensure(OUT, STACK_MODELS / 'nets')
d = N.load_static(N.S4A); d25 = d[d.year == 2025].reset_index(drop=True); d26 = d[d.year == 2026].reset_index(drop=True)
ck = N.OUT / 'seqX_2025_EDDFEDDMEHAMLSZH.npz'
if ck.exists(): z = np.load(ck); Xq25, h25 = z['X'], z['has']
else: Xq25, h25 = N.build_seq(d25, 2025); np.savez(ck, X=Xq25, has=h25)
d25 = d25[h25].reset_index(drop=True); Xq25 = Xq25[h25]
c26 = f'{N.OUT}/seqX_2026.npz'
if os.path.exists(c26): z = np.load(c26); Xq26, h26 = z['X'], z['has']
else: Xq26, h26 = N.build_seq(d26, 2026); np.savez(c26, X=Xq26, has=h26)
d26 = d26[h26].reset_index(drop=True); Xq26 = Xq26[h26]
tgt = np.clip(d25.TAXITIME_SEC_mvt.to_numpy(float), 0, 3600) - d25.pred.to_numpy(float); mes = d25.mes.to_numpy(); blk = d25.bloco.to_numpy(float)
ap25 = pd.Categorical(d25.ADEP_mvt, categories=N.S4A).codes.astype(np.int64); ap26 = pd.Categorical(d26.ADEP_mvt, categories=N.S4A).codes.astype(np.int64)
cols0 = N.static_cols(d25)
print('linhas 2025', len(d25), '2026', len(d26), 'blocos jan', np.unique(blk[(mes == 1) & np.isfinite(blk)]).size, flush=True)
VAL = {1: 4, 7: 10}
for cfg in cfgs:
    tag, kind, seeds, scheme = cfg['tag'], cfg['kind'], cfg.get('seeds', 3), cfg.get('scheme', 'lmo'); kw = dict(cfg.get('kw', {})); folds = cfg.get('folds', [1, 7, 2026])
    nopred = kw.pop('nopred', False); epochs = kw.pop('epochs', 12); threads = kw.pop('threads', 4)
    cols = [c for c in cols0 if not (nopred and 'pred' in c)]; res = []; t0 = time.time(); outp = f'{OUT}/nnh3_{tag}.parquet'
    def run(Xs, Xq, ap, tg, tr, va, te, save=None):
        import gc; ps = []
        for s in range(seeds):
            p_, _, net = N.train_nn(kind, Xs, Xq, ap, tg, tr, va, te, epochs=epochs, seed=s, threads=threads, verbose=False, dev=args.device, **kw)
            ps.append(p_)
            if save is not None: torch.save({'state': {kk: vv.cpu() for kk, vv in net.state_dict().items()}, 'kind': kind, 'kw': kw, 'cols': cols, 'ds': Xs.shape[1], 'apts': N.S4A, **save}, STACK_MODELS / 'nets' / f'{tag}_seed{s}.pt')
            gc.collect(); (torch.mps.empty_cache() if args.device == 'mps' else (torch.cuda.empty_cache() if args.device == 'cuda' else None))
        return np.mean(ps, 0)
    for T in folds:
        if T == 2026:
            va_m = [4, 10]; tr = np.flatnonzero(~np.isin(mes, va_m)); va = np.flatnonzero(np.isin(mes, va_m))
            Xs = N.prep_static(pd.concat([d25, d26], ignore_index=True), cols, d25.iloc[tr]); Xq = np.concatenate([Xq25, Xq26]); ap = np.r_[ap25, ap26]; tg = np.r_[tgt, np.zeros(len(d26))]; te = np.arange(len(d25), len(d25) + len(d26))
            pr = N.prep_params(d25.iloc[tr], cols); p = run(Xs, Xq, ap, tg, tr, va, te, save={'med': pr['med'], 'sc': pr['sc'], 'miss': pr['miss']}); res.append(pd.DataFrame({'MVT_ID_mvt': d26.MVT_ID_mvt.to_numpy(), 'ano': 2026, 'nn_c': p, 'apt': d26.ADEP_mvt.to_numpy()}))
        elif scheme == 'lmo':
            vm = VAL[T]; tr = np.flatnonzero((mes != T) & (mes != vm)); va = np.flatnonzero(mes == vm); te = np.flatnonzero(mes == T)
            Xs = N.prep_static(d25, cols, d25.iloc[tr]); p = run(Xs, Xq25, ap25, tgt, tr, va, te)
            res.append(pd.DataFrame({'MVT_ID_mvt': d25.MVT_ID_mvt.to_numpy()[te], 'ano': 2025, 'nn_c': p, 'apt': d25.ADEP_mvt.to_numpy()[te]})); print(tag, T, 'rmse %.1f -> %.1f' % (np.sqrt(np.mean(tgt[te] ** 2)), np.sqrt(np.mean((p - tgt[te]) ** 2))), f'{time.time()-t0:.0f}s', flush=True)
        else:
            vm = VAL[T]; mT = mes == T; va = np.flatnonzero(mes == vm); Xs = N.prep_static(d25, cols, d25.iloc[np.flatnonzero((mes != T) & (mes != vm))])
            for b in sorted(np.unique(blk[mT & np.isfinite(blk)])):
                te = np.flatnonzero(mT & (blk == b)); tr = np.flatnonzero((mes != vm) & ~((mT) & (blk == b)))
                p = run(Xs, Xq25, ap25, tgt, tr, va, te)
                res.append(pd.DataFrame({'MVT_ID_mvt': d25.MVT_ID_mvt.to_numpy()[te], 'ano': 2025, 'nn_c': p, 'apt': d25.ADEP_mvt.to_numpy()[te]}))
                print(tag, T, 'bloco', int(b), 'rmse %.1f -> %.1f' % (np.sqrt(np.mean(tgt[te] ** 2)), np.sqrt(np.mean((p - tgt[te]) ** 2))), f'{time.time()-t0:.0f}s', flush=True)
        pd.concat(res).to_parquet(outp, index=False)
    print(tag, 'feito', f'{time.time()-t0:.0f}s', flush=True)
