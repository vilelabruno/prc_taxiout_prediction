import json, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from prc.layers import ampliado_f4_corrector as C
from prc.layers.adsb_corrector_champion import P, BASE, ADSB
from prc.layers.adsb_corrector_callsign import obs_mask
from prc.layers.adsb_position_corrector import STAB_ALL, CORE, APPROVED, mask_airport_feats
from prc.adsb.pos_features import POS_FULL
from prc.otimista.inputs import OTI, FINAL_DIR
from prc.otimista.lib import load_holdout, S4, _i64
from prc.otimista.data import nb_path, APTS_OX, legacy_off, OX, OXQ
from prc.otimista.build import PB, PM, POOL, wmap, allm

FIN = S4
POOL_APPLY = ['LEBL', 'EGLL', 'LEMD']
OXALL = OX + OXQ
THR = 0.15
PARAMS_S4 = dict(PB, force_row_wise=True)
PARAMS_POOL = dict(PM, force_col_wise=True)
SEEDS = (3, 11, 29)
GROUPS = {'adsb': ADSB, 'pos': POS_FULL, 'new': C.NEW_FEATS, 'ox': OXALL}


def read_ox(path):
    o = pd.read_parquet(path); o['MVT_ID_mvt'] = o.MVT_ID_mvt.astype('int64'); return o


def merge_ox(d, ox):
    idx = d.index; d = _i64(d).merge(ox, on='MVT_ID_mvt', how='left'); d.index = idx; return d


def auc_dev(a, b, nmin=200, nsamp=15000):
    from sklearn.metrics import roc_auc_score
    a = a.dropna(); b = b.dropna(); na, nb = len(a), len(b)
    if na < nmin or nb < nmin: return np.nan, 0.5, na, nb
    a = a.sample(min(na, nsamp), random_state=0); b = b.sample(min(nb, nsamp), random_state=0)
    auc = float(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a.values, b.values]))
    return auc, abs(auc - 0.5), na, nb


class CtxFinal:

    def __init__(self, T, rk, hold=None, ox_final=FINAL_DIR / 'ox_final.parquet', thr=THR, gate_eddf=120.0, zero_days=('2026-06-11',), log=print):
        t0 = time.time(); self.T = int(T); self.thr = thr; self.log = log; self.gate_eddf = gate_eddf; self.zero_days = tuple(zero_days or ())
        self.h = hold if hold is not None else load_holdout()
        self.h = merge_ox(self.h, pd.concat([read_ox(OTI / f'ox/ox_m{k}.parquet') for k in (1, 7)], ignore_index=True))
        rk = rk[rk.mes == self.T].reset_index(drop=True).copy(); rk = merge_ox(rk, read_ox(ox_final))
        rk['adsb_cov_raw'] = rk.adsb_cov.to_numpy(); rk['gate'] = ''
        if self.zero_days:
            z = rk.dia.isin(self.zero_days).to_numpy() & (rk.adsb_cov == 1).to_numpy(); rk.loc[z, 'adsb_cov'] = 0; rk.loc[z, 'gate'] = 'dia_sem_bruto'
        if gate_eddf is not None:
            g = (rk.ADEP_mvt == 'EDDF').to_numpy() & (rk.adsb_cov == 1).to_numpy() & ~(rk.adsb_first_off >= gate_eddf).to_numpy()
            rk.loc[g, 'adsb_cov'] = 0; rk.loc[g, 'gate'] = 'eddf_first_off'
        self.rk = rk; self.mr = obs_mask(rk); self.covr = (rk.adsb_cov == 1).to_numpy(); self.aptr = rk.ADEP_mvt.to_numpy()
        self.F = C.feature_list(); self.feats = self.F + OXALL
        self.own_raw = merge_ox(_i64(pd.read_parquet(nb_path(self.T))), read_ox(OTI / f'ox/ox_m{self.T}.parquet'))
        self.drift = self._drift_table(); self.keeps = self._keeps()
        self.rkp = self.prep(rk); self._src = {}
        self.log(f'[T={self.T}] ctx: 2026 {len(rk)} linhas, elegiveis (mascara & cobertura) {int((self.mr & self.covr).sum())}, gates {rk.gate.value_counts().to_dict()}, mes proprio 2025 {len(self.own_raw)} linhas, {time.time()-t0:.0f}s')

    def _drift_table(self, nmin=200):
        o = self.own_raw; sel25 = obs_mask(o) & (o.adsb_cov == 1).to_numpy(); sel26 = self.mr & self.covr; rows = []; self.drift_src = {}
        for ap in APTS_OX:
            a25 = o[sel25 & (o.ADEP_mvt == ap).to_numpy()]; a26 = self.rk[sel26 & (self.aptr == ap)]; fonte = self.T
            if len(a25) < nmin:
                for M in sorted(allm(self.T), key=lambda mm: (abs(mm - self.T) if abs(mm - self.T) <= 6 else 12 - abs(mm - self.T), mm)):
                    e = self.h[self.h.mes == M] if M in (1, 7) else merge_ox(_i64(pd.read_parquet(nb_path(M))), read_ox(OTI / f'ox/ox_m{M}.parquet'))
                    e = e[(e.ADEP_mvt == ap).to_numpy()]; e = e[obs_mask(e) & (e.adsb_cov == 1).to_numpy()]
                    if len(e) >= nmin: a25 = e; fonte = M; break
                self.log(f'[T={self.T}] deriva {ap}: sem linhas cobertas em {self.T}/2025 -> fonte substituta {fonte}/2025 ({len(a25)} linhas)')
            self.drift_src[ap] = fonte
            for grp, feats in GROUPS.items():
                for f in feats:
                    auc, dev, na, nb = auc_dev(a25[f], a26[f])
                    rows.append(dict(alvo=self.T, apt=ap, fonte25=fonte, grupo=grp, feat=f, auc=auc, dev=dev, n25=na, n26=nb, stab_all=(f in STAB_ALL.get(ap, [])) if grp == 'adsb' else None, core=(f in CORE) if grp == 'pos' else None))
        d = pd.DataFrame(rows); d['keep'] = False
        return d

    def _keeps(self):
        d = self.drift; dev = {(r.apt, r.feat): r.dev for r in d.itertuples()}
        ok = lambda ap, f: dev.get((ap, f), 0.5) <= self.thr
        stab = {ap: ([f for f in fs if ok(ap, f)] if ap in APTS_OX else list(fs)) for ap, fs in STAB_ALL.items()}
        pos = {ap: list(CORE) + [f for f in POS_FULL if f not in CORE and ok(ap, f)] for ap in APPROVED}
        new = {ap: [f for f in C.NEW_FEATS if ok(ap, f)] for ap in APTS_OX}
        ox = {ap: [f for f in OXALL if ok(ap, f)] for ap in APTS_OX}
        keeps = dict(adsb=stab, pos=pos, new=new, ox=ox)
        k = d.apply(lambda r: r.feat in keeps[r.grupo].get(r.apt, []), axis=1); self.drift['keep'] = k.to_numpy()
        return keeps

    def prep(self, d):
        d = mask_airport_feats(d, ADSB, self.keeps['adsb']); d = mask_airport_feats(d, POS_FULL, self.keeps['pos'])
        d = mask_airport_feats(d, C.NEW_FEATS, self.keeps['new']); return mask_airport_feats(d, OXALL, self.keeps['ox'])

    def save_drift(self, out_dir, tag):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        self.drift.to_csv(out_dir / f'drift_T{self.T}_{tag}.csv', index=False)
        json.dump(dict(T=self.T, thr=self.thr, gate_eddf=self.gate_eddf, zero_days=list(self.zero_days), fonte25=self.drift_src, keeps=self.keeps), open(out_dir / f'keep_T{self.T}_{tag}.json', 'w'), indent=1)

    def src(self, M, apts):
        apts = tuple(apts); k = (int(M), apts)
        if k not in self._src:
            if M in (1, 7): e = self.h[self.h.mes == M].copy()
            elif M == self.T: e = self.own_raw.copy()
            else: e = merge_ox(_i64(pd.read_parquet(nb_path(M))), read_ox(OTI / f'ox/ox_m{M}.parquet'))
            e = self.prep(e); r = self.rkp[self.mr & self.covr]
            e, dropped = C.drift_zero_neighbors(e, r, self.feats, list(apts))
            me = obs_mask(e) & (e.adsb_cov == 1).to_numpy() & e.ADEP_mvt.isin(list(apts)).to_numpy()
            self._src[k] = (e[me].reset_index(drop=True), dropped)
        return self._src[k]

    def clear(self): self._src = {}

    def train_set(self, apts, legacy, include_own=True, feats=None):
        feats = feats or self.feats; months = allm(self.T) + ([self.T] if include_own else []); w = wmap(self.T, months)
        X, y, wv, info = [], [], [], {}
        for M in months:
            e, dropped = self.src(M, apts); e = legacy_off(e) if legacy else e
            X.append(e[feats]); y.append(np.clip(e.TAXITIME_SEC_mvt.astype(float).to_numpy(), 0, 3600) - e.pred.to_numpy()); wv.append(np.full(len(e), w[M]))
            info[M] = dict(n=int(len(e)), w=w[M], zeradas={a: v for a, v in dropped.items() if v})
        return pd.concat(X, ignore_index=True), np.concatenate(y), np.concatenate(wv), info

    def key(self, tag, apts, legacy, params, seeds, include_own, info, feats=None):
        spec = dict(T=self.T, tag=tag, apts=list(apts), legacy=legacy, params={k: params[k] for k in sorted(params)}, base=dict(P), seeds=list(seeds), include_own=include_own, feats=list(feats or self.feats),
                    keeps={g: {ap: v for ap, v in k.items() if ap in apts} for g, k in self.keeps.items()}, thr=self.thr, gate_eddf=self.gate_eddf, zero_days=list(self.zero_days), fontes={str(k): v for k, v in info.items()}, lgb=lgb.__version__)
        return hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:12], spec

    def fit(self, X, y, w, params, seeds, n_jobs):
        bs = []; t0 = time.time()
        for s in seeds:
            bs.append(lgb.LGBMRegressor(**dict(P, **params, n_jobs=n_jobs, random_state=s)).fit(X, y, sample_weight=w).booster_)
            self.log(f'    semente {s} ok {time.time()-t0:.0f}s')
        return bs

    @staticmethod
    def predict(boosters, X):
        return np.mean([b.predict(X.to_numpy(dtype=float)) for b in boosters], axis=0)

    def fit_or_load(self, tag, apts_train, legacy, params, seeds, n_jobs, model_dir=None, use_cache=True, include_own=True):
        t0 = time.time(); X, y, w, info = self.train_set(apts_train, legacy, include_own); key, spec = self.key(tag, apts_train, legacy, params, seeds, include_own, info)
        files = [Path(model_dir) / f'T{self.T}_{tag}_{key}_s{s}.txt' for s in seeds] if model_dir else []
        if use_cache and files and all(f.exists() for f in files):
            bs = [lgb.Booster(model_file=str(f)) for f in files]; self.log(f'[T={self.T}] {tag}: cache {key} ({len(y)} linhas de treino)'); return bs, key, info, len(y), True
        self.log(f'[T={self.T}] {tag}: treino {len(y)} linhas ({ {k: v["n"] for k, v in info.items()} }), {len(self.feats)} features, {len(seeds)} sementes, {n_jobs} threads')
        bs = self.fit(X, y, w, params, seeds, n_jobs)
        if model_dir:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            for b, f in zip(bs, files): b.save_model(str(f))
            json.dump(spec, open(Path(model_dir) / f'T{self.T}_{tag}_{key}.json', 'w'), indent=1, default=str)
        self.log(f'[T={self.T}] {tag}: treinado em {time.time()-t0:.0f}s -> {key}'); return bs, key, info, len(y), False

    def apply26(self, boosters, apts_apply, legacy):
        rk = legacy_off(self.rkp) if legacy else self.rkp; te = self.mr & self.covr & np.isin(self.aptr, list(apts_apply))
        c = np.zeros(len(rk)); c[te] = self.predict(boosters, rk.loc[te, self.feats]); return c, te

    def lmo_eval(self, boosters, apts_eval, apts_src, legacy, doses=(1.0,), B=1000, seed=7):
        e, _ = self.src(self.T, apts_src); e = legacy_off(e) if legacy else e
        y = e.TAXITIME_SEC_mvt.astype(float).to_numpy(); pr = e.pred.to_numpy(float); apt = e.ADEP_mvt.to_numpy(); dia = e.dia.to_numpy()
        c = self.predict(boosters, e[self.feats]); ee = np.clip(y, 0, 3600) - pr; rows = []
        for dose in doses:
            q1 = np.maximum(pr + dose * c, 0.0)
            for ap in ['ALL'] + list(apts_eval):
                s = np.isin(apt, list(apts_eval)) if ap == 'ALL' else (apt == ap)
                n = int(s.sum())
                if n == 0: rows.append(dict(alvo=self.T, apt=ap, dose=dose, n=0)); continue
                g = pd.DataFrame({'d': dia[s], 'e0': ((pr - y) ** 2)[s], 'e1': ((q1 - y) ** 2)[s], 'k': 1}).groupby('d').sum(); rng = np.random.RandomState(seed); bs = np.empty(B)
                for b in range(B):
                    ix = rng.randint(0, len(g), len(g)); nn = g.k.values[ix].sum(); bs[b] = np.sqrt(g.e1.values[ix].sum() / nn) - np.sqrt(g.e0.values[ix].sum() / nn)
                r0 = float(np.sqrt(((pr - y)[s] ** 2).mean())); r1 = float(np.sqrt(((q1 - y)[s] ** 2).mean())); gd = (g.e0 - g.e1)
                rows.append(dict(alvo=self.T, apt=ap, dose=dose, n=n, rmse_base=round(r0, 2), rmse_novo=round(r1, 2), d_rmse=round(r1 - r0, 2), lo=round(float(np.percentile(bs, 5)), 2), hi=round(float(np.percentile(bs, 95)), 2),
                                 s_eff=round(float((c[s] * ee[s]).sum() / max((c[s] ** 2).sum(), 1e-9)), 3), rms_c=round(float(np.sqrt((c[s] ** 2).mean())), 1), media_c=round(float(c[s].mean()), 1), dias_pos=f'{int((gd > 0).sum())}/{len(gd)}'))
        return pd.DataFrame(rows), c, e
