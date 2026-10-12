
from __future__ import annotations

import numpy as np
import pandas as pd

from prc.baseb.models.base import Model, build, register, valid_kwargs


def segment_mask(spec: dict, meta: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(meta), dtype=bool)
    airports = spec.get("airports")
    if airports:
        mask &= meta["ADEP_mvt"].astype(str).isin(list(airports)).to_numpy()
    matched = spec.get("matched")
    if matched is not None:
        mask &= meta["AOBT_3_flt"].notna().to_numpy() == bool(matched)
    return mask


def _build(spec: dict) -> Model:
    return build(spec)


@register
class Segmented(Model):

    name = "segmented"

    def __init__(
        self,
        base: dict | None = None,
        segments: list[dict] | None = None,
        drop_segment_rows_from_base: bool = False,
        drop_segment_rows_from_valid: bool | None = None,
        es_exclude: dict | list[dict] | None = None,
        **params,
    ):
        super().__init__(
            base=base,
            segments=segments,
            drop_segment_rows_from_base=drop_segment_rows_from_base,
            drop_segment_rows_from_valid=drop_segment_rows_from_valid,
            es_exclude=es_exclude,
            **params,
        )
        self.base_spec = dict(base or {"name": "lightgbm"})
        self.segment_specs = [dict(s) for s in (segments or [])]
        self.drop = bool(drop_segment_rows_from_base)
        self.drop_valid = (
            self.drop if drop_segment_rows_from_valid is None else bool(drop_segment_rows_from_valid)
        )
        if es_exclude is None:
            self.es_exclude: list[dict] = []
        elif isinstance(es_exclude, dict):
            self.es_exclude = [dict(es_exclude)]
        else:
            self.es_exclude = [dict(s) for s in es_exclude]

    def fit(
        self,
        X,
        y,
        meta,
        valid: tuple[pd.DataFrame, np.ndarray] | None = None,
        valid_meta: pd.DataFrame | None = None,
    ):
        y = np.asarray(y, dtype=float)
        masks = [segment_mask(s, meta) for s in self.segment_specs]
        covered = (
            np.any(masks, axis=0) if masks else np.zeros(len(meta), dtype=bool)
        )
        base_rows = ~covered if self.drop else np.ones(len(meta), dtype=bool)

        self.n_es_rows = None if valid is None else int(len(valid[1]))
        if valid is not None and valid_meta is not None:
            vdrop = np.zeros(len(valid_meta), dtype=bool)
            if self.drop_valid and self.segment_specs:
                vdrop |= np.any([segment_mask(s, valid_meta) for s in self.segment_specs], axis=0)
            for spec in self.es_exclude:
                vdrop |= segment_mask(spec, valid_meta)
            if vdrop.any():
                valid = (valid[0][~vdrop], np.asarray(valid[1], dtype=float)[~vdrop])
                valid_meta = valid_meta[~vdrop]
            self.n_es_rows = int((~vdrop).sum())

        self.base = _build(self.base_spec)
        Xb, yb, mb = X[base_rows], y[base_rows], meta[base_rows]
        base_kwargs = valid_kwargs(self.base.fit, valid, valid_meta)
        self.base.fit(Xb, yb, mb, **base_kwargs)

        self.segments: list[tuple[dict, Model]] = []
        for spec, mask in zip(self.segment_specs, masks, strict=True):
            model = _build(spec["model"])
            if mask.sum() == 0:
                raise ValueError(f"segment {spec} matched no training rows")
            model.fit(X[mask], y[mask], meta[mask])
            self.segments.append((spec, model))
        self.n_segment_rows = [int(m.sum()) for m in masks]
        return self

    @property
    def best_iteration(self) -> int | None:
        return getattr(self.base, "best_iteration", None)

    @property
    def iteration_tree(self) -> list | None:
        return getattr(self.base, "iteration_tree", None)

    def predict(self, X, meta):
        pred = np.asarray(self.base.predict(X, meta), dtype=float)
        for spec, model in self.segments:
            mask = segment_mask(spec, meta)
            if mask.any():
                pred[mask] = np.asarray(model.predict(X[mask], meta[mask]), dtype=float)
        return pred

    def importance(self):
        return self.base.importance() if hasattr(self.base, "importance") else None


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = 1e-6):
    sw = w.sum()
    if sw <= 1e-9:
        return (0.0, 0.0)
    mx = float((w * x).sum() / sw)
    my = float((w * y).sum() / sw)
    vxx = float((w * (x - mx) ** 2).sum() / sw) + ridge
    vxy = float((w * (x - mx) * (y - my)).sum() / sw)
    b = vxy / vxx
    return (my - b * mx, b)


@register
class SchedOLS(Model):

    name = "sched_ols"

    def __init__(self, feature: str = "mvt_sched", per_airport: bool = False, **params):
        super().__init__(feature=feature, per_airport=per_airport, **params)
        self.feature = feature
        self.per_airport = per_airport

    def fit(self, X, y, meta):
        x = X[self.feature].to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        w = ok.astype(float)
        self.global_ = _wls(np.nan_to_num(x), y, w)
        self.fallback_mean = float(y[ok].mean()) if ok.any() else 0.0
        self.by_airport: dict[str, tuple[float, float]] = {}
        if self.per_airport:
            apt = meta["ADEP_mvt"].astype(str).to_numpy()
            for a in np.unique(apt):
                m = (apt == a) & ok
                if m.sum() >= 30:
                    self.by_airport[a] = _wls(x[m], y[m], np.ones(int(m.sum())))
        return self

    def predict(self, X, meta):
        x = X[self.feature].to_numpy(dtype=float)
        apt = meta["ADEP_mvt"].astype(str).to_numpy()
        out = np.empty(len(x), dtype=float)
        for i in range(len(x)):
            a, b = self.by_airport.get(apt[i], self.global_)
            out[i] = a + b * x[i] if np.isfinite(x[i]) else self.fallback_mean
        return out


@register
class SchedMixture(Model):

    name = "sched_mixture"

    def __init__(
        self,
        feature: str = "mvt_sched",
        gate_features: tuple[str, ...] = ("mvt_sched", "hour"),
        max_iter: int = 200,
        tol: float = 1e-6,
        min_sigma: float = 60.0,
        seed: int = 0,
        **params,
    ):
        super().__init__(
            feature=feature,
            gate_features=list(gate_features),
            max_iter=max_iter,
            tol=tol,
            min_sigma=min_sigma,
            seed=seed,
            **params,
        )
        self.feature = feature
        self.gate_features = list(gate_features)
        self.max_iter = max_iter
        self.tol = tol
        self.min_sigma = min_sigma
        self.seed = seed

    def _gate_matrix(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for name in self.gate_features:
            v = X[name].to_numpy(dtype=float)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            if name == "hour":
                cols.append(np.sin(2 * np.pi * v / 24))
                cols.append(np.cos(2 * np.pi * v / 24))
            elif name == self.feature:
                cols.append(np.sign(v) * np.log1p(np.abs(v)))
            else:
                cols.append(v)
        G = np.column_stack(cols) if cols else np.zeros((len(X), 0))
        return G

    def _standardise(self, G: np.ndarray) -> np.ndarray:
        return (G - self.g_mean) / self.g_scale

    def fit(self, X, y, meta):
        rng = np.random.default_rng(self.seed)
        x = np.nan_to_num(X[self.feature].to_numpy(dtype=float), nan=0.0)
        y = np.asarray(y, dtype=float)
        G = self._gate_matrix(X)
        self.g_mean = G.mean(axis=0) if G.size else np.zeros(0)
        self.g_scale = np.where(G.std(axis=0) > 1e-9, G.std(axis=0), 1.0) if G.size else np.ones(0)
        Gs = self._standardise(G) if G.size else G

        r = (y > 0.5 * np.maximum(x, 0.0)).astype(float)
        r = np.clip(r + rng.normal(0, 1e-3, len(r)), 1e-3, 1 - 1e-3)
        if r.mean() < 0.02 or r.mean() > 0.98:
            r = np.clip(np.full(len(y), 0.5) + rng.normal(0, 1e-2, len(y)), 1e-3, 1 - 1e-3)

        self.gate = None
        prev = -np.inf
        for _ in range(self.max_iter):
            self.coef1 = _wls(x, y, r)
            self.coef0 = _wls(x, y, 1.0 - r)
            self.s1 = max(self.min_sigma, _wstd(y - (self.coef1[0] + self.coef1[1] * x), r))
            self.s0 = max(self.min_sigma, _wstd(y - (self.coef0[0] + self.coef0[1] * x), 1.0 - r))
            p = self._fit_gate(Gs, r)
            d1 = p * _npdf(y - (self.coef1[0] + self.coef1[1] * x), self.s1)
            d0 = (1 - p) * _npdf(y - (self.coef0[0] + self.coef0[1] * x), self.s0)
            tot = d1 + d0 + 1e-300
            r = np.clip(d1 / tot, 1e-6, 1 - 1e-6)
            ll = float(np.log(tot).mean())
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        self.loglik = prev
        self.weight_gate_held = float(r.mean())
        return self

    def _fit_gate(self, Gs: np.ndarray, r: np.ndarray) -> np.ndarray:
        share = float(np.clip(r.mean(), 1e-4, 1 - 1e-4))
        if Gs.shape[1] == 0 or len(r) < 40:
            self.gate = None
            self.const_p = share
            return np.full(len(r), share)
        try:
            from sklearn.linear_model import LogisticRegression

            Xd = np.vstack([Gs, Gs])
            yd = np.concatenate([np.ones(len(r)), np.zeros(len(r))])
            wd = np.concatenate([r, 1.0 - r])
            gate = LogisticRegression(C=1.0, max_iter=1000)
            gate.fit(Xd, yd, sample_weight=wd)
            self.gate = gate
            self.const_p = share
            return np.clip(gate.predict_proba(Gs)[:, 1], 1e-6, 1 - 1e-6)
        except Exception:
            self.gate = None
            self.const_p = share
            return np.full(len(r), share)

    def mixing(self, X) -> np.ndarray:
        if self.gate is None:
            return np.full(len(X), self.const_p)
        Gs = self._standardise(self._gate_matrix(X))
        return np.clip(self.gate.predict_proba(Gs)[:, 1], 1e-6, 1 - 1e-6)

    def predict(self, X, meta=None):
        x = np.nan_to_num(X[self.feature].to_numpy(dtype=float), nan=0.0)
        p = self.mixing(X)
        m1 = self.coef1[0] + self.coef1[1] * x
        m0 = self.coef0[0] + self.coef0[1] * x
        return p * m1 + (1 - p) * m0


@register
class SnapGate(Model):

    name = "snap_gate"

    def __init__(
        self,
        feature: str = "mvt_sched",
        tol: float = 300.0,
        hour_feature: str = "hour",
        te_keys: tuple[str, ...] = ("STAND_mvt:1", "FLIGHT_mvt:3"),
        prior_n: float = 20.0,
        C: float = 1.0,
        floor_quantile: float | None = None,
        day_count: bool = False,
        **params,
    ):
        super().__init__(
            feature=feature,
            tol=tol,
            hour_feature=hour_feature,
            te_keys=list(te_keys),
            prior_n=prior_n,
            C=C,
            floor_quantile=floor_quantile,
            day_count=day_count,
            **params,
        )
        self.day_count = bool(day_count)
        if floor_quantile is not None and not 0.0 <= floor_quantile < 1.0:
            raise ValueError(f"floor_quantile must be in [0, 1), got {floor_quantile!r}")
        self.floor_quantile = floor_quantile
        self.feature = feature
        self.tol = float(tol)
        self.hour_feature = hour_feature
        self.te_keys = list(te_keys)
        self.prior_n = float(prior_n)
        self.C = float(C)

    @staticmethod
    def _key_values(meta: pd.DataFrame, key: str) -> np.ndarray:
        col, _, k = key.partition(":")
        if meta is None or col not in meta.columns:
            return np.full(len(meta) if meta is not None else 0, "?", dtype=object)
        s = meta[col].astype("string").fillna("?")
        if k:
            s = s.str.slice(0, int(k))
        return s.to_numpy(dtype=object)

    def _design(self, X: pd.DataFrame, meta: pd.DataFrame | None) -> np.ndarray:
        x = np.nan_to_num(X[self.feature].to_numpy(dtype=float), nan=0.0)
        cols = [np.sign(x) * np.log1p(np.abs(x))]
        if self.hour_feature in X.columns:
            h = np.nan_to_num(X[self.hour_feature].to_numpy(dtype=float), nan=0.0)
            cols += [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)]
        for key in self.te_keys:
            enc, prior = self.encoders.get(key, ({}, self.prior))
            vals = self._key_values(meta, key) if meta is not None else []
            cols.append(np.array([enc.get(v, prior) for v in vals], dtype=float)
                        if len(vals) else np.full(len(x), prior))
        if self.day_count:
            if meta is None or "MVT_TIME_UTC_mvt" not in meta.columns:
                raise ValueError("snap_gate day_count needs MVT_TIME_UTC_mvt in meta")
            day = pd.Series(meta["MVT_TIME_UTC_mvt"].to_numpy()).dt.floor("D")
            cols.append(np.log1p(day.map(day.value_counts()).to_numpy(dtype=float)))
        return np.column_stack(cols)

    def fit(self, X, y, meta):
        y = np.asarray(y, dtype=float)
        x = np.nan_to_num(X[self.feature].to_numpy(dtype=float), nan=0.0)
        snap = (np.abs(y - x) < self.tol).astype(float)
        self.snap_rate = float(snap.mean())
        self.prior = self.snap_rate
        self.floor = (float(np.quantile(y, self.floor_quantile))
                      if self.floor_quantile is not None and len(y) else None)

        self.encoders: dict[str, tuple[dict, float]] = {}
        for key in self.te_keys:
            vals = self._key_values(meta, key)
            if len(vals) == 0:
                continue
            g = pd.DataFrame({"k": vals, "v": snap}).groupby("k")["v"].agg(["sum", "size"])
            enc = ((g["sum"] + self.prior * self.prior_n) / (g["size"] + self.prior_n)).to_dict()
            self.encoders[key] = (enc, self.prior)

        self.coef1 = _wls(x, y, snap)
        self.coef0 = _wls(x, y, 1.0 - snap)
        if snap.sum() < 5:
            self.coef1 = (0.0, 1.0)
        if (1 - snap).sum() < 5:
            self.coef0 = (float(y.mean()) if len(y) else 0.0, 0.0)

        G = self._design(X, meta)
        self.g_mean = G.mean(axis=0)
        self.g_scale = np.where(G.std(axis=0) > 1e-9, G.std(axis=0), 1.0)
        self.gate = None
        if 0.01 < self.snap_rate < 0.99 and len(y) >= 40:
            try:
                from sklearn.linear_model import LogisticRegression

                self.gate = LogisticRegression(C=self.C, max_iter=1000).fit(
                    (G - self.g_mean) / self.g_scale, snap.astype(int)
                )
            except Exception:
                self.gate = None
        return self

    def mixing(self, X, meta=None) -> np.ndarray:
        if self.gate is None:
            return np.full(len(X), self.snap_rate)
        G = (self._design(X, meta) - self.g_mean) / self.g_scale
        return np.clip(self.gate.predict_proba(G)[:, 1], 1e-6, 1 - 1e-6)

    def predict(self, X, meta=None):
        x = np.nan_to_num(X[self.feature].to_numpy(dtype=float), nan=0.0)
        p = self.mixing(X, meta)
        out = p * (self.coef1[0] + self.coef1[1] * x) + (1 - p) * (
            self.coef0[0] + self.coef0[1] * x
        )
        return out if self.floor is None else np.maximum(out, self.floor)


def _wstd(res: np.ndarray, w: np.ndarray) -> float:
    sw = w.sum()
    if sw <= 1e-9:
        return 0.0
    return float(np.sqrt((w * res**2).sum() / sw))


def _npdf(res: np.ndarray, sigma: float) -> np.ndarray:
    z = res / sigma
    return np.exp(-0.5 * np.clip(z * z, 0, 700)) / (sigma * np.sqrt(2 * np.pi))


@register
class PerAirport(Model):

    name = "per_airport"

    def __init__(
        self,
        base: dict | None = None,
        min_rows: int = 2000,
        min_valid_rows: int = 200,
        fallback: bool = True,
        key: str | list[str] = "ADEP_mvt",
        rounds: dict | None = None,
        **params,
    ):
        super().__init__(
            base=base,
            min_rows=min_rows,
            min_valid_rows=min_valid_rows,
            fallback=fallback,
            key=key,
            rounds=rounds,
            **params,
        )
        self.base_spec = dict(base or {"name": "lightgbm"})
        self.min_rows = int(min_rows)
        self.min_valid_rows = int(min_valid_rows)
        self.use_fallback = bool(fallback)
        self.key = [str(k) for k in key] if isinstance(key, (list, tuple)) else str(key)
        self.rounds = {str(k): int(v) for k, v in (rounds or {}).items()}

    def _key_of(self, meta: pd.DataFrame) -> np.ndarray:
        cols = [self.key] if isinstance(self.key, str) else self.key
        parts = []
        for c in cols:
            col = meta[c]
            parts.append(col.where(col.notna(), "__na__").astype(object).astype(str))
        out = parts[0]
        for part in parts[1:]:
            out = out + "/" + part
        return out.to_numpy(dtype=object)

    def _spec_for(self, level: str) -> dict:
        n = self.rounds.get(level)
        if n is None:
            return self.base_spec
        return {**self.base_spec, "num_boost_round": int(n), "early_stopping": 0}

    def _fit_one(self, model, X, y, meta, valid, valid_meta=None):
        model.fit(X, y, meta, **valid_kwargs(model.fit, valid, valid_meta))
        return model

    def fit(
        self,
        X,
        y,
        meta,
        valid: tuple[pd.DataFrame, np.ndarray] | None = None,
        valid_meta: pd.DataFrame | None = None,
    ):
        y = np.asarray(y, dtype=float)
        key = self._key_of(meta)
        vkey = None if valid_meta is None else self._key_of(valid_meta)

        levels, counts = np.unique(key, return_counts=True)
        self.levels = [str(a) for a, n in zip(levels, counts, strict=True) if n >= self.min_rows]
        self.n_rows = {str(a): int(n) for a, n in zip(levels, counts, strict=True)}

        self.models: dict[str, Model] = {}
        self.specialist_iterations: dict[str, int | None] = {}
        for a in self.levels:
            m = key == a
            sub = valid
            if valid is not None and vkey is not None:
                vm = vkey == a
                if vm.sum() >= self.min_valid_rows:
                    sub = (valid[0][vm], np.asarray(valid[1], dtype=float)[vm])
            model = self._fit_one(_build(self._spec_for(a)), X[m], y[m], meta[m], sub)
            self.models[a] = model
            self.specialist_iterations[a] = getattr(model, "best_iteration", None)

        self.fallback_ = None
        self.fallback_mean_ = float(np.mean(y)) if len(y) else 0.0
        uncovered = ~np.isin(key, self.levels)
        if self.use_fallback and (uncovered.any() or len(self.levels) == 0):
            self.fallback_ = self._fit_one(
                _build(self.base_spec), X, y, meta, valid, valid_meta
            )
        return self

    @property
    def best_iteration(self) -> int | None:
        its = [i for i in getattr(self, "specialist_iterations", {}).values() if i]
        return max(its) if its else None

    @property
    def iteration_tree(self) -> dict | None:
        tree = getattr(self, "specialist_iterations", None)
        return dict(tree) if tree else None

    def predict(self, X, meta):
        key = self._key_of(meta)
        pred = np.full(len(X), np.nan, dtype=float)
        for a, model in self.models.items():
            m = key == a
            if m.any():
                pred[m] = np.asarray(model.predict(X[m], meta[m]), dtype=float)
        rest = ~np.isfinite(pred)
        if rest.any():
            if self.fallback_ is not None:
                pred[rest] = np.asarray(
                    self.fallback_.predict(X[rest], meta[rest]), dtype=float
                )
            else:
                pred[rest] = self.fallback_mean_
        return pred

    def importance(self) -> dict[str, float] | None:
        total: dict[str, float] = {}
        seen = 0.0
        for a, model in self.models.items():
            imp = model.importance() if hasattr(model, "importance") else None
            if not imp:
                continue
            w = float(self.n_rows.get(a, 1))
            seen += w
            for k, v in imp.items():
                total[k] = total.get(k, 0.0) + w * float(v)
        if not total:
            return None
        total = {k: v / seen for k, v in total.items()}
        return dict(sorted(total.items(), key=lambda kv: -kv[1]))
