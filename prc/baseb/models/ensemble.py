
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from prc.baseb.models.base import Model, build, register, valid_kwargs

SEED_KEYS = ("seed", "bagging_seed", "feature_fraction_seed", "data_random_seed")


@register
class Ensemble(Model):

    name = "ensemble"

    def __init__(
        self,
        members: list[dict] | None = None,
        base: dict | None = None,
        seeds: list[int | None] | None = None,
        weights: list[float] | None = None,
        **params,
    ):
        super().__init__(members=members, base=base, seeds=seeds, weights=weights, **params)
        if members:
            specs = [dict(m) for m in members]
        elif base is not None and seeds:
            specs = [_with_seed(base, s) for s in seeds]
        else:
            raise ValueError("ensemble needs either `members` or both `base` and `seeds`")
        if not specs:
            raise ValueError("ensemble needs at least one member")
        self.member_specs = specs
        if weights is None:
            w = np.ones(len(specs), dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if len(w) != len(specs):
                raise ValueError(f"{len(w)} weights for {len(specs)} members")
            if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
                raise ValueError("weights must be finite, non-negative and not all zero")
        self.weights = w / w.sum()

    def fit(
        self,
        X,
        y,
        meta,
        valid: tuple[pd.DataFrame, np.ndarray] | None = None,
        valid_meta: pd.DataFrame | None = None,
    ):
        self.models: list[Model] = []
        for spec in self.member_specs:
            model = build(spec)
            model.fit(X, y, meta, **valid_kwargs(model.fit, valid, valid_meta))
            self.models.append(model)
        self.best_iterations = [getattr(m, "best_iteration", None) for m in self.models]
        return self

    @property
    def best_iteration(self) -> int | None:
        its = [i for i in getattr(self, "best_iterations", []) if i]
        return max(its) if its else None

    @property
    def iteration_tree(self) -> list:
        out = []
        for m in getattr(self, "models", []):
            sub = getattr(m, "iteration_tree", None)
            out.append(sub if sub is not None else getattr(m, "best_iteration", None))
        return out

    def predict(self, X, meta):
        preds = [np.asarray(m.predict(X, meta), dtype=float) for m in self.models]
        self.member_predictions_ = preds
        return np.average(np.vstack(preds), axis=0, weights=self.weights)

    def importance(self) -> dict[str, float] | None:
        total: dict[str, float] = {}
        seen = 0.0
        for w, model in zip(self.weights, self.models, strict=True):
            imp = model.importance() if hasattr(model, "importance") else None
            if not imp:
                continue
            seen += w
            for k, v in imp.items():
                total[k] = total.get(k, 0.0) + w * float(v)
        if not total:
            return None
        total = {k: v / seen for k, v in total.items()}
        return dict(sorted(total.items(), key=lambda kv: -kv[1]))


def _with_seed(spec: dict, seed: Any) -> dict:
    out = dict(spec)
    if seed is None:
        return out
    for key in SEED_KEYS:
        out[key] = int(seed)
    return out
