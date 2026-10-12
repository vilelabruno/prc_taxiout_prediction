from __future__ import annotations

import numpy as np
import pandas as pd

from prc.baseb.models.base import Model, register


@register
class ClockBaseline(Model):

    name = "clock_baseline"

    def fit(self, X, y, meta):
        self.coef: dict[str, tuple[float, float]] = {}
        self.fallback: dict[str, tuple[float, float]] = {}
        apt = meta["ADEP_mvt"].astype(str).to_numpy()
        for a in np.unique(apt):
            m = apt == a
            self.coef[a] = _ols(X["mvt_aobt"].to_numpy()[m], y[m])
            self.fallback[a] = _ols(X["mvt_sched"].to_numpy()[m], y[m])
        self.global_ = _ols(X["mvt_aobt"].to_numpy(), y)
        return self

    def predict(self, X, meta):
        apt = meta["ADEP_mvt"].astype(str).to_numpy()
        x1 = X["mvt_aobt"].to_numpy(dtype=float)
        x2 = X["mvt_sched"].to_numpy(dtype=float)
        pred = np.empty(len(X))
        for i, a in enumerate(apt):
            b, w = self.coef.get(a, self.global_)
            if np.isfinite(x1[i]):
                pred[i] = b + w * x1[i]
            else:
                fb, fw = self.fallback.get(a, self.global_)
                pred[i] = fb + fw * x2[i] if np.isfinite(x2[i]) else b
        return pred


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return (float(np.nanmean(y)) if len(y) else 0.0, 0.0)
    x, y = x[ok], y[ok]
    w = float(np.cov(x, y, bias=True)[0, 1] / (np.var(x) + 1e-9))
    b = float(np.mean(y) - w * np.mean(x))
    return b, w


@register
class LightGBM(Model):
    name = "lightgbm"

    DEFAULTS = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "num_threads": 0,
    }

    def __init__(self, num_boost_round: int = 2000, early_stopping: int = 100, **params):
        super().__init__(num_boost_round=num_boost_round, early_stopping=early_stopping, **params)
        self.lgb_params = {**self.DEFAULTS, **params}
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping

    def fit(self, X, y, meta, valid: tuple[pd.DataFrame, np.ndarray] | None = None):
        import lightgbm as lgb

        self.features = list(X.columns)
        dtrain = lgb.Dataset(X, label=y)
        callbacks = []
        valid_sets = []
        if valid is not None and self.early_stopping > 0:
            valid_sets = [lgb.Dataset(valid[0][self.features], label=valid[1], reference=dtrain)]
            callbacks.append(lgb.early_stopping(self.early_stopping, verbose=False))
        self.booster = lgb.train(
            self.lgb_params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets or None,
            callbacks=callbacks or None,
        )
        self.best_iteration = self.booster.best_iteration or self.num_boost_round
        return self

    def predict(self, X, meta):
        return self.booster.predict(X[self.features], num_iteration=self.best_iteration)

    def importance(self) -> dict[str, float]:
        gain = self.booster.feature_importance(importance_type="gain")
        return dict(sorted(zip(self.features, gain.tolist(), strict=True), key=lambda kv: -kv[1]))


@register
class CatBoost(Model):

    name = "catboost"

    DEFAULTS = {
        "loss_function": "RMSE",
        "depth": 9,
        "learning_rate": 0.05,
        "l2_leaf_reg": 7.0,
        "border_count": 254,
        "random_strength": 1.0,
        "max_ctr_complexity": 2,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "rsm": 0.8,
        "allow_writing_files": False,
        "verbose": False,
    }

    def __init__(
        self,
        num_boost_round: int = 5000,
        early_stopping: int = 100,
        cat_features: list[str] | None = None,
        thread_count: int = 8,
        **params,
    ):
        super().__init__(
            num_boost_round=num_boost_round,
            early_stopping=early_stopping,
            cat_features=cat_features,
            thread_count=thread_count,
            **params,
        )
        self.cb_params = {**self.DEFAULTS, **params, "thread_count": int(thread_count)}
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping
        self.cat_features_cfg = cat_features

    def _pool(self, X: pd.DataFrame, y: np.ndarray | None = None):
        from catboost import Pool

        frame = X[self.features].copy()
        for c in self.cat_features:
            frame[c] = frame[c].astype("int32")
        for c in frame.columns:
            if c in self.cat_features:
                continue
            col = frame[c].to_numpy(dtype=float, copy=True)
            np.nan_to_num(col, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
            frame[c] = col
        return Pool(frame, label=y, cat_features=self.cat_features)

    def fit(self, X, y, meta, valid: tuple[pd.DataFrame, np.ndarray] | None = None):
        from catboost import CatBoostRegressor

        self.features = list(X.columns)
        if self.cat_features_cfg is None:
            self.cat_features = [c for c in self.features if c.startswith("cat_")]
        else:
            self.cat_features = [c for c in self.cat_features_cfg if c in self.features]

        train_pool = self._pool(X, np.asarray(y, dtype=float))
        kwargs: dict = {}
        if valid is not None and self.early_stopping > 0:
            kwargs["eval_set"] = self._pool(valid[0], np.asarray(valid[1], dtype=float))
            kwargs["early_stopping_rounds"] = self.early_stopping
        self.model = CatBoostRegressor(iterations=self.num_boost_round, **self.cb_params)
        self.model.fit(train_pool, **kwargs)
        best = self.model.get_best_iteration()
        self.best_iteration = int(best) + 1 if best is not None else self.num_boost_round
        return self

    def predict(self, X, meta):
        return np.asarray(self.model.predict(self._pool(X)), dtype=float)

    def importance(self) -> dict[str, float]:
        gain = self.model.get_feature_importance()
        return dict(sorted(zip(self.features, [float(g) for g in gain], strict=True),
                           key=lambda kv: -kv[1]))
