from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

MODELS: dict[str, type[Model]] = {}


class Model:
    name: str = ""

    def __init__(self, **params):
        self.params = params

    def fit(self, X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame) -> Model:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


def register(cls: type[Model]) -> type[Model]:
    MODELS[cls.name] = cls
    return cls


def build(spec: dict) -> Model:
    spec = dict(spec)
    name = spec.pop("name")
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")
    return MODELS[name](**spec)


def valid_kwargs(fit, valid, valid_meta=None) -> dict:
    if valid is None:
        return {}
    params = inspect.signature(fit).parameters
    if "valid" not in params:
        return {}
    out: dict = {"valid": valid}
    if valid_meta is not None and "valid_meta" in params:
        out["valid_meta"] = valid_meta
    return out
