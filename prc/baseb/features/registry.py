from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from prc.baseb.loader import assert_training

FAMILIES: dict[str, type[FeatureFamily]] = {}


class FeatureFamily:

    name: str = ""
    needs_context: bool = False
    columns: list[str]

    def __init__(self, **params):
        self.params = params
        self.columns = []
        self.context: pd.DataFrame | None = None
        self._fitted = False

    def fit(self, train: pd.DataFrame, context: pd.DataFrame | None = None) -> FeatureFamily:
        assert_training(train, what=f"fit feature family {self.name!r}")
        if context is not None:
            assert_training(context, what=f"fit feature family {self.name!r}")
        self.context = context if context is not None else train
        self._fit(train)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"feature family {self.name!r} used before fit()")
        self.context = context if context is not None else df
        out = self._transform(df)
        self.columns = list(out.columns)
        return out

    def _fit(self, train: pd.DataFrame) -> None:
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


def register(cls: type[FeatureFamily]) -> type[FeatureFamily]:
    if not cls.name:
        raise ValueError("feature family needs a name")
    FAMILIES[cls.name] = cls
    return cls


def build_features(
    train: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    families: Iterable[str | dict],
    contexts: dict[str, pd.DataFrame] | None = None,
    train_context: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.DataFrame], list[FeatureFamily]]:
    contexts = contexts or {}
    fitted: list[FeatureFamily] = []
    for spec in families:
        if isinstance(spec, str):
            name, params = spec, {}
        else:
            spec = dict(spec)
            name, params = spec.pop("name"), spec
        if name not in FAMILIES:
            raise KeyError(f"unknown feature family {name!r}; known: {sorted(FAMILIES)}")
        fitted.append(FAMILIES[name](**params).fit(train, context=train_context))

    out: dict[str, pd.DataFrame] = {}
    for key, frame in frames.items():
        parts = [fam.transform(frame, context=contexts.get(key)) for fam in fitted]
        feats = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)
        feats.index = frame.index
        out[key] = feats
    return out, fitted
