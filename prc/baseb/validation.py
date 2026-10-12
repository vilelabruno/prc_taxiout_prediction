
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from prc.baseb.config import DEFAULT_HOLDOUT_MONTHS, TARGET


def split_by_month(
    df: pd.DataFrame, holdout_months: Iterable[int] = DEFAULT_HOLDOUT_MONTHS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = df["MVT_TIME_UTC_mvt"].dt.month
    hold = months.isin(list(holdout_months))
    train, valid = df[~hold].copy(), df[hold].copy()
    train.attrs.update(df.attrs)
    valid.attrs.update(df.attrs)
    return train, valid


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


WITHIN_MINUTES = (1, 3, 5)


def within_minutes(y: np.ndarray, p: np.ndarray, minutes: Iterable[int] = WITHIN_MINUTES) -> dict:
    err = np.abs(np.asarray(y, dtype=float) - np.asarray(p, dtype=float))
    return {str(m): float((err <= m * 60).mean() * 100) for m in minutes}


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(p, dtype=float))))


def evaluate(valid: pd.DataFrame, pred: np.ndarray) -> dict:
    y = valid[TARGET].to_numpy(dtype=float)
    pred = np.asarray(pred, dtype=float)
    matched = valid["AOBT_3_flt"].notna().to_numpy()
    out = {
        "rmse": rmse(y, pred),
        "n": int(len(y)),
        "rmse_matched": rmse(y[matched], pred[matched]) if matched.any() else None,
        "rmse_unmatched": rmse(y[~matched], pred[~matched]) if (~matched).any() else None,
        "share_unmatched": float((~matched).mean()),
        "sse_share_unmatched": float(((y - pred)[~matched] ** 2).sum() / ((y - pred) ** 2).sum())
        if (~matched).any()
        else 0.0,
        "mae": mae(y, pred),
        "within_min": within_minutes(y, pred),
        "within_min_matched": within_minutes(y[matched], pred[matched]) if matched.any() else None,
        "per_airport": {},
    }
    for apt, idx in valid.groupby("ADEP_mvt").indices.items():
        out["per_airport"][str(apt)] = {"rmse": rmse(y[idx], pred[idx]), "n": int(len(idx))}
    sse = float(((y - pred) ** 2).sum()) or 1.0
    out["per_airport_unmatched"] = {}
    for apt, idx in valid[~matched].groupby("ADEP_mvt").indices.items():
        pos = np.flatnonzero(~matched)[idx]
        out["per_airport_unmatched"][str(apt)] = {
            "rmse": rmse(y[pos], pred[pos]),
            "n": int(len(pos)),
            "sse_share": float(((y - pred)[pos] ** 2).sum() / sse),
        }
    return out
