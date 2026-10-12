from collections.abc import Iterable

import pandas as pd

from prc.baseb.config import PHASE_COL, TARGET, TIMESTAMP_COLUMNS, TRAINING_MONTHS, training_filename
from prc.paths import RANKING, SUBMITTING, TRAINING_DIR

PROVENANCE_ATTR = "prc26_provenance"


class DataPolicyError(RuntimeError):
    pass


def _coerce_timestamps(df):
    for col in TIMESTAMP_COLUMNS:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        elif col in df.columns and df[col].dt.tz is None:
            df[col] = df[col].dt.tz_localize("UTC")
    return df


def _read(path, columns=None):
    df = pd.read_parquet(path, columns=list(columns) if columns else None)
    return _coerce_timestamps(df)


def load_training(months: Iterable[str] | None = None, columns: Iterable[str] | None = None):
    months = tuple(months) if months else TRAINING_MONTHS
    frames = []
    for month in months:
        frame = _read(TRAINING_DIR / training_filename(month), columns)
        frame["month"] = month
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df.attrs[PROVENANCE_ATTR] = "training"
    return df


def load_ranking():
    df = _read(RANKING)
    df.attrs[PROVENANCE_ATTR] = "ranking"
    return df


def load_submission_template():
    df = pd.read_parquet(SUBMITTING)
    df.attrs[PROVENANCE_ATTR] = "submitting"
    return df


def assert_training(df, what="fit"):
    if df.attrs.get(PROVENANCE_ATTR) != "training":
        raise DataPolicyError(
            f"attempted to {what} on data with provenance "
            f"{df.attrs.get(PROVENANCE_ATTR)!r}; only training data may be used"
        )


def departures(df):
    out = df[df[PHASE_COL] == "DEP"].copy()
    out.attrs.update(df.attrs)
    return out


def with_target(df):
    out = departures(df)
    out = out[out[TARGET].notna()].copy()
    out.attrs.update(df.attrs)
    return out
