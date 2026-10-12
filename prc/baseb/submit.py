import numpy as np
import pandas as pd

from prc.baseb.config import ID_COL, TARGET


class SubmissionError(ValueError):
    pass


def validate_submission(sub, template):
    if list(sub.columns) != [ID_COL, TARGET]:
        raise SubmissionError(f"columns must be [{ID_COL}, {TARGET}], got {list(sub.columns)}")
    if len(sub) != len(template):
        raise SubmissionError(f"row count {len(sub)} != template {len(template)}")
    if sub[ID_COL].duplicated().any():
        raise SubmissionError("duplicate MVT_ID_mvt")
    if set(sub[ID_COL]) != set(template[ID_COL]):
        raise SubmissionError("MVT_ID_mvt set differs from template")
    y = sub[TARGET].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise SubmissionError("non-finite predictions")
    if (y < 0).any():
        raise SubmissionError("negative taxi times")


def build_submission(pred, template):
    out = template[[ID_COL]].copy()
    out[TARGET] = out[ID_COL].map(pred).astype("float64")
    if out[TARGET].isna().any():
        missing = int(out[TARGET].isna().sum())
        raise SubmissionError(f"{missing} template ids have no prediction")
    validate_submission(out, template)
    return out


def write_submission(sub, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(path, index=False)
    return path
