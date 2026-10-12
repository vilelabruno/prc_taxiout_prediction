
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from prc.baseb.config import ID_COL, TARGET
from prc.baseb.loader import departures, load_ranking, load_submission_template, load_training, with_target
from prc.baseb.models.segment import SnapGate
from prc.baseb.submit import build_submission, write_submission
from prc.paths import ARTIFACTS

OUT = ARTIFACTS / "baseb"
V34 = OUT / "base_b_v34.parquet"
V35 = OUT / "base_b_v35.parquet"
GLOBAL_FLOOR_Q = 0.001
SEG_FLOOR_Q = 0.05
PRED = "TAXITIME_SEC_mvt"


def seg(df: pd.DataFrame) -> np.ndarray:
    return ((df["ADEP_mvt"].astype(str) == "LIRF") & df["AOBT_3_flt"].isna()).to_numpy()


def design(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "mvt_sched": (df["MVT_TIME_UTC_mvt"] - df["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy(),
        "hour": df["MVT_TIME_UTC_mvt"].dt.hour.to_numpy(),
    }, index=df.index)


def main(write: bool) -> None:
    train = with_target(load_training(None))
    rank = departures(load_ranking())
    tr, rk = train[seg(train)], rank[seg(rank)]
    ytr = tr[TARGET].to_numpy(float)
    print(f"LIRF-unmatched: {len(tr)} training rows (12 months), {len(rk)} ranking rows")

    v34 = pd.read_parquet(V34)
    pred = pd.Series(v34[PRED].to_numpy(float), index=v34[ID_COL].to_numpy())
    rk_ids = rk[ID_COL].to_numpy()

    shipped = SnapGate(tol=300).fit(design(tr), ytr, tr)
    same = np.clip(shipped.predict(design(rk), rk), 0, 172800)
    diff = np.abs(same - pred.loc[rk_ids].to_numpy())
    print(f"identity: shipped gate refit vs v34 on segment rows, max |diff| {diff.max():.4f} s")
    if diff.max() > 0.01:
        raise SystemExit("the segment refit does not reproduce v34 - substitution would not isolate H-047")

    new = SnapGate(tol=300, floor_quantile=SEG_FLOOR_Q, day_count=True).fit(design(tr), ytr, tr)
    seg_pred = np.clip(new.predict(design(rk), rk), 0, 172800)
    out = pred.copy()
    out.loc[rk_ids] = seg_pred
    lo = float(np.quantile(train[TARGET].to_numpy(float), GLOBAL_FLOOR_Q))
    lifted = int((out < lo).sum())
    out = out.clip(lower=lo)
    zeros34 = int((pred == 0).sum())
    print(f"segment floor {new.floor:.0f} s; global floor {lo:.0f} s lifts {lifted} rows; "
          f"zero predictions {zeros34} -> {int((out == 0).sum())}")
    print(f"rms(v35 - v34) {float(np.sqrt(np.mean((out - pred) ** 2))):.2f} s over all rows; "
          f"on segment rows {float(np.sqrt(np.mean((seg_pred - pred.loc[rk_ids].to_numpy()) ** 2))):.1f} s")

    if write:
        sub = build_submission(out, load_submission_template())
        path = write_submission(sub, V35)
        print(f"wrote {path}")


if __name__ == "__main__":
    main("--write" in sys.argv)
