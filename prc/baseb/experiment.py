
from __future__ import annotations

import inspect
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from prc.baseb.config import TARGET
from prc.baseb.features import build_features
from prc.baseb.loader import load_training, with_target
from prc.baseb.models import MODELS
from prc.baseb.validation import evaluate, split_by_month
from prc.paths import WORK

TARGET_MODES = ("direct", "residual_over_aobt")


def transform_target(mode: str, y: np.ndarray, feats: pd.DataFrame) -> np.ndarray:
    if mode == "direct":
        return y
    if mode == "residual_over_aobt":
        base = feats["mvt_aobt"].to_numpy(dtype=float)
        return np.where(np.isfinite(base), y - base, y)
    raise ValueError(f"unknown target mode {mode}")


def inverse_target(mode: str, z: np.ndarray, feats: pd.DataFrame) -> np.ndarray:
    if mode == "direct":
        return z
    base = feats["mvt_aobt"].to_numpy(dtype=float)
    return np.where(np.isfinite(base), z + base, z)


CLIP_GROUP_KEYS = {
    "airport": lambda meta: meta["ADEP_mvt"].astype(str).to_numpy(),
    "matched": lambda meta: meta["AOBT_3_flt"].notna().to_numpy(),
}


def clip_train_target(
    z: np.ndarray, meta: pd.DataFrame, spec: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    q = float(spec.get("quantile", 0.999))
    if not 0.0 < q < 1.0:
        raise ValueError(f"target_clip.quantile must be in (0, 1), got {q}")
    side = spec.get("side", "upper")
    if side not in ("upper", "both"):
        raise ValueError(f"target_clip.side must be 'upper' or 'both', got {side!r}")
    by = list(spec.get("by", ["airport", "matched"]))
    unknown = [k for k in by if k not in CLIP_GROUP_KEYS]
    if unknown:
        raise ValueError(f"unknown target_clip.by keys {unknown}; have {sorted(CLIP_GROUP_KEYS)}")

    z = np.asarray(z, dtype=float)
    frame = pd.DataFrame({k: CLIP_GROUP_KEYS[k](meta) for k in by})
    frame["z"] = z
    grouped = frame.groupby(by, sort=False)["z"] if by else frame["z"]
    hi = grouped.transform("quantile", q).to_numpy(float) if by else np.quantile(z, q)
    out = np.minimum(z, hi)
    info: dict[str, Any] = {
        "quantile": q,
        "by": by,
        "side": side,
        "n_clipped_upper": int((z > hi).sum()),
    }
    if side == "both":
        lo = grouped.transform("quantile", 1.0 - q).to_numpy(float) if by else np.quantile(z, 1 - q)
        info["n_clipped_lower"] = int((z < lo).sum())
        out = np.maximum(out, lo)
    info["n_clipped"] = int((out != z).sum())
    info["sum_abs_shift"] = float(np.abs(out - z).sum())
    return out, info


def run_experiment(
    config: dict[str, Any],
    df: pd.DataFrame | None = None,
    records_dir: Path | None = None,
) -> dict[str, Any]:
    config = dict(config)
    exp_id = config.setdefault("id", f"exp-{uuid.uuid4().hex[:8]}")
    target_mode = config.get("target", "direct")
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target must be one of {TARGET_MODES}")
    t0 = time.time()

    if df is None:
        df = load_training(config.get("months"))
    dep = with_target(df)
    holdout = config.get("holdout_months", (1, 7))
    train, valid = split_by_month(dep, holdout)
    ctx_train, ctx_valid = split_by_month(df, holdout)

    feats, families = build_features(
        train,
        {"train": train, "valid": valid},
        config["features"],
        contexts={"train": ctx_train, "valid": ctx_valid},
        train_context=ctx_train,
    )
    Xtr, Xva = feats["train"], feats["valid"]
    ytr = train[TARGET].to_numpy(dtype=float)
    yva = valid[TARGET].to_numpy(dtype=float)

    mspec = dict(config.get("model", {"name": "lightgbm"}))
    mname = mspec.pop("name")
    model = MODELS[mname](**mspec)
    ztr = transform_target(target_mode, ytr, Xtr)
    clip_info = None
    if config.get("target_clip"):
        ztr, clip_info = clip_train_target(ztr, train, dict(config["target_clip"]))
    accepts = inspect.signature(model.fit).parameters
    kwargs: dict[str, Any] = {}
    if "valid" in accepts:
        kwargs["valid"] = (Xva, transform_target(target_mode, yva, Xva))
    if "valid_meta" in accepts:
        kwargs["valid_meta"] = valid
    model.fit(Xtr, ztr, train, **kwargs)
    pred = inverse_target(target_mode, model.predict(Xva, valid), Xva)
    lo, hi = config.get("clip", (0, None))
    pred = np.clip(pred, lo, hi)

    metrics = evaluate(valid, pred)
    preds_path = None
    if preds_dir := os.environ.get("PRC_HOLDOUT_PREDS_DIR"):
        preds_path = Path(preds_dir) / f"{exp_id}.npz"
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(preds_path, pred=pred.astype("float64"))
    record = {
        "id": exp_id,
        "hypothesis": config.get("hypothesis"),
        "config": config,
        "metrics": metrics,
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "feature_columns": list(Xtr.columns),
        "importance": model.importance() if hasattr(model, "importance") else None,
        "best_iteration": getattr(model, "best_iteration", None),
        "best_iterations": getattr(model, "iteration_tree", None),
        "target_clip": clip_info,
        "holdout_preds": str(preds_path) if preds_path else None,
        "seconds": round(time.time() - t0, 1),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    records_dir = records_dir or (WORK / "baseb" / "experiments")
    records_dir.mkdir(parents=True, exist_ok=True)
    (records_dir / f"{exp_id}.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())
