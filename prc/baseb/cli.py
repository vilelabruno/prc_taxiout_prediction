import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prc.baseb.config import ID_COL, TARGET
from prc.baseb.experiment import inverse_target, load_config, run_experiment, transform_target
from prc.baseb.features import build_features
from prc.baseb.loader import departures, load_ranking, load_submission_template, load_training, with_target
from prc.baseb.models import MODELS
from prc.baseb.submit import build_submission, write_submission
from prc.paths import ARTIFACTS, MODELS as MODELS_DIR

OUT = ARTIFACTS / "baseb"
MODEL_DIR = MODELS_DIR / "baseb"


def build(config_path):
    config = load_config(config_path)
    full = load_training(config.get("months"))
    train = with_target(full)
    ranking = load_ranking()
    rank_dep = departures(ranking)
    feats, families = build_features(
        train,
        {"train": train, "rank": rank_dep},
        config["features"],
        contexts={"train": full, "rank": ranking},
        train_context=full,
    )
    mspec = dict(config.get("model", {"name": "lightgbm"}))
    mname = mspec.pop("name")
    mode = config.get("target", "direct")
    model = MODELS[mname](**mspec)
    y = train[TARGET].to_numpy(dtype=float)
    model.fit(feats["train"], transform_target(mode, y, feats["train"]), train)
    pred = inverse_target(mode, model.predict(feats["rank"], rank_dep), feats["rank"])
    lo, hi = config.get("clip", (0, None))
    pred = np.clip(pred, lo, hi)
    series = pd.Series(pred, index=rank_dep[ID_COL].to_numpy())
    OUT.mkdir(parents=True, exist_ok=True)
    series.to_frame("pred").to_parquet(OUT / "pred_backup_final.parquet")
    sub = build_submission(series, load_submission_template())
    path = write_submission(sub, OUT / "base_b_v34.parquet")
    print(f"wrote {path}")
    for fam in families:
        fam.context = None
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "model.pkl", "wb") as fh:
        pickle.dump({"config": config, "families": families, "model": model}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {MODEL_DIR / 'model.pkl'}")


def holdout(config_path):
    rec = run_experiment(load_config(config_path))
    print(json.dumps({"id": rec["id"], "metrics": rec["metrics"]}, indent=2, default=str))


def v35(write):
    from prc.baseb import v35 as mod

    mod.main(write)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m prc.baseb.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("build", "holdout"):
        p = sub.add_parser(name)
        p.add_argument("config", type=Path)
    p = sub.add_parser("v35")
    p.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "build":
        build(args.config)
    elif args.cmd == "holdout":
        holdout(args.config)
    else:
        v35(args.write)


if __name__ == "__main__":
    sys.exit(main())
