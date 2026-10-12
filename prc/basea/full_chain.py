from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prc.basea import residual_chain as rc
from prc.basea import residual_chain_stack_adsb as chain
from prc.basea.residual_chain_stack_adsb import RESIDUAL_CAP, SEED, SPECIALIST_SEEDS, STAGE_WIDTH
from prc.paths import ARTIFACTS, MODELS, RANKING, SUBMITTING, TRAINING_DIR, WORK

COMPONENTS = ["cat70", "specialist_single", "light70", "cat93", "specialist_five",
              "light123", "specialist_arrival", "light153", "airport_experts",
              "cat153", "cat183"]
MODEL_DIR = MODELS / "basea"
HEAVY = ["cat93", "cat153", "cat183"]


def ids_hash(ids: pd.Series) -> str:
    return hashlib.sha256("|".join(map(str, ids)).encode()).hexdigest()[:16]


def prepare(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    dep, x = rc.build_matrix(rc.load_raw(args.data), rc.STAGES["following"])
    rdep, rx = rc.build_matrix(pd.read_parquet(args.ranking), rc.STAGES["following"])
    if list(x.columns) != list(rx.columns):
        raise ValueError("colunas de treino e ranking divergem")
    for name, frame in (("dep", dep), ("x", x), ("rdep", rdep), ("rx", rx)):
        frame.to_parquet(args.out / f"{name}.parquet")
        back = pd.read_parquet(args.out / f"{name}.parquet")
        if not back.dtypes.equals(frame.dtypes) or not back.index.equals(frame.index):
            raise ValueError(f"{name}: ida e volta do parquet alterou dtypes ou índice")
    print(json.dumps({"train": list(x.shape), "ranking": list(rx.shape),
                      "ids_sha": ids_hash(rdep[rc.ID])}))


def load(args: argparse.Namespace):
    dep, x, rdep, rx = (pd.read_parquet(args.matrices / f"{n}.parquet") for n in ("dep", "x", "rdep", "rx"))
    y = dep[rc.TARGET].astype(float)
    fit = y.ge(0)
    offset = rc.baseline(x)
    residual = y - offset
    return dep, x, y, fit, residual, rdep, rx, rc.baseline(rx)


def component(name: str, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    dep, x, y, fit, residual, rdep, rx, roffset = load(args)
    threads = args.threads
    scale = args.trees_scale

    def scaled(trees: int) -> int:
        return max(10, int(round(trees * scale)))

    lirf_rank = rc.lirf_scope(rx).to_numpy()
    lirf_fit = rc.lirf_scope(x) & fit
    ordinary_fit = fit & ~rc.lirf_scope(x)
    capped = residual.clip(-RESIDUAL_CAP, RESIDUAL_CAP)
    stage = lambda frame, n: frame.iloc[:, :STAGE_WIDTH[n]]
    cat = dict(depth=9, learning_rate=0.05, l2=7, borders=254, threads=threads)
    cbm = MODEL_DIR / f"{name}.cbm"
    txt = MODEL_DIR / f"{name}.txt"

    def specialist(fx, rxf, seeds):
        parts = [np.maximum(0.0, chain.fit_catboost(
            fx.loc[lirf_fit], residual.loc[lirf_fit], rxf, trees=scaled(1014), depth=5,
            learning_rate=0.04, l2=10, threads=threads, seed=seed,
            save_path=cbm if len(seeds) == 1 else MODEL_DIR / f"{name}_{seed}.cbm")
            + roffset.to_numpy()) for seed in seeds]
        return np.mean(parts, axis=0)

    if name == "cat70":
        out = chain.fit_catboost(stage(x, "carrier").loc[fit], residual.loc[fit], stage(rx, "carrier"),
                                 trees=scaled(2980), depth=9, learning_rate=0.045, l2=5, threads=threads, save_path=cbm)
    elif name == "light70":
        out = chain.fit_lightgbm(stage(x, "carrier").loc[fit], residual.loc[fit], stage(rx, "carrier"),
                                 trees=scaled(679), threads=threads, save_path=txt)
    elif name == "cat93":
        out = chain.fit_catboost(stage(x, "duration").loc[fit], residual.loc[fit], stage(rx, "duration"),
                                 trees=scaled(4998), **cat, save_path=cbm)
    elif name == "specialist_single":
        out = specialist(stage(x, "carrier"), stage(rx, "carrier"), [SEED])
    elif name == "specialist_five":
        out = specialist(stage(x, "carrier"), stage(rx, "carrier"), SPECIALIST_SEEDS)
    elif name == "light123":
        out = chain.fit_lightgbm(stage(x, "arrival").loc[fit], residual.loc[fit], stage(rx, "arrival"),
                                 trees=scaled(679), threads=threads, save_path=txt)
    elif name == "specialist_arrival":
        columns = list(stage(x, "carrier").columns) + [c for c in x.columns if c.startswith("completed_arrival_")]
        out = specialist(x[columns], rx[columns], SPECIALIST_SEEDS)
    elif name == "light153":
        out = chain.fit_lightgbm(stage(x, "neighbor").loc[fit], residual.loc[fit], stage(rx, "neighbor"),
                                 trees=scaled(679), threads=threads, save_path=txt)
    elif name == "airport_experts":
        out = np.full(len(rdep), np.nan)
        x123, r123 = stage(x, "arrival"), stage(rx, "arrival")
        rank_airport = rdep["ADEP_mvt"].to_numpy()
        for airport in sorted(dep["ADEP_mvt"].dropna().unique()):
            local_fit = ordinary_fit & dep["ADEP_mvt"].eq(airport)
            local_rank = rank_airport == airport
            if not local_fit.any() or not local_rank.any():
                continue
            local = chain.fit_lightgbm(x123.loc[local_fit], capped.loc[local_fit], r123.loc[local_rank],
                                       trees=scaled(679), threads=threads,
                                       save_path=MODEL_DIR / f"{name}_{airport}.txt")
            out[local_rank] = np.maximum(0.0, local + roffset.to_numpy()[local_rank])
    elif name == "cat153":
        out = chain.fit_catboost(stage(x, "neighbor").loc[ordinary_fit], capped.loc[ordinary_fit],
                                 stage(rx, "neighbor"), trees=scaled(4999), **cat, save_path=cbm)
    elif name == "cat183":
        out = chain.fit_catboost(stage(x, "following").loc[ordinary_fit], capped.loc[ordinary_fit],
                                 stage(rx, "following"), trees=scaled(4999), **cat, save_path=cbm)
    else:
        raise ValueError(name)
    meta = {"component": name, "trees_scale": scale, "threads": threads, "train_rows": int(fit.sum()),
            "ranking_rows": len(rdep), "ids_sha": ids_hash(rdep[rc.ID]), "lirf_rank": int(lirf_rank.sum())}
    return np.asarray(out, dtype="float64"), meta


def fit_one(args: argparse.Namespace) -> None:
    args.cache.mkdir(parents=True, exist_ok=True)
    target = args.cache / f"sub_{args.name}.npy"
    if target.exists():
        print(f"já existe: {target}")
        return
    started = time.time()
    values, meta = component(args.name, args)
    meta["seconds"] = round(time.time() - started, 1)
    np.save(args.cache / f"tmp_{args.name}.npy", values)
    (args.cache / f"sub_{args.name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    (args.cache / f"tmp_{args.name}.npy").rename(target)
    print(json.dumps(meta))


def assemble(args: argparse.Namespace) -> None:
    rdep, _ = rc.build_matrix(pd.read_parquet(args.ranking), rc.STAGES["following"])
    expected = ids_hash(rdep[rc.ID])
    for name in COMPONENTS:
        meta = json.loads((args.cache / f"sub_{name}.json").read_text())
        values = np.load(args.cache / f"sub_{name}.npy")
        if meta["ids_sha"] != expected or len(values) != len(rdep) or meta["trees_scale"] != args.trees_scale:
            raise ValueError(f"{name}: componente incompatível com o ranking/escala")
        if name != "airport_experts" and not np.isfinite(values).all():
            raise ValueError(f"{name}: valores não finitos")
    chain.make_submission(args.data, args.ranking, args.template, args.output,
                          args.threads, args.cache, args.trees_scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "fit", "assemble"):
        p = sub.add_parser(command)
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--trees-scale", type=float, default=1.2)
        if command == "prepare":
            p.add_argument("--data", type=Path, default=TRAINING_DIR)
            p.add_argument("--ranking", type=Path, default=RANKING)
            p.add_argument("--out", type=Path, default=WORK / "basea" / "matrices")
        elif command == "fit":
            p.add_argument("--name", choices=COMPONENTS, required=True)
            p.add_argument("--matrices", type=Path, default=WORK / "basea" / "matrices")
            p.add_argument("--cache", type=Path, default=ARTIFACTS / "base_a_full_final")
        else:
            p.add_argument("--data", type=Path, default=TRAINING_DIR)
            p.add_argument("--ranking", type=Path, default=RANKING)
            p.add_argument("--template", type=Path, default=SUBMITTING)
            p.add_argument("--cache", type=Path, default=ARTIFACTS / "base_a_full_final")
            p.add_argument("--output", type=Path, default=ARTIFACTS / "base_a_full_final" / "base_a_full_final.parquet")
    args = parser.parse_args()
    {"prepare": prepare, "fit": fit_one, "assemble": assemble}[args.command](args)


if __name__ == "__main__":
    main()
