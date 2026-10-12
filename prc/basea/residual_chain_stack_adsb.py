from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prc.basea import residual_chain as rc
from prc.paths import OOF, RANKING, SUBMITTING, TRAINING_DIR, WORK

SEED = 20260907
SPECIALIST_SEEDS = [20260907, 20260908, 20260909, 20260910, 20260911]
STAGE_WIDTH = {"base": 65, "carrier": 70, "duration": 93, "service": 97,
               "arrival": 123, "neighbor": 153, "following": 183}
RESIDUAL_CAP = 7200.0
VERSIONS = [f"v{n}" for n in range(2, 14)]


def catboost(trees: int, depth: int, learning_rate: float, threads: int,
             l2: float | None = None, borders: int | None = None, seed: int = SEED):
    from catboost import CatBoostRegressor
    params: dict[str, Any] = dict(
        iterations=trees, depth=depth, learning_rate=learning_rate, loss_function="RMSE",
        random_seed=seed, thread_count=threads, allow_writing_files=False, verbose=500)
    if l2 is not None:
        params["l2_leaf_reg"] = l2
    if borders is not None:
        params["border_count"] = borders
    return CatBoostRegressor(**params)


def lightgbm(trees: int, threads: int, seed: int = SEED):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=trees, num_leaves=63, learning_rate=0.035,
                             min_child_samples=80, reg_lambda=10.0, colsample_bytree=0.9,
                             random_state=seed, n_jobs=threads, verbose=-1, deterministic=True,
                             force_col_wise=True)


class ComponentCache:

    def __init__(self, directory: Path | None):
        self.directory = directory
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, name: str, compute) -> np.ndarray:
        if self.directory is None:
            return compute()
        path = self.directory / f"{name}.npy"
        if path.exists():
            print(f"           cache: {name}", flush=True)
            return np.load(path)
        values = compute()
        np.save(path, values)
        return values


def _categorical(frame: pd.DataFrame) -> list[str]:
    return [c for c in rc.CATEGORIES if c in frame.columns]


def _as_category(frames: list[pd.DataFrame], columns: list[str]) -> list[pd.DataFrame]:
    converted = [f.copy() for f in frames]
    for column in columns:
        union = pd.api.types.union_categoricals(
            [f[column].astype("category") for f in converted]).categories
        for frame in converted:
            frame[column] = pd.Categorical(frame[column], categories=union)
    return converted


def fit_catboost(x_fit, y_fit, x_eval, *, trees, depth, learning_rate, threads,
                 l2=None, borders=None, seed=SEED, save_path=None) -> np.ndarray:
    model = catboost(trees, depth, learning_rate, threads, l2, borders, seed)
    model.fit(x_fit, y_fit, cat_features=_categorical(x_fit))
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(save_path))
    return np.asarray(model.predict(x_eval), dtype="float64")


def fit_lightgbm(x_fit, y_fit, x_eval, *, trees, threads, seed=SEED, save_path=None) -> np.ndarray:
    columns = _categorical(x_fit)
    fitted, evaluated = _as_category([x_fit, x_eval], columns)
    model = lightgbm(trees, threads, seed)
    model.fit(fitted, y_fit, categorical_feature=columns)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(save_path))
    return np.asarray(model.predict(evaluated), dtype="float64")


def clock_overlay(dep: pd.DataFrame, x: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
    actual = x["movement_minus_AOBT_3_flt"]
    last = x["movement_minus_LOBT_flt"]
    matches = x["nm_departure_matches"].eq(1)
    disagreement = (pd.to_datetime(dep["AOBT_3_flt"], utc=True, errors="coerce")
                    - pd.to_datetime(dep["LOBT_flt"], utc=True, errors="coerce")).dt.total_seconds()
    trigger = (~rc.lirf_scope(x) & matches
               & actual.notna() & actual.between(rc.VALID_LOW, rc.VALID_HIGH)
               & last.notna() & last.between(rc.VALID_LOW, rc.VALID_HIGH)
               & disagreement.gt(7200))
    result = prediction.copy()
    result[trigger.to_numpy()] = last[trigger].clip(lower=0).to_numpy()
    return result, int(trigger.sum())


def _planning_time(dep: pd.DataFrame) -> pd.Series:
    candidates = ("LOBT_flt", "IOBT_flt", "EOBT_1_flt", "SCHED_TIME_UTC_mvt")
    value = pd.Series(pd.NaT, index=dep.index, dtype="datetime64[ns, UTC]")
    for column in candidates:
        if column in dep:
            parsed = pd.to_datetime(dep[column], utc=True, errors="coerce")
            value = value.fillna(parsed)
    return value


def _past_window_count(time: pd.Series, groups: pd.DataFrame, seconds: int) -> np.ndarray:
    result = np.full(len(time), np.nan, dtype="float32")
    valid = time.notna() & ~groups.isna().any(axis=1)
    if not valid.any():
        return result
    work = groups.loc[valid].copy()
    work["_time"] = time.loc[valid].astype("int64")
    work["_row"] = np.flatnonzero(valid.to_numpy())
    keys = list(groups.columns)
    work = work.sort_values(keys + ["_time"], kind="mergesort")
    window = np.int64(seconds) * 1_000_000_000
    for _, part in work.groupby(keys, sort=False, observed=True):
        stamps = part["_time"].to_numpy(dtype="int64", copy=False)
        earlier = np.searchsorted(stamps, stamps, side="left")
        left = np.searchsorted(stamps, stamps - window, side="left")
        result[part["_row"].to_numpy(dtype=int, copy=False)] = (earlier - left).astype("float32")
    return result


def _past_group_mean(time: pd.Series, groups: pd.DataFrame, values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    mean = np.full(len(time), np.nan, dtype="float32")
    count_out = np.zeros(len(time), dtype="float32")
    valid = time.notna() & values.notna() & ~groups.isna().any(axis=1)
    if not valid.any():
        return mean, count_out
    keys = list(groups.columns)
    work = groups.loc[valid].copy()
    work["_time"] = time.loc[valid].astype("int64")
    work["_value"] = values.loc[valid].astype("float64")
    aggregate = work.groupby(keys + ["_time"], observed=True, sort=False)["_value"].agg(["sum", "count"]).reset_index()
    aggregate = aggregate.sort_values(keys + ["_time"], kind="mergesort")
    aggregate["_prior_sum"] = aggregate.groupby(keys, observed=True)["sum"].cumsum() - aggregate["sum"]
    aggregate["_prior_count"] = aggregate.groupby(keys, observed=True)["count"].cumsum() - aggregate["count"]
    mapped = work.merge(aggregate[keys + ["_time", "_prior_sum", "_prior_count"]], on=keys + ["_time"], how="left", sort=False)
    rows = np.flatnonzero(valid.to_numpy())
    counts = mapped["_prior_count"].to_numpy(dtype="float32", copy=False)
    count_out[rows] = counts
    with np.errstate(divide="ignore", invalid="ignore"):
        mean[rows] = (mapped["_prior_sum"].to_numpy(dtype="float64", copy=False) / counts).astype("float32")
    return mean, count_out


def add_operational_pressure_features(dep: pd.DataFrame, x: pd.DataFrame,
                                      enhanced: pd.DataFrame) -> pd.DataFrame:
    planned = _planning_time(dep)
    airport = x.get("ADEP_mvt", pd.Series(np.nan, index=x.index)).astype("string")
    carrier = x.get("movement_carrier", pd.Series(np.nan, index=x.index)).astype("string")
    stand = x.get("STAND_mvt", pd.Series(np.nan, index=x.index)).astype("string")
    runway = x.get("RUNWAY_mvt", pd.Series(np.nan, index=x.index)).astype("string")
    regimes = {
        "apt": pd.DataFrame({"airport": airport}),
        "apt_carrier": pd.DataFrame({"airport": airport, "carrier": carrier}),
        "apt_stand": pd.DataFrame({"airport": airport, "stand": stand}),
        "apt_runway": pd.DataFrame({"airport": airport, "runway": runway}),
    }
    for minutes in (15, 30, 60):
        suffix = f"{minutes}m"
        airport_count = _past_window_count(planned, regimes["apt"], minutes * 60)
        carrier_count = _past_window_count(planned, regimes["apt_carrier"], minutes * 60)
        enhanced[f"sched_pressure_apt_{suffix}"] = airport_count
        enhanced[f"sched_pressure_apt_carrier_{suffix}"] = carrier_count
        enhanced[f"sched_pressure_apt_stand_{suffix}"] = _past_window_count(planned, regimes["apt_stand"], minutes * 60)
        enhanced[f"sched_pressure_apt_runway_{suffix}"] = _past_window_count(planned, regimes["apt_runway"], minutes * 60)
        enhanced[f"sched_share_apt_carrier_{suffix}"] = np.divide(
            carrier_count, airport_count, out=np.zeros(len(x), dtype="float32"), where=airport_count > 0
        )

    metrics: dict[str, pd.Series] = {}
    if "adsb_dist" in enhanced:
        metrics["dist"] = enhanced["adsb_dist"].clip(lower=0)
    if {"adsb_disp", "adsb_dist"}.issubset(enhanced):
        metrics["directness"] = (enhanced["adsb_disp"] / enhanced["adsb_dist"].clip(lower=1.0)).clip(0, 1)
    if "adsb_mov_time" in enhanced:
        metrics["mov_time"] = enhanced["adsb_mov_time"].clip(lower=0)
    for name, values in metrics.items():
        stand_mean, stand_n = _past_group_mean(planned, regimes["apt_stand"], values)
        apt_mean, apt_n = _past_group_mean(planned, regimes["apt"], values)
        enhanced[f"adsb_hist_apt_stand_{name}"] = stand_mean
        enhanced[f"adsb_hist_apt_{name}"] = apt_mean
        enhanced[f"adsb_hist_apt_stand_{name}_n"] = stand_n
        enhanced[f"adsb_hist_apt_{name}_delta"] = (stand_mean - apt_mean).astype("float32")
        if name != "directness":
            enhanced[f"adsb_hist_apt_stand_{name}_ratio"] = np.divide(
                stand_mean, apt_mean, out=np.full(len(x), np.nan, dtype="float32"), where=apt_mean > 0
            )
    return enhanced


def append_adsb(dep: pd.DataFrame, x: pd.DataFrame, path: Path | None,
                history: tuple[pd.DataFrame, pd.DataFrame, list[str]] | None = None) -> tuple[pd.DataFrame, list[str]]:
    if path is None:
        return x, []
    adsb = pd.read_parquet(path)
    if rc.ID not in adsb:
        raise ValueError(f"{path} has no {rc.ID}")
    features = [c for c in adsb.columns if c != rc.ID]
    aligned = dep[[rc.ID]].merge(adsb, on=rc.ID, how="left", sort=False)[features]
    aligned.index = x.index
    enhanced = aligned.copy()
    coverage = enhanced.get("adsb_cov", pd.Series(0.0, index=enhanced.index)).fillna(0.0)
    enhanced["adsb_available"] = (coverage > 0).astype("int8")
    for col in features:
        if enhanced[col].isna().any():
            enhanced[f"adsb_missing_{col}"] = enhanced[col].isna().astype("int8")

    clock_cols = [c for c in ("ltxfm_m_aobt", "ltx0_m_aobt", "txfm_m_aobt", "tx0_m_aobt")
                  if c in enhanced]
    for col in clock_cols:
        clipped = enhanced[col].clip(-7200, 7200)
        enhanced[f"phys_abs_{col}"] = clipped.abs().astype("float32")
        enhanced[f"phys_sq_{col}"] = (clipped * clipped).astype("float32")
        enhanced[f"phys_cov_{col}"] = (coverage * enhanced[col].fillna(0.0)).astype("float32")
        for nm in ("mvt_aobt", "aobt_eobt", "aobt_lobt", "aobt_sched"):
            if nm in x:
                enhanced[f"phys_x_{col}_{nm}"] = (enhanced[col].fillna(0.0) * x[nm].clip(-7200, 7200)).astype("float32")

    if {"adsb_dist", "adsb_mov_time"}.issubset(enhanced):
        enhanced["adsb_effective_speed"] = (
            enhanced["adsb_dist"] / enhanced["adsb_mov_time"].clip(lower=1.0)
        ).clip(0, 100).astype("float32")
    if {"adsb_stat_time", "adsb_mov_time"}.issubset(enhanced):
        enhanced["adsb_stop_share"] = (
            enhanced["adsb_stat_time"] /
            (enhanced["adsb_stat_time"] + enhanced["adsb_mov_time"]).clip(lower=1.0)
        ).clip(0, 1).astype("float32")
    if {"adsb_disp", "adsb_dist"}.issubset(enhanced):
        enhanced["adsb_path_directness"] = (
            enhanced["adsb_disp"] / enhanced["adsb_dist"].clip(lower=1.0)
        ).clip(0, 1).astype("float32")
    for traffic in ("scheduled_departures_30m", "movement_departures_30m",
                    "airport_dep_previous_15m", "airport_dep_previous_30m",
                    "runway_dep_previous_15m", "runway_dep_previous_30m"):
        if traffic in x:
            enhanced[f"adsb_cov_x_{traffic}"] = (
                coverage * np.log1p(x[traffic].clip(lower=0))
            ).astype("float32")
            if "ox_q_ahead" in enhanced:
                enhanced[f"adsb_queue_x_{traffic}"] = (
                    enhanced["ox_q_ahead"].fillna(0).clip(0, 100) * np.log1p(x[traffic].clip(lower=0))
                ).astype("float32")

    bins = pd.cut(coverage, [-np.inf, 0, .25, .75, np.inf], labels=["none", "low", "mid", "high"]).astype(str)
    for key in ("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "movement_carrier"):
        if key in x:
            name = f"adsb_regime_{key}"
            enhanced[name] = x[key].fillna("UNKNOWN").astype(str) + "|" + bins
            if name not in rc.CATEGORIES:
                rc.CATEGORIES.append(name)

    if history is None:
        enhanced = add_operational_pressure_features(dep, x, enhanced)
    else:
        history_dep, history_x, _ = history
        history_base = history_x.reindex(columns=x.columns)
        history_enhanced = history_x.reindex(columns=enhanced.columns)
        joined_dep = pd.concat([history_dep, dep], ignore_index=True)
        joined_x = pd.concat([history_base, x], ignore_index=True)
        joined_enhanced = pd.concat([history_enhanced, enhanced], ignore_index=True)
        enriched = add_operational_pressure_features(joined_dep, joined_x, joined_enhanced)
        enhanced = enriched.iloc[-len(dep):].set_axis(x.index, axis=0)

    enhanced = enhanced.replace([np.inf, -np.inf], np.nan)
    result = pd.concat([x, enhanced], axis=1)
    return result, list(enhanced.columns)


def run(data: Path, threads: int, upto: str, report_path: Path,
        trees_scale: float = 1.0, cache_dir: Path | None = None,
        adsb_path: Path | None = None) -> dict[str, Any]:
    started = time.time()
    cached = ComponentCache(cache_dir)

    def log(message: str) -> None:
        print(f"[{time.time() - started:7.1f}s] {message}", flush=True)

    def scaled(trees: int) -> int:
        return max(10, int(round(trees * trees_scale)))

    log("construindo a matriz-base e as features ADS-B")
    dep, x = rc.build_matrix(rc.load_raw(data), rc.STAGES["following"])
    x, adsb_columns = append_adsb(dep, x, adsb_path)
    if x.shape[1] < STAGE_WIDTH["following"]:
        raise ValueError(f"matriz-base incompleta: {x.shape}")
    y = dep[rc.TARGET].astype(float)
    clock = pd.to_datetime(dep[rc.TIME], utc=True)
    holdout = clock.dt.month.isin([1, 7])
    fit = ~holdout & y.ge(0)
    offset = rc.baseline(x)
    residual = y - offset
    lirf = rc.lirf_scope(x)
    log(f"matriz {x.shape} | fit {int(fit.sum()):,} | holdout {int(holdout.sum()):,} "
        f"| escopo LIRF {int(lirf.sum()):,} (fit {int((lirf & fit).sum()):,})")

    y_hold = y[holdout].to_numpy()
    base_hold = offset[holdout].to_numpy()
    lirf_hold = lirf[holdout].to_numpy()
    ordinary = ~lirf_hold
    scores: dict[str, Any] = {}

    def stage(name: str):
        width = STAGE_WIDTH[name]
        base = x.iloc[:, :width]
        return pd.concat([base, x[adsb_columns]], axis=1) if adsb_columns else base

    def record(version: str, prediction: np.ndarray) -> np.ndarray:
        prediction = np.maximum(0.0, prediction)
        entry = {
            "rmse": rc.rmse(y_hold, prediction),
            "jan": rc.rmse(y_hold[clock[holdout].dt.month.eq(1).to_numpy()],
                           prediction[clock[holdout].dt.month.eq(1).to_numpy()]),
            "jul": rc.rmse(y_hold[clock[holdout].dt.month.eq(7).to_numpy()],
                           prediction[clock[holdout].dt.month.eq(7).to_numpy()]),
            "lirf": rc.rmse(y_hold[lirf_hold], prediction[lirf_hold]) if lirf_hold.any() else None,
        }
        scores[version] = entry
        log(f"{version}: RMSE {entry['rmse']:.4f} (jan {entry['jan']:.3f} · jul {entry['jul']:.3f}"
            + (f" · LIRF {entry['lirf']:.1f}" if entry["lirf"] else "") + ")")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
        return prediction

    def residual_to_prediction(values: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, values + base_hold)

    def stop(version: str) -> bool:
        return VERSIONS.index(version) >= VERSIONS.index(upto)

    log("v2: CatBoost65 (depth 8, lr .055)")
    x65 = stage("base")
    cat65 = cached("cat65", lambda: fit_catboost(
        x65.loc[fit], residual.loc[fit], x65.loc[holdout],
        trees=scaled(2200), depth=8, learning_rate=0.055, threads=threads))
    v2 = record("v2", residual_to_prediction(cat65))
    if stop("v2"):
        return scores

    log("v3: Cat70 global (depth 9, lr .045, L2 5) + especialista LIRF (depth 5)")
    x70 = stage("carrier")
    cat70 = residual_to_prediction(cached("cat70", lambda: fit_catboost(
        x70.loc[fit], residual.loc[fit], x70.loc[holdout],
        trees=scaled(2980), depth=9, learning_rate=0.045, l2=5, threads=threads)))
    lirf_fit = fit & lirf

    def specialist(features: pd.DataFrame, seeds: list[int]) -> np.ndarray:
        parts = [np.maximum(0.0, fit_catboost(
            features.loc[lirf_fit], residual.loc[lirf_fit], features.loc[holdout],
            trees=scaled(1014), depth=5, learning_rate=0.04, l2=10, threads=threads, seed=seed)
            + base_hold) for seed in seeds]
        return np.mean(parts, axis=0)

    single = cached("specialist_single", lambda: specialist(x70, [SEED]))
    v3 = cat70.copy()
    v3[lirf_hold] = single[lirf_hold]
    v3 = record("v3", v3)
    if stop("v3"):
        return scores

    log("v4: + LGBM70 (63 folhas, 679 árvores) a 25%")
    light70 = residual_to_prediction(cached("light70", lambda: fit_lightgbm(
        x70.loc[fit], residual.loc[fit], x70.loc[holdout],
        trees=scaled(679), threads=threads)))
    v4 = 0.25 * np.maximum(0.0, light70) + 0.75 * np.maximum(0.0, cat70)
    v4[lirf_hold] = single[lirf_hold]
    v4 = record("v4", v4)
    if stop("v4"):
        return scores

    log("v5: + Cat93 duração (depth 9, lr .05, L2 7, 254 borders)")
    x93 = stage("duration")
    cat93 = residual_to_prediction(cached("cat93", lambda: fit_catboost(
        x93.loc[fit], residual.loc[fit], x93.loc[holdout],
        trees=scaled(4998), depth=9, learning_rate=0.05, l2=7, borders=254, threads=threads)))
    v5 = 0.25 * v4 + 0.75 * np.maximum(0.0, cat93)
    v5[lirf_hold] = single[lirf_hold]
    v5 = record("v5", v5)
    if stop("v5"):
        return scores

    log("v6: especialista LIRF com cinco seeds")
    five = cached("specialist_five", lambda: specialist(x70, SPECIALIST_SEEDS))
    v6 = v5.copy()
    v6[lirf_hold] = five[lirf_hold]
    v6 = record("v6", v6)
    if stop("v6"):
        return scores

    log("v7: + LGBM123 com chegadas concluídas, e especialista de chegadas")
    x123 = stage("arrival")
    light123 = residual_to_prediction(cached("light123", lambda: fit_lightgbm(
        x123.loc[fit], residual.loc[fit], x123.loc[holdout],
        trees=scaled(679), threads=threads)))
    arrival_columns = list(x70.columns) + [c for c in x.columns if c.startswith("completed_arrival_")]
    arrival_specialist = cached("specialist_arrival",
                                lambda: specialist(x[arrival_columns], SPECIALIST_SEEDS))
    v7 = 0.25 * np.maximum(0.0, light123) + 0.75 * v6
    v7[lirf_hold] = 0.5 * arrival_specialist[lirf_hold] + 0.5 * v6[lirf_hold]
    v7 = record("v7", v7)
    if stop("v7"):
        return scores

    log("v8: + Cat123 profundo (resíduo limitado a ±7200 s), mistura 50/50")
    ordinary_fit = fit & ~lirf
    capped = residual.clip(-RESIDUAL_CAP, RESIDUAL_CAP)
    cat123 = residual_to_prediction(cached("cat123", lambda: fit_catboost(
        x123.loc[ordinary_fit], capped.loc[ordinary_fit], x123.loc[holdout],
        trees=scaled(4999), depth=9, learning_rate=0.05, l2=7, borders=254, threads=threads)))
    v8 = v7.copy()
    v8[ordinary] = 0.5 * v7[ordinary] + 0.5 * np.maximum(0.0, cat123)[ordinary]
    v8 = record("v8", v8)
    if stop("v8"):
        return scores

    log("v9: + experts por aeroporto e LGBM153 com vizinhos NM, a 25%")
    x153 = stage("neighbor")
    light153 = residual_to_prediction(cached("light153", lambda: fit_lightgbm(
        x153.loc[fit], residual.loc[fit], x153.loc[holdout],
        trees=scaled(679), threads=threads)))
    airport_hold = dep.loc[holdout, "ADEP_mvt"].to_numpy()

    def airport_experts() -> np.ndarray:
        values = np.full(len(y_hold), np.nan)
        for airport in sorted(dep["ADEP_mvt"].dropna().unique()):
            local_fit = ordinary_fit & dep["ADEP_mvt"].eq(airport)
            local_hold = airport_hold == airport
            if not local_fit.any() or not local_hold.any():
                continue
            local = fit_lightgbm(x123.loc[local_fit], capped.loc[local_fit],
                                 x123.loc[holdout].loc[local_hold], trees=scaled(679),
                                 threads=threads)
            values[local_hold] = np.maximum(0.0, local + base_hold[local_hold])
        return values

    experts = cached("airport_experts", airport_experts)
    experts = np.where(np.isnan(experts), np.maximum(0.0, light153), experts)
    combined = 0.5 * experts + 0.5 * np.maximum(0.0, light153)
    v9 = v8.copy()
    v9[ordinary] = 0.75 * v8[ordinary] + 0.25 * combined[ordinary]
    v9 = record("v9", v9)
    if stop("v9"):
        return scores

    log("v10: + Cat153 substituindo Cat123 no peso .375")
    cat153 = residual_to_prediction(cached("cat153", lambda: fit_catboost(
        x153.loc[ordinary_fit], capped.loc[ordinary_fit], x153.loc[holdout],
        trees=scaled(4999), depth=9, learning_rate=0.05, l2=7, borders=254, threads=threads)))
    v10 = v9.copy()
    v10[ordinary] = (v9 + 0.375 * (np.maximum(0.0, cat153) - np.maximum(0.0, cat123)))[ordinary]
    v10 = record("v10", v10)
    if stop("v10"):
        return scores

    v11, triggered = clock_overlay(dep.loc[holdout], x.loc[holdout], v10)
    log(f"v11: regra de clock overlay disparou em {triggered} linhas")
    v11 = record("v11", v11)
    if stop("v11"):
        return scores

    log("v12: + Cat183 com vizinhos seguintes, substituindo Cat153 no peso .375")
    x183 = stage("following")
    cat183 = residual_to_prediction(cached("cat183", lambda: fit_catboost(
        x183.loc[ordinary_fit], capped.loc[ordinary_fit], x183.loc[holdout],
        trees=scaled(4999), depth=9, learning_rate=0.05, l2=7, borders=254, threads=threads)))
    protected = lirf_hold | (v11 != v10)
    v12 = v11.copy()
    change = 0.375 * (np.maximum(0.0, cat183) - np.maximum(0.0, cat153))
    v12[~protected] = (v11 + change)[~protected]
    v12 = record("v12", v12)
    if stop("v12"):
        return scores

    log("v13: especialista dedicado (linhas sem AOBT válido)")
    aobt_valid = (x["movement_minus_AOBT_3_flt"].notna()
                  & x["movement_minus_AOBT_3_flt"].between(rc.VALID_LOW, rc.VALID_HIGH)
                  & x["nm_departure_matches"].eq(1))
    aobt_invalid_fit = fit & ~aobt_valid & ~rc.lirf_scope(x)
    aobt_invalid_hold = ((~aobt_valid) & ~rc.lirf_scope(x))[holdout].to_numpy()
    aobt_specialist = cached("aobt_specialist", lambda: fit_catboost(
        x65.loc[aobt_invalid_fit], residual.loc[aobt_invalid_fit], x65.loc[holdout],
        trees=scaled(800), depth=8, learning_rate=0.055, threads=threads))
    aobt_specialist = residual_to_prediction(aobt_specialist)
    v13 = v12.copy()
    v13[aobt_invalid_hold] = aobt_specialist[aobt_invalid_hold]
    record("v13", v13)
    return scores


def make_submission(data: Path, ranking_path: Path, template_path: Path, output: Path,
                    threads: int, cache_dir: Path | None, trees_scale: float = 1.0,
                    adsb_path: Path | None = None) -> Path:
    if output.exists():
        raise ValueError(f"{output} já existe; use um caminho novo")
    started = time.time()
    cached = ComponentCache(cache_dir)

    def log(message: str) -> None:
        print(f"[{time.time() - started:7.1f}s] {message}", flush=True)

    def scaled(trees: int) -> int:
        return max(10, int(round(trees * trees_scale)))

    log("construindo a matriz de treino full e as features ADS-B")
    dep, x = rc.build_matrix(rc.load_raw(data), rc.STAGES["following"])
    x, adsb_columns = append_adsb(dep, x, adsb_path)
    y = dep[rc.TARGET].astype(float)
    fit = y.ge(0)
    offset = rc.baseline(x)
    residual = y - offset
    lirf_fit_scope = rc.lirf_scope(x) & fit

    log("construindo a matriz do ranking e as features ADS-B")
    ranking_dep, ranking_x = rc.build_matrix(pd.read_parquet(ranking_path), rc.STAGES["following"])
    ranking_x, ranking_adsb_columns = append_adsb(ranking_dep, ranking_x, adsb_path)
    if adsb_columns != ranking_adsb_columns:
        raise ValueError("as colunas ADS-B do treino e ranking divergem")
    ranking_offset = rc.baseline(ranking_x)
    lirf_rank = rc.lirf_scope(ranking_x).to_numpy()
    ordinary_rank = ~lirf_rank

    def stage(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        base = frame.iloc[:, :STAGE_WIDTH[name]]
        return pd.concat([base, frame[adsb_columns]], axis=1) if adsb_columns else base

    def to_prediction(values: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, values + ranking_offset.to_numpy())

    def specialist(features_fit: pd.DataFrame, features_rank: pd.DataFrame, seeds: list[int]) -> np.ndarray:
        parts = [np.maximum(0.0, fit_catboost(
            features_fit.loc[lirf_fit_scope], residual.loc[lirf_fit_scope], features_rank,
            trees=scaled(1014), depth=5, learning_rate=0.04, l2=10, threads=threads, seed=seed)
            + ranking_offset.to_numpy()) for seed in seeds]
        return np.mean(parts, axis=0)

    capped = residual.clip(-RESIDUAL_CAP, RESIDUAL_CAP)
    ordinary_fit = fit & ~rc.lirf_scope(x)

    x65, x70, x93, x123, x153, x183 = (stage(x, n) for n in
                                        ("base", "carrier", "duration", "arrival", "neighbor", "following"))
    r65, r70, r93, r123, r153, r183 = (stage(ranking_x, n) for n in
                                        ("base", "carrier", "duration", "arrival", "neighbor", "following"))

    log("v2: CatBoost65")
    cat65 = to_prediction(cached("sub_cat65", lambda: fit_catboost(
        x65.loc[fit], residual.loc[fit], r65, trees=scaled(2200), depth=8, learning_rate=0.055, threads=threads)))
    log("v3: Cat70 + especialista LIRF")
    cat70 = to_prediction(cached("sub_cat70", lambda: fit_catboost(
        x70.loc[fit], residual.loc[fit], r70, trees=scaled(2980), depth=9, learning_rate=0.045, l2=5, threads=threads)))
    single = cached("sub_specialist_single", lambda: specialist(x70, r70, [SEED]))
    v3 = cat70.copy()
    v3[lirf_rank] = single[lirf_rank]

    log("v4: + LGBM70 a 25%")
    light70 = to_prediction(cached("sub_light70", lambda: fit_lightgbm(
        x70.loc[fit], residual.loc[fit], r70, trees=scaled(679), threads=threads)))
    v4 = 0.25 * np.maximum(0.0, light70) + 0.75 * np.maximum(0.0, cat70)
    v4[lirf_rank] = single[lirf_rank]

    log("v5: + Cat93 duração")
    cat93 = to_prediction(cached("sub_cat93", lambda: fit_catboost(
        x93.loc[fit], residual.loc[fit], r93, trees=scaled(4998), depth=9, learning_rate=0.05, l2=7, borders=254, threads=threads)))
    v5 = 0.25 * v4 + 0.75 * np.maximum(0.0, cat93)
    v5[lirf_rank] = single[lirf_rank]

    log("v6: especialista com cinco seeds")
    five = cached("sub_specialist_five", lambda: specialist(x70, r70, SPECIALIST_SEEDS))
    v6 = v5.copy()
    v6[lirf_rank] = five[lirf_rank]

    log("v7: + LGBM123 e especialista de chegadas")
    light123 = to_prediction(cached("sub_light123", lambda: fit_lightgbm(
        x123.loc[fit], residual.loc[fit], r123, trees=scaled(679), threads=threads)))
    arrival_columns = list(x70.columns) + [c for c in x.columns if c.startswith("completed_arrival_")]
    arrival_specialist = cached("sub_specialist_arrival",
                                lambda: specialist(x[arrival_columns], ranking_x[arrival_columns], SPECIALIST_SEEDS))
    v7 = 0.25 * np.maximum(0.0, light123) + 0.75 * v6
    v7[lirf_rank] = 0.5 * arrival_specialist[lirf_rank] + 0.5 * v6[lirf_rank]

    log("v9: + experts por aeroporto e LGBM153")
    light153 = to_prediction(cached("sub_light153", lambda: fit_lightgbm(
        x153.loc[fit], residual.loc[fit], r153, trees=scaled(679), threads=threads)))

    def airport_experts() -> np.ndarray:
        values = np.full(len(ranking_dep), np.nan)
        rank_airport = ranking_dep["ADEP_mvt"].to_numpy()
        for airport in sorted(dep["ADEP_mvt"].dropna().unique()):
            local_fit = ordinary_fit & dep["ADEP_mvt"].eq(airport)
            local_rank = rank_airport == airport
            if not local_fit.any() or not local_rank.any():
                continue
            local = fit_lightgbm(x123.loc[local_fit], capped.loc[local_fit], r123.loc[local_rank],
                                 trees=scaled(679), threads=threads)
            values[local_rank] = np.maximum(0.0, local + ranking_offset.to_numpy()[local_rank])
        return values

    experts = cached("sub_airport_experts", airport_experts)
    experts = np.where(np.isnan(experts), np.maximum(0.0, light153), experts)
    combined = 0.5 * experts + 0.5 * np.maximum(0.0, light153)

    log("v10: Cat153 no lugar de Cat123")
    cat153 = to_prediction(cached("sub_cat153", lambda: fit_catboost(
        x153.loc[ordinary_fit], capped.loc[ordinary_fit], r153,
        trees=scaled(4999), depth=9, learning_rate=0.05, l2=7, borders=254, threads=threads)))
    v10 = v7.copy()
    v10[ordinary_rank] = (0.375 * v7 + 0.25 * combined + 0.375 * np.maximum(0.0, cat153))[ordinary_rank]

    log("v11: regra de clock overlay")
    v11, triggered = clock_overlay(ranking_dep, ranking_x, v10)
    log(f"v11: disparou em {triggered} linhas")

    log("v12: + Cat183 com vizinhos seguintes")
    cat183 = to_prediction(cached("sub_cat183", lambda: fit_catboost(
        x183.loc[ordinary_fit], capped.loc[ordinary_fit], r183,
        trees=scaled(4999), depth=9, learning_rate=0.05, l2=7, borders=254, threads=threads)))
    protected = lirf_rank | (v11 != v10)
    v12 = v11.copy()
    change = 0.375 * (np.maximum(0.0, cat183) - np.maximum(0.0, cat153))
    v12[~protected] = (v11 + change)[~protected]

    template = pd.read_parquet(template_path)
    result = template[[rc.ID, rc.TARGET]].copy()
    result[rc.TARGET] = template[rc.ID].map(pd.Series(v12, index=ranking_dep[rc.ID].to_numpy())).astype(float)
    if result[rc.TARGET].isna().any() or (result[rc.TARGET] < 0).any():
        raise ValueError("Previsões faltando ou negativas")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    log(f"{output} — {len(result):,} linhas, média {result[rc.TARGET].mean():.1f}s, zerados {(result[rc.TARGET] == 0).sum()}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=TRAINING_DIR)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--upto", choices=VERSIONS, default="v13")
    parser.add_argument("--trees-scale", type=float, default=1.0)
    parser.add_argument("--report", type=Path, default=WORK / "basea" / "residual_chain_progress.json")
    parser.add_argument("--cache", type=Path, default=OOF / "basea")
    parser.add_argument("--adsb", type=Path, default=None)
    parser.add_argument("--make-submission", type=Path, default=None)
    parser.add_argument("--ranking", type=Path, default=RANKING)
    parser.add_argument("--template", type=Path, default=SUBMITTING)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.make_submission is not None:
        make_submission(args.data, args.ranking, args.template, args.make_submission,
                        args.threads, args.cache, args.trees_scale, args.adsb)
        return
    run(args.data, args.threads, args.upto, args.report, args.trees_scale, args.cache, args.adsb)


if __name__ == "__main__":
    main()
