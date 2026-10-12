from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prc.paths import DATA_ROOT

ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
PHASE = "PHASE_mvt"
TIME = "SCHED_TIME_UTC_mvt"
MOVEMENT = "MVT_TIME_UTC_mvt"

BASE_CATEGORIES = [
    "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt",
    "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
]
CONTEXT_CATEGORIES = ["RUNWAY_mvt", "airport_RUNWAY_mvt", "STAND_mvt", "airport_STAND_mvt"]
EXTRA_CATEGORIES = ["airport_operator", "stand_runway", "ADEP_flt", "ADES_flt",
                    "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt"]
CATEGORIES = BASE_CATEGORIES + CONTEXT_CATEGORIES + EXTRA_CATEGORIES + [
    "movement_carrier", "airport_movement_carrier",
    "airport_route", "airport_aircraft", "airport_carrier_runway",
    "movement_service", "nm_service", "airport_movement_service", "airport_nm_service",
]

PLANNED_CLOCKS = ["IOBT_flt", "EOBT_1_flt"]
OBSERVED_CLOCKS = ["AOBT_3_flt", "LOBT_flt", "ARVT_1_flt", "ARVT_3_flt"]
TRAFFIC_WINDOWS = [15, 30, 60]
VALID_LOW, VALID_HIGH = -7200.0, 172800.0
DEFAULT_BASELINE_SECONDS = 1000.0


def _epoch_seconds(values: pd.Series) -> np.ndarray:
    return pd.to_datetime(values, utc=True).dt.as_unit("ns").astype("int64").to_numpy() / 1e9


def rmse(actual: Any, predicted: Any) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def load_raw(data: Path) -> pd.DataFrame:
    monthly = sorted(data.glob("training_*.parquet"))
    if len(monthly) == 12:
        return pd.concat([pd.read_parquet(p) for p in monthly], ignore_index=True)
    consolidated = DATA_ROOT / "processed" / "training_all_2025.parquet"
    if consolidated.exists():
        return pd.read_parquet(consolidated)
    raise FileNotFoundError("Esperava doze training_*.parquet ou o consolidado processado")


def departures(raw: pd.DataFrame) -> pd.DataFrame:
    result = raw.loc[raw[PHASE].eq("DEP")].copy()
    result[TIME] = pd.to_datetime(result[TIME], utc=True)
    return result


def movement_context(dep: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({
        "phase": context[PHASE].to_numpy(),
        "airport": np.where(context[PHASE].eq("DEP"), context["ADEP_mvt"], context["ADES_mvt"]),
        "runway": context["RUNWAY_mvt"].fillna("UNKNOWN").astype(str).to_numpy(),
        "seconds": _epoch_seconds(context[MOVEMENT]),
    })
    times = _epoch_seconds(dep[MOVEMENT])
    query = pd.DataFrame({
        "airport": dep["ADEP_mvt"].fillna("UNKNOWN").astype(str).to_numpy(),
        "runway": dep["RUNWAY_mvt"].fillna("UNKNOWN").astype(str).to_numpy(),
    })
    out = pd.DataFrame(index=dep.index)
    for phase in ["DEP", "ARR"]:
        pool = base.loc[base["phase"].eq(phase)]
        for runway_scope in [False, True]:
            keys = ["airport", "runway"] if runway_scope else ["airport"]
            prefix = ("runway" if runway_scope else "airport") + "_" + phase.lower()
            events = {k if isinstance(k, tuple) else (k,): np.sort(g["seconds"].to_numpy())
                      for k, g in pool.groupby(keys, dropna=False)}
            counts = {m: np.zeros(len(dep), dtype="float32") for m in TRAFFIC_WINDOWS}
            headway = np.full(len(dep), 3600.0, dtype="float32")
            for key, positions in query.groupby(keys, dropna=False).indices.items():
                pooled = events.get(key if isinstance(key, tuple) else (key,))
                if pooled is None or not len(pooled):
                    continue
                asked = times[positions]
                right = np.searchsorted(pooled, asked, side="left")
                for minutes, values in counts.items():
                    values[positions] = right - np.searchsorted(pooled, asked - minutes * 60, side="left")
                previous = np.maximum(0, right - 1)
                headway[positions] = np.where(right > 0, np.minimum(3600.0, asked - pooled[previous]), 3600.0)
            for minutes, values in counts.items():
                out[f"{prefix}_previous_{minutes}m"] = values
            out[f"{prefix}_headway_seconds"] = headway
    return out


def _normalized(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper().replace("", pd.NA).fillna("UNKNOWN").astype(str)


def _rolling_stats(times: np.ndarray, values: np.ndarray, asked: np.ndarray,
                   windows: list[int]) -> dict[str, np.ndarray]:
    right = np.searchsorted(times, asked, side="left")
    sums = np.r_[0.0, np.cumsum(values)]
    squares = np.r_[0.0, np.cumsum(values * values)]
    stats: dict[str, np.ndarray] = {}
    for minutes in windows:
        left = np.searchsorted(times, asked - minutes * 60, side="left")
        count = (right - left).astype(float)
        mean = np.divide(sums[right] - sums[left], count, out=np.full(len(count), np.nan), where=count > 0)
        second = np.divide(squares[right] - squares[left], count, out=np.full(len(count), np.nan), where=count > 0)
        stats[f"{minutes}m_count"] = count
        stats[f"{minutes}m_mean"] = mean
        stats[f"{minutes}m_std"] = np.sqrt(np.maximum(0.0, second - mean * mean))
    stats["_right"] = right.astype(float)
    return stats


def completed_arrival_features(dep: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    columns = ["ADES_mvt", "RUNWAY_mvt", "STAND_mvt", MOVEMENT, "BLOCK_TIME_UTC_mvt",
               TARGET, "AIRCRAFT_TYPE_mvt", "FLIGHT_mvt"]
    arr = context.loc[context[PHASE].eq("ARR"), columns].copy()
    landed = pd.to_datetime(arr[MOVEMENT], utc=True, errors="coerce")
    blocked = pd.to_datetime(arr["BLOCK_TIME_UTC_mvt"], utc=True, errors="coerce")
    taxi = pd.to_numeric(arr[TARGET], errors="coerce")
    keep = landed.notna() & blocked.notna() & blocked.ge(landed) & taxi.between(0, 7200)
    arr = arr.loc[keep]
    arr = arr.assign(
        completed=_epoch_seconds(blocked.loc[keep]),
        taxi_in=taxi.loc[keep].astype(float).to_numpy(),
        airport=_normalized(arr["ADES_mvt"]), runway=_normalized(arr["RUNWAY_mvt"]),
        stand=_normalized(arr["STAND_mvt"]), aircraft=_normalized(arr["AIRCRAFT_TYPE_mvt"]),
        carrier=_normalized(arr["FLIGHT_mvt"]).str.extract(r"^([A-Z]{2,4})", expand=False).fillna("UNKNOWN"),
    )
    asked = _epoch_seconds(dep[MOVEMENT])
    keys = pd.DataFrame({"airport": _normalized(dep["ADEP_mvt"]), "runway": _normalized(dep["RUNWAY_mvt"]),
                         "stand": _normalized(dep["STAND_mvt"])}, index=dep.index)
    out = pd.DataFrame(index=dep.index)

    for label, group in [("airport", ["airport"]), ("runway", ["airport", "runway"])]:
        names = [f"{m}m_{s}" for m in TRAFFIC_WINDOWS for s in ("count", "mean", "std")]
        result = {name: np.full(len(dep), np.nan) for name in names + ["last_age_seconds", "last_taxi_seconds"]}
        pools = {k if isinstance(k, tuple) else (k,): g.sort_values("completed", kind="stable")
                 for k, g in arr.groupby(group, dropna=False)}
        for key, positions in keys.groupby(group, dropna=False).indices.items():
            pool = pools.get(key if isinstance(key, tuple) else (key,))
            if pool is None or pool.empty:
                continue
            times, values, query = pool["completed"].to_numpy(), pool["taxi_in"].to_numpy(), asked[positions]
            stats = _rolling_stats(times, values, query, TRAFFIC_WINDOWS)
            for name in names:
                result[name][positions] = stats[name]
            latest = pool.groupby("completed")["taxi_in"].mean()
            stamps = latest.index.to_numpy()
            right = np.searchsorted(stamps, query, side="left")
            previous = np.maximum(right - 1, 0)
            age = query - stamps[previous]
            recent = (right > 0) & (age <= 7200)
            result["last_age_seconds"][positions[recent]] = age[recent]
            result["last_taxi_seconds"][positions[recent]] = latest.to_numpy()[previous[recent]]
        for name, values in result.items():
            out[f"completed_arrival_{label}_{name}"] = values.astype("float32")

    stand = {name: np.full(len(dep), np.nan)
             for name in ["last_age_seconds", "last_taxi_seconds", "aircraft_equal", "carrier_equal"]}
    pools = {k: g.sort_values("completed", kind="stable")
             for k, g in arr.groupby(["airport", "stand"], dropna=False) if k[1] != "UNKNOWN"}
    aircraft = _normalized(dep["AIRCRAFT_TYPE_mvt"]).to_numpy()
    carrier = _normalized(dep["FLIGHT_mvt"]).str.extract(r"^([A-Z]{2,4})", expand=False).fillna("UNKNOWN").to_numpy()
    for key, positions in keys.groupby(["airport", "stand"], dropna=False).indices.items():
        pool = pools.get(key if isinstance(key, tuple) else (key,))
        if pool is None or pool.empty:
            continue
        times = pool["completed"].to_numpy()
        right = np.searchsorted(times, asked[positions], side="left")
        previous = np.maximum(right - 1, 0)
        ambiguous = pool["completed"].duplicated(keep=False).to_numpy()
        age = asked[positions] - times[previous]
        recent = (right > 0) & (age <= 86400) & ~ambiguous[previous]
        target, source = positions[recent], previous[recent]
        stand["last_age_seconds"][target] = age[recent]
        stand["last_taxi_seconds"][target] = pool["taxi_in"].to_numpy()[source]
        stand["aircraft_equal"][target] = (aircraft[target] == pool["aircraft"].to_numpy()[source]).astype(float)
        stand["carrier_equal"][target] = (carrier[target] == pool["carrier"].to_numpy()[source]).astype(float)
    for name, values in stand.items():
        out[f"completed_arrival_stand_{name}"] = values.astype("float32")
    return out


NEIGHBOR_WINDOWS = [15, 60]
_NM_GROUPS = [("airport", ["airport", "period"]),
              ("runway", ["airport", "period", "runway"]),
              ("stand", ["airport", "period", "stand"])]


def _nm_proxy_pool(dep: pd.DataFrame, context: pd.DataFrame):
    columns = ["ADEP_mvt", "ADEP_flt", "RUNWAY_mvt", "STAND_mvt", MOVEMENT, "AOBT_3_flt"]
    pool = context.loc[context[PHASE].eq("DEP"), columns].copy()
    movement = pd.to_datetime(pool[MOVEMENT], utc=True, errors="coerce")
    offblock = pd.to_datetime(pool["AOBT_3_flt"], utc=True, errors="coerce")
    proxy = (movement - offblock).dt.total_seconds()
    keep = (movement.notna() & offblock.notna() & proxy.between(0, 7200)
            & _normalized(pool["ADEP_mvt"]).eq(_normalized(pool["ADEP_flt"])))
    pool = pool.loc[keep]
    movement = movement.loc[keep]
    pool = pool.assign(
        proxy=proxy.loc[keep].to_numpy(), seconds=_epoch_seconds(movement),
        period=movement.dt.strftime("%Y-%m"), airport=_normalized(pool["ADEP_mvt"]),
        runway=_normalized(pool["RUNWAY_mvt"]), stand=_normalized(pool["STAND_mvt"]),
    )
    clock = pd.to_datetime(dep[MOVEMENT], utc=True)
    asked = _epoch_seconds(clock)
    own = (clock - pd.to_datetime(dep["AOBT_3_flt"], utc=True, errors="coerce")).dt.total_seconds()
    own = own.where(own.between(0, 7200)
                    & _normalized(dep["ADEP_mvt"]).eq(_normalized(dep["ADEP_flt"]))).to_numpy()
    keys = pd.DataFrame({"airport": _normalized(dep["ADEP_mvt"]), "runway": _normalized(dep["RUNWAY_mvt"]),
                         "stand": _normalized(dep["STAND_mvt"]), "period": clock.dt.strftime("%Y-%m")},
                        index=dep.index)
    return pool, asked, own, keys


def completed_nm_neighbors(dep: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    pool, asked, own, keys = _nm_proxy_pool(dep, context)
    out = pd.DataFrame(index=dep.index)
    for label, group in _NM_GROUPS:
        scoped = pool if label == "airport" else pool.loc[pool[label].ne("UNKNOWN")]
        names = [f"{m}m_{s}" for m in NEIGHBOR_WINDOWS for s in ("count", "mean", "std", "own_minus_mean")]
        result = {name: np.full(len(dep), np.nan) for name in names + ["last_age_seconds", "last_proxy"]}
        pools = {k: g.sort_values(["seconds", "proxy"], kind="stable")
                 for k, g in scoped.groupby(group, dropna=False)}
        for key, positions in keys.groupby(group, dropna=False).indices.items():
            candidate = pools.get(key if isinstance(key, tuple) else (key,))
            if candidate is None or candidate.empty:
                continue
            times, values = candidate["seconds"].to_numpy(), candidate["proxy"].to_numpy()
            query = asked[positions]
            stats = _rolling_stats(times, values, query, NEIGHBOR_WINDOWS)
            for minutes in NEIGHBOR_WINDOWS:
                for statistic in ("count", "mean", "std"):
                    result[f"{minutes}m_{statistic}"][positions] = stats[f"{minutes}m_{statistic}"]
                result[f"{minutes}m_own_minus_mean"][positions] = own[positions] - stats[f"{minutes}m_mean"]
            right = stats["_right"].astype(int)
            last = np.maximum(0, right - 1)
            recent = (right > 0) & ((query - times[last]) <= 7200)
            sums = np.r_[0.0, np.cumsum(values)]
            tie_start = np.searchsorted(times, times[last], side="left")
            tie_end = np.searchsorted(times, times[last], side="right")
            tied_mean = (sums[tie_end] - sums[tie_start]) / (tie_end - tie_start)
            result["last_age_seconds"][positions[recent]] = query[recent] - times[last[recent]]
            result["last_proxy"][positions[recent]] = tied_mean[recent]
        for name, values in result.items():
            out[f"nm_neighbor_{label}_{name}"] = values.astype("float32")
    return out


def following_nm_neighbors(dep: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    pool, asked, own, keys = _nm_proxy_pool(dep, context)
    out = pd.DataFrame(index=dep.index)
    for label, group in _NM_GROUPS:
        scoped = pool if label == "airport" else pool.loc[pool[label].ne("UNKNOWN")]
        names = [f"{m}m_{s}" for m in NEIGHBOR_WINDOWS for s in ("count", "mean", "std", "own_minus_mean")]
        result = {name: np.full(len(dep), np.nan) for name in names + ["next_wait_seconds", "next_proxy"]}
        pools = {k: g.sort_values(["seconds", "proxy"], kind="stable")
                 for k, g in scoped.groupby(group, dropna=False)}
        for key, positions in keys.groupby(group, dropna=False).indices.items():
            candidate = pools.get(key if isinstance(key, tuple) else (key,))
            if candidate is None or candidate.empty:
                continue
            times, values = candidate["seconds"].to_numpy(), candidate["proxy"].to_numpy()
            query = asked[positions]
            sums = np.r_[0.0, np.cumsum(values)]
            squares = np.r_[0.0, np.cumsum(values * values)]
            left = np.searchsorted(times, query, side="right")
            for minutes in NEIGHBOR_WINDOWS:
                right = np.searchsorted(times, query + minutes * 60, side="right")
                count = (right - left).astype(float)
                mean = np.divide(sums[right] - sums[left], count,
                                 out=np.full(len(count), np.nan), where=count > 0)
                second = np.divide(squares[right] - squares[left], count,
                                   out=np.full(len(count), np.nan), where=count > 0)
                result[f"{minutes}m_count"][positions] = count
                result[f"{minutes}m_mean"][positions] = mean
                result[f"{minutes}m_std"][positions] = np.sqrt(np.maximum(0.0, second - mean * mean))
                result[f"{minutes}m_own_minus_mean"][positions] = own[positions] - mean
            following = np.minimum(left, len(times) - 1)
            available = (left < len(times)) & ((times[following] - query) <= 7200)
            start = np.searchsorted(times, times[following], side="left")
            end = np.searchsorted(times, times[following], side="right")
            tied_mean = (sums[end] - sums[start]) / (end - start)
            result["next_wait_seconds"][positions[available]] = times[following[available]] - query[available]
            result["next_proxy"][positions[available]] = tied_mean[available]
        for name, values in result.items():
            out[f"nm_following_{label}_{name}"] = values.astype("float32")
    return out


def carrier_features(x: pd.DataFrame, dep: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["movement_carrier"] = _normalized(dep["FLIGHT_mvt"]).str.extract(
        r"^([A-Z]{2,4})", expand=False).fillna("UNKNOWN")
    x["airport_movement_carrier"] = x["ADEP_mvt"] + ":" + x["movement_carrier"]
    schedule = pd.to_datetime(dep[TIME], utc=True)
    movement = pd.to_datetime(dep[MOVEMENT], utc=True)
    x["scheduled_day"] = schedule.dt.day.astype(float)
    x["movement_second"] = movement.dt.second.astype(float)
    x["schedule_second"] = schedule.dt.second.astype(float)
    return x


def service_features(x: pd.DataFrame, dep: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["movement_service"] = _normalized(dep["FLIGHT_mvt"])
    x["nm_service"] = _normalized(dep["CALLSIGN_flt"])
    x["airport_movement_service"] = x["ADEP_mvt"] + ":" + x["movement_service"]
    x["airport_nm_service"] = x["ADEP_mvt"] + ":" + x["nm_service"]
    return x


def duration_features(x: pd.DataFrame, dep: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    for label, left, right in [
        ("callsign", "FLIGHT_mvt", "CALLSIGN_flt"),
        ("aircraft", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"),
        ("origin", "ADEP_mvt", "ADEP_flt"),
        ("destination", "ADES_mvt", "ADES_flt"),
        ("rule", "FLIGHT_RULE_mvt", "FLIGHT_RULE_flt"),
    ]:
        a = dep[left].astype("string").str.strip().str.upper()
        b = dep[right].astype("string").str.strip().str.upper()
        x[f"{label}_both_present"] = (a.notna() & b.notna()).astype(float)
        x[f"{label}_equal"] = a.eq(b).fillna(False).astype(float)
    x["airport_route"] = x["ADEP_mvt"] + ":" + x["ADES_mvt"]
    x["airport_aircraft"] = x["ADEP_mvt"] + ":" + x["AIRCRAFT_TYPE_mvt"]
    x["airport_carrier_runway"] = x["airport_movement_carrier"] + ":" + x["RUNWAY_mvt"]
    x["nm_planned_flight_duration"] = x["movement_minus_EOBT_1_flt"] - x["movement_minus_ARVT_1_flt"]
    x["nm_actual_flight_duration"] = x["movement_minus_AOBT_3_flt"] - x["movement_minus_ARVT_3_flt"]
    x["nm_flight_duration_change"] = x["nm_actual_flight_duration"] - x["nm_planned_flight_duration"]
    x["nm_duration_ratio"] = x["nm_actual_flight_duration"] / x["nm_planned_flight_duration"].clip(lower=120)
    x["nm_actual_duration_plausible"] = x["nm_actual_flight_duration"].between(120, 100000).astype(float)
    x["nm_planned_duration_plausible"] = x["nm_planned_flight_duration"].between(120, 100000).astype(float)
    x["nm_actual_offblock_schedule_delay"] = x["movement_schedule_delta_sec"] - x["movement_minus_AOBT_3_flt"]
    x["nm_estimated_offblock_schedule_delay"] = x["movement_schedule_delta_sec"] - x["movement_minus_EOBT_1_flt"]
    x["nm_initial_offblock_schedule_delay"] = x["movement_schedule_delta_sec"] - x["movement_minus_IOBT_flt"]
    x["schedule_delay_days"] = np.floor(x["movement_schedule_delta_sec"] / 86400)
    return x


CARRIER_CATEGORIES = ["movement_carrier", "airport_movement_carrier"]
SERVICE_CATEGORIES = ["movement_service", "nm_service",
                      "airport_movement_service", "airport_nm_service"]
DURATION_CATEGORIES = ["airport_route", "airport_aircraft", "airport_carrier_runway"]

STAGES = {
    "base": (),
    "carrier": ("carrier",),
    "duration": ("carrier", "duration"),
    "service": ("carrier", "duration", "service"),
    "arrival": ("carrier", "duration", "service", "arrival"),
    "neighbor": ("carrier", "duration", "service", "arrival", "neighbor"),
    "following": ("carrier", "duration", "service", "arrival", "neighbor", "following"),
}

def build_matrix(raw: pd.DataFrame, extras: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    dep = departures(raw)
    clock = pd.to_datetime(dep[TIME], utc=True)
    movement = pd.to_datetime(dep[MOVEMENT], utc=True)
    x = pd.DataFrame(index=dep.index)

    for column in BASE_CATEGORIES:
        x[column] = dep[column].fillna("UNKNOWN").astype(str) if column in dep else "UNKNOWN"
    x["hour_utc"] = clock.dt.hour.astype(float)
    x["weekday_utc"] = clock.dt.dayofweek.astype(float)
    x["month_utc"] = clock.dt.month.astype(float)
    x["hour_sin"] = np.sin(2 * np.pi * clock.dt.hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * clock.dt.hour / 24)
    schedule_bin = pd.DataFrame({"airport": dep["ADEP_mvt"], "bin": clock.dt.floor("30min")}, index=dep.index)
    x["scheduled_departures_30m"] = schedule_bin.groupby(["airport", "bin"], dropna=False)["airport"].transform("size").astype(float)

    for column in ["RUNWAY_mvt", "STAND_mvt"]:
        x[column] = dep[column].fillna("UNKNOWN").astype(str)
        x["airport_" + column] = x["ADEP_mvt"] + ":" + x[column]
    x["movement_hour_utc"] = movement.dt.hour.astype(float)
    x["movement_weekday_utc"] = movement.dt.dayofweek.astype(float)
    x["movement_month_utc"] = movement.dt.month.astype(float)
    x["movement_schedule_delta_sec"] = (movement - clock).dt.total_seconds().clip(-86400, 172800)
    movement_bin = pd.DataFrame({"airport": dep["ADEP_mvt"], "bin": movement.dt.floor("30min")}, index=dep.index)
    x["movement_departures_30m"] = movement_bin.groupby(["airport", "bin"], dropna=False)["airport"].transform("size").astype(float)

    x = pd.concat([x, movement_context(dep, raw)], axis=1)
    x["scheduled_minute_of_day"] = clock.dt.hour * 60 + clock.dt.minute
    x["movement_minute_of_day"] = movement.dt.hour * 60 + movement.dt.minute
    x["movement_day_of_year"] = movement.dt.dayofyear
    x["airport_operator"] = x["ADEP_mvt"] + ":" + x["AIRCRAFT_OPERATOR_flt"]
    x["stand_runway"] = x["airport_STAND_mvt"] + ":" + x["RUNWAY_mvt"]

    for column in PLANNED_CLOCKS:
        planned = pd.to_datetime(dep[column], utc=True, errors="coerce")
        x[f"movement_minus_{column}"] = (movement - planned).dt.total_seconds().clip(-604800, 604800)
        x[f"schedule_minus_{column}"] = (clock - planned).dt.total_seconds().clip(-604800, 604800)
        x[f"missing_{column}"] = planned.isna().astype(int)
    for column in ["ADEP_flt", "ADES_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt"]:
        x[column] = dep[column].fillna("UNKNOWN").astype(str) if column in dep else "UNKNOWN"
    x["nm_departure_matches"] = (x["ADEP_flt"] == x["ADEP_mvt"]).astype(int)
    x["nm_destination_matches"] = (x["ADES_flt"] == x["ADES_mvt"]).astype(int)

    for column in OBSERVED_CLOCKS:
        observed = pd.to_datetime(dep[column], utc=True, errors="coerce")
        x[f"movement_minus_{column}"] = (movement - observed).dt.total_seconds().clip(-604800, 604800)
        x[f"missing_{column}"] = observed.isna().astype(int)
    x["nm_actual_vs_estimated_seconds"] = x["movement_minus_EOBT_1_flt"] - x["movement_minus_AOBT_3_flt"]
    x["nm_last_vs_actual_seconds"] = x["movement_minus_AOBT_3_flt"] - x["movement_minus_LOBT_flt"]

    if "carrier" in extras:
        x = carrier_features(x, dep)
    if "duration" in extras:
        x = duration_features(x, dep)
    if "service" in extras:
        x = service_features(x, dep)
    if "arrival" in extras:
        x = pd.concat([x, completed_arrival_features(dep, raw)], axis=1)
    if "neighbor" in extras:
        x = pd.concat([x, completed_nm_neighbors(dep, raw)], axis=1)
    if "following" in extras:
        x = pd.concat([x, following_nm_neighbors(dep, raw)], axis=1)
    return dep, x


def _override(current: pd.Series, x: pd.DataFrame, column: str) -> pd.Series:
    delta = x[f"movement_minus_{column}"]
    valid = delta.notna() & delta.between(VALID_LOW, VALID_HIGH) & x["nm_departure_matches"].eq(1)
    return delta.where(valid, current).clip(lower=0).astype(float)


def baseline(x: pd.DataFrame) -> pd.Series:
    result = _override(pd.Series(DEFAULT_BASELINE_SECONDS, index=x.index, dtype=float), x, "EOBT_1_flt")
    for column in ["LOBT_flt", "AOBT_3_flt"]:
        result = _override(result, x, column)
    unmatched = lirf_scope(x)
    return x["movement_schedule_delta_sec"].clip(0, 172800).where(unmatched, result).astype(float)


def lirf_scope(x: pd.DataFrame) -> pd.Series:
    return x["missing_IOBT_flt"].eq(1) & x["ADEP_mvt"].eq("LIRF")
