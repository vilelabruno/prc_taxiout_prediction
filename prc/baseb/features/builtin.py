
from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from prc.baseb.config import TARGET
from prc.baseb.features.registry import FeatureFamily, register


def _sec(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).dt.total_seconds()


@register
class Clocks(FeatureFamily):

    name = "clocks"

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        mvt = df["MVT_TIME_UTC_mvt"]
        out = pd.DataFrame(index=df.index)
        out["mvt_aobt"] = _sec(mvt, df["AOBT_3_flt"])
        out["mvt_eobt"] = _sec(mvt, df["EOBT_1_flt"])
        out["mvt_sched"] = _sec(mvt, df["SCHED_TIME_UTC_mvt"])
        out["mvt_lobt"] = _sec(mvt, df["LOBT_flt"])
        out["mvt_iobt"] = _sec(mvt, df["IOBT_flt"])
        out["aobt_eobt"] = _sec(df["AOBT_3_flt"], df["EOBT_1_flt"])
        out["aobt_lobt"] = _sec(df["AOBT_3_flt"], df["LOBT_flt"])
        out["aobt_sched"] = _sec(df["AOBT_3_flt"], df["SCHED_TIME_UTC_mvt"])
        out["is_matched"] = df["AOBT_3_flt"].notna().astype("int8")
        out["mvt_sec_of_min"] = mvt.dt.second.astype("float32")
        return out


@register
class Calendar(FeatureFamily):
    name = "calendar"

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        t = df["MVT_TIME_UTC_mvt"]
        out = pd.DataFrame(index=df.index)
        out["hour"] = t.dt.hour.astype("int8")
        out["minute_of_day"] = (t.dt.hour * 60 + t.dt.minute).astype("int16")
        out["dow"] = t.dt.dayofweek.astype("int8")
        out["month"] = t.dt.month.astype("int8")
        out["day_of_year"] = t.dt.dayofyear.astype("int16")
        return out


@register
class Categoricals(FeatureFamily):

    name = "categoricals"
    cols = (
        "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt",
        "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt",
        "FLIGHT_RULE_flt",
    )

    def _fit(self, train: pd.DataFrame) -> None:
        self.levels = {
            c: pd.Index(train[c].dropna().astype(str).unique()) for c in self.cols if c in train
        }

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c, levels in self.levels.items():
            values = df[c].astype(str).where(df[c].notna(), None)
            out[f"cat_{c}"] = levels.get_indexer(values).astype("int32")
        return out


@register
class Geometry(FeatureFamily):

    name = "geometry"

    def __init__(self, min_count: int = 10, **params):
        super().__init__(min_count=min_count, **params)
        self.min_count = min_count

    @staticmethod
    def _keys(df: pd.DataFrame):
        apt = df["ADEP_mvt"].astype("str")
        stand = df["STAND_mvt"].astype("str")
        runway = df["RUNWAY_mvt"].astype("str")
        return (
            apt.str.cat([stand, runway], sep="|"),
            apt.str.cat(runway, sep="|"),
            apt,
        )

    def _fit(self, train: pd.DataFrame) -> None:
        dep = train[(train["PHASE_mvt"] == "DEP") & train[TARGET].notna()]
        k3, k2, k1 = self._keys(dep)
        y = dep[TARGET]
        self.tables = []
        for key in (k3, k2, k1):
            g = y.groupby(key)
            stats = pd.DataFrame(
                {"median": g.median(), "p10": g.quantile(0.10), "n": g.size()}
            )
            self.tables.append(stats[stats["n"] >= self.min_count])
        self.global_median = float(y.median())
        self.global_p10 = float(y.quantile(0.10))

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = self._keys(df)
        out = pd.DataFrame(index=df.index)
        med = pd.Series(np.nan, index=df.index)
        p10 = pd.Series(np.nan, index=df.index)
        level = pd.Series(0, index=df.index, dtype="int8")
        for depth, (key, table) in enumerate(zip(keys, self.tables, strict=True), start=1):
            m = key.map(table["median"])
            fill = med.isna() & m.notna()
            med[fill] = m[fill]
            p10[fill] = key.map(table["p10"])[fill]
            level[fill] = 4 - depth
        out["geo_median"] = med.fillna(self.global_median)
        out["geo_p10"] = p10.fillna(self.global_p10)
        out["geo_level"] = level
        return out


def _local_airport(df: pd.DataFrame) -> pd.Series:
    is_dep = (df["PHASE_mvt"] == "DEP").to_numpy()
    return pd.Series(
        np.where(is_dep, df["ADEP_mvt"].to_numpy(), df["ADES_mvt"].to_numpy()),
        index=df.index,
    ).astype("str")


def _window_count(
    q_key: pd.Series, q_t: np.ndarray, ev_key: pd.Series, ev_t: np.ndarray, delta: pd.Timedelta
) -> np.ndarray:
    return _window_agg(q_key, q_t, ev_key, ev_t, None, delta)[0]


def _window_agg(q_key, q_t, ev_key, ev_t, ev_v, delta: pd.Timedelta, strictly_before: bool = False):
    n = len(q_t)
    count = np.zeros(n, dtype="int32")
    mean = np.full(n, np.nan) if ev_v is not None else None
    delta_ns = np.int64(delta.value)
    side_hi = "left" if strictly_before else "right"
    side_lo = "left" if strictly_before else "right"

    q_key = pd.Series(np.asarray(q_key), copy=False)
    ev_key = pd.Series(np.asarray(ev_key), copy=False)
    codes, _ = pd.factorize(pd.concat([q_key, ev_key], ignore_index=True), use_na_sentinel=True)
    qc, ec = codes[:n], codes[n:]

    by_key: dict[int, np.ndarray] = {}
    keep = np.flatnonzero(ec >= 0)
    if len(keep):
        order = keep[np.lexsort((ev_t[keep], ec[keep]))]
        for start, stop in _group_slices(ec[order]):
            by_key[int(ec[order[start]])] = order[start:stop]

    for k, grp in pd.Series(np.arange(n)).groupby(qc, sort=False).indices.items():
        rows = by_key.get(int(k))
        if rows is None:
            continue
        times = ev_t[rows]
        q = q_t[grp]
        hi = np.searchsorted(times, q, side=side_hi)
        lo = np.searchsorted(times, q - delta_ns, side=side_lo)
        count[grp] = (hi - lo).astype("int32")
        if ev_v is not None:
            csum = np.concatenate(([0.0], np.nancumsum(ev_v[rows])))
            cnt = np.concatenate(([0], np.cumsum(np.isfinite(ev_v[rows]))))
            num = csum[hi] - csum[lo]
            den = (cnt[hi] - cnt[lo]).astype("float64")
            mean[grp] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return count, mean


@register
class Traffic(FeatureFamily):

    name = "traffic"
    needs_context = True

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        ctx = self.context if self.context is not None else df
        q_t = _ns(df["MVT_TIME_UTC_mvt"])
        q_apt = df["ADEP_mvt"].astype("str")
        q_rwy = q_apt.str.cat(df["RUNWAY_mvt"].astype("str"), sep="|")

        ev_t = _ns(ctx["MVT_TIME_UTC_mvt"])
        ev_apt = _local_airport(ctx)
        is_dep = (ctx["PHASE_mvt"] == "DEP").to_numpy()
        ev_rwy = ev_apt.str.cat(ctx["RUNWAY_mvt"].astype("str"), sep="|")

        out["dep_15m"] = _window_count(
            q_apt, q_t, ev_apt[is_dep], ev_t[is_dep], pd.Timedelta("15min")
        )
        out["rwy_dep_10m"] = _window_count(
            q_rwy, q_t, ev_rwy[is_dep], ev_t[is_dep], pd.Timedelta("10min")
        )

        mvt_aobt = _sec(ctx["MVT_TIME_UTC_mvt"], ctx["AOBT_3_flt"]).to_numpy(dtype="float64")
        _, regime = _window_agg(
            q_apt,
            q_t,
            ev_apt[is_dep],
            ev_t[is_dep],
            mvt_aobt[is_dep],
            pd.Timedelta("30min"),
            strictly_before=True,
        )
        out["regime_30m"] = regime
        return out


@register
class ArrivalPressure(FeatureFamily):

    name = "arrivals"
    needs_context = True

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ctx = self.context if self.context is not None else df
        out = pd.DataFrame(index=df.index)
        q_t = _ns(df["MVT_TIME_UTC_mvt"])
        q_apt = df["ADEP_mvt"].astype("str")

        apt = _local_airport(ctx)
        is_arr = (ctx["PHASE_mvt"] == "ARR").to_numpy()
        ldg_t = _ns(ctx["MVT_TIME_UTC_mvt"])
        ib_t = _ns(ctx["BLOCK_TIME_UTC_mvt"])
        has_ib = ctx["BLOCK_TIME_UTC_mvt"].notna().to_numpy()

        arr_apt, arr_t = apt[is_arr], ldg_t[is_arr]
        for label, delta in (("15m", "15min"), ("30m", "30min"), ("60m", "60min")):
            out[f"arr_{label}"] = _window_count(
                q_apt, q_t, arr_apt, arr_t, pd.Timedelta(delta)
            ).astype("int16")

        window = np.int64(pd.Timedelta("15min").value)
        out["arr_next_15m"] = _window_count(
            q_apt, q_t + window, arr_apt, arr_t, pd.Timedelta("15min")
        ).astype("int16")

        ib_ok = is_arr & has_ib
        big = pd.Timedelta("36500D")
        landed = _window_count(q_apt, q_t, apt[ib_ok], ldg_t[ib_ok], big)
        blocked = _window_count(q_apt, q_t, apt[ib_ok], ib_t[ib_ok], big)
        out["arr_taxiing"] = (landed - blocked).astype("int16")
        out["arr_ib_15m"] = _window_count(
            q_apt, q_t, apt[ib_ok], ib_t[ib_ok], pd.Timedelta("15min")
        ).astype("int16")

        is_dep = (ctx["PHASE_mvt"] == "DEP").to_numpy()
        dep_30m = _window_count(q_apt, q_t, apt[is_dep], ldg_t[is_dep], pd.Timedelta("30min"))
        total = (out["arr_30m"].to_numpy(dtype="float64") + dep_30m).astype("float64")
        out["arr_dep_ratio"] = np.where(
            total > 0, out["arr_30m"].to_numpy(dtype="float64") / np.where(total > 0, total, 1.0),
            np.nan,
        )
        return out


def _centred_stats(
    q_key: pd.Series,
    q_t: np.ndarray,
    ev_key: pd.Series,
    ev_t: np.ndarray,
    cols: dict[str, tuple[np.ndarray, str]],
    half: pd.Timedelta,
) -> dict[str, np.ndarray]:
    nq = len(q_t)
    key = pd.concat(
        [pd.Series(np.asarray(q_key)), pd.Series(np.asarray(ev_key))], ignore_index=True
    )
    t = np.concatenate([np.asarray(q_t), np.asarray(ev_t)])
    codes = pd.factorize(key, use_na_sentinel=True)[0]
    out = {name: np.full(nq, np.nan) for name in cols}
    usable = np.flatnonzero((codes >= 0) & (t != np.iinfo("int64").min))
    if not len(usable):
        return out
    order = usable[np.lexsort((t[usable], codes[usable]))]
    padded = {
        name: np.concatenate([np.full(nq, np.nan), np.asarray(v, dtype="float64")])
        for name, (v, _) in cols.items()
    }
    window = pd.Timedelta(half) * 2
    ordered_t = t[order]
    for start, stop in _group_slices(codes[order]):
        rows = order[start:stop]
        take = np.flatnonzero(rows < nq)
        if not len(take):
            continue
        index = pd.DatetimeIndex(ordered_t[start:stop].astype("datetime64[ns]"))
        for name, (_, stat) in cols.items():
            roll = pd.Series(padded[name][rows], index=index).rolling(
                window, center=True, min_periods=1
            )
            out[name][rows[take]] = getattr(roll, stat)().to_numpy(dtype="float64")[take]
    return out


@register
class LocalRegime(FeatureFamily):

    name = "local_regime"
    needs_context = True

    HALF_LONG = pd.Timedelta("30min")
    HALF_SHORT = pd.Timedelta("10min")

    BLOCKS = ("dep", "arr", "dev")
    COLUMN_ORDER = (
        "reg_apt_med30", "reg_apt_n30", "reg_apt_all30", "reg_apt_share30", "reg_apt_med10",
        "reg_apt_bwd30", "reg_apt_fwd30", "reg_apt_trend", "reg_rwy_med30", "reg_rwy_n30",
        "reg_arr_taxiin_med30", "reg_arr_taxiin_n30", "reg_apt_dev",
    )

    def __init__(self, blocks: Iterable[str] = BLOCKS, **params):
        super().__init__(blocks=list(blocks), **params)
        unknown = set(blocks) - set(self.BLOCKS)
        if unknown:
            raise ValueError(f"unknown local_regime block(s) {sorted(unknown)}")
        self.blocks = tuple(blocks)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ctx = self.context if self.context is not None else df
        out = pd.DataFrame(index=df.index)

        q_t = _ns(df["MVT_TIME_UTC_mvt"])
        q_apt = df["ADEP_mvt"].astype("str")
        q_rwy = q_apt.str.cat(df["RUNWAY_mvt"].astype("str"), sep="|")

        apt = _local_airport(ctx)
        ev_t = _ns(ctx["MVT_TIME_UTC_mvt"])
        is_dep = (ctx["PHASE_mvt"] == "DEP").to_numpy()
        is_arr = (ctx["PHASE_mvt"] == "ARR").to_numpy()

        dep_apt, dep_t = apt[is_dep], ev_t[is_dep]
        base = _sec(ctx["MVT_TIME_UTC_mvt"], ctx["AOBT_3_flt"]).to_numpy(dtype="float64")[is_dep]

        med30 = None
        if {"dep", "dev"} & set(self.blocks):
            med30 = _centred_stats(
                q_apt, q_t, dep_apt, dep_t, {"m": (base, "median")}, self.HALF_LONG
            )["m"]

        if "dep" in self.blocks:
            dep_rwy = apt.str.cat(ctx["RUNWAY_mvt"].astype("str"), sep="|")[is_dep]
            ones = np.ones(len(base))
            out["reg_apt_med30"] = med30
            counts = _centred_stats(
                q_apt, q_t, dep_apt, dep_t,
                {"reg_apt_n30": (base, "count"), "reg_apt_all30": (ones, "count")},
                self.HALF_LONG,
            )
            for name, values in counts.items():
                out[name] = values
            total = out["reg_apt_all30"].to_numpy()
            out["reg_apt_share30"] = np.where(
                total > 0, out["reg_apt_n30"].to_numpy() / np.where(total > 0, total, 1.0), np.nan
            )
            out["reg_apt_med10"] = _centred_stats(
                q_apt, q_t, dep_apt, dep_t, {"m": (base, "median")}, self.HALF_SHORT
            )["m"]

            window = np.int64(self.HALF_LONG.value)
            _, bwd = _window_agg(
                q_apt, q_t, dep_apt, dep_t, base, self.HALF_LONG, strictly_before=True
            )
            _, fwd = _window_agg(q_apt, q_t + window, dep_apt, dep_t, base, self.HALF_LONG)
            out["reg_apt_bwd30"] = bwd
            out["reg_apt_fwd30"] = fwd
            out["reg_apt_trend"] = fwd - bwd

            rwy30 = _centred_stats(
                q_rwy, q_t, dep_rwy, dep_t,
                {"reg_rwy_med30": (base, "median"), "reg_rwy_n30": (base, "count")},
                self.HALF_LONG,
            )
            for name, values in rwy30.items():
                out[name] = values

        if "arr" in self.blocks:
            taxiin = ctx[TARGET].to_numpy(dtype="float64")[is_arr]
            arr30 = _centred_stats(
                q_apt, q_t, apt[is_arr], ev_t[is_arr],
                {"reg_arr_taxiin_med30": (taxiin, "median"),
                 "reg_arr_taxiin_n30": (taxiin, "count")},
                self.HALF_LONG,
            )
            for name, values in arr30.items():
                out[name] = values

        if "dev" in self.blocks:
            out["reg_apt_dev"] = (
                _sec(df["MVT_TIME_UTC_mvt"], df["AOBT_3_flt"]).to_numpy(dtype="float64") - med30
            )
        return out[[c for c in self.COLUMN_ORDER if c in out.columns]]


@register
class Sequence(FeatureFamily):

    name = "sequence"

    WINDOW = pd.Timedelta("10min")

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        t = _ns(df["MVT_TIME_UTC_mvt"])
        apt = df["ADEP_mvt"].astype("str")
        rwy = apt.str.cat(df["RUNWAY_mvt"].astype("str"), sep="|")
        window = np.int64(self.WINDOW.value)

        out = pd.DataFrame(index=df.index)
        for prefix, key in (("rwy", rwy), ("apt", apt)):
            prev_gap, next_gap, gap_mean5, pos, after = _sequence_stats(key, t, window)
            out[f"seq_{prefix}_prev_gap"] = prev_gap
            out[f"seq_{prefix}_next_gap"] = next_gap
            out[f"seq_{prefix}_pos_10m"] = pos
            if prefix == "rwy":
                out["seq_rwy_gap_mean5"] = gap_mean5
                out["seq_rwy_burst_10m"] = (pos + after + 1).astype("int16")

        aobt = _ns(df["AOBT_3_flt"])
        valid_aobt = df["AOBT_3_flt"].notna().to_numpy()
        out["seq_apt_taxiing"] = _taxiing_now(apt, t, aobt, valid_aobt)
        out["seq_rwy_taxiing"] = _taxiing_now(rwy, t, aobt, valid_aobt)
        return out


@register
class QueueRate(FeatureFamily):

    name = "queue_rate"

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        off = _ns(df["MVT_TIME_UTC_mvt"]).astype("float64")
        has = df["AOBT_3_flt"].notna().to_numpy()
        out_t = np.where(has, _ns(df["AOBT_3_flt"]).astype("float64"), np.nan)
        apt = df["ADEP_mvt"].astype("str").to_numpy()
        q = np.full(len(df), np.nan)
        for airport in pd.unique(apt):
            here = np.flatnonzero(apt == airport)
            offs = np.sort(off[here])
            ok = here[has[here]]
            n = (np.searchsorted(offs, off[ok], side="right")
                 - np.searchsorted(offs, out_t[ok], side="right") - 1)
            q[ok] = np.maximum(n, 0)
        window = np.maximum((off - out_t) / 1e9, 60.0)
        return pd.DataFrame({"qrate_10m": (q / window * 600).astype("float32")}, index=df.index)


@register
class OperatorClockSpread(FeatureFamily):

    name = "op_clock_spread"

    GAPS = {"eobt": "EOBT_1_flt", "lobt": "LOBT_flt"}

    def __init__(self, min_rows: int = 200, **params):
        super().__init__(min_rows=min_rows, **params)
        self.min_rows = min_rows

    def _gap(self, df: pd.DataFrame, col: str) -> pd.Series:
        return _sec(df["AOBT_3_flt"], df[col])

    def _fit(self, train: pd.DataFrame) -> None:
        dep = train[train["PHASE_mvt"] == "DEP"]
        op = dep["AIRCRAFT_OPERATOR_flt"].astype("str")
        self.stats: dict[str, tuple[pd.DataFrame, float, float]] = {}
        for name, col in self.GAPS.items():
            g = pd.Series(np.asarray(self._gap(dep, col), dtype=float), index=dep.index)
            ok = g.notna() & dep["AIRCRAFT_OPERATOR_flt"].notna()
            grp = g[ok].groupby(op[ok])
            st = pd.DataFrame({"med": grp.median(), "iqr": grp.quantile(0.75) - grp.quantile(0.25),
                               "n": grp.size()})
            st = st[st["n"] >= self.min_rows]
            pooled_med = float(g[ok].median()) if ok.any() else 0.0
            pooled_iqr = float(g[ok].quantile(0.75) - g[ok].quantile(0.25)) if ok.any() else 1.0
            self.stats[name] = (st, pooled_med, pooled_iqr)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        op = df["AIRCRAFT_OPERATOR_flt"].astype("str").where(df["AIRCRAFT_OPERATOR_flt"].notna())
        for name, col in self.GAPS.items():
            st, pooled_med, pooled_iqr = self.stats[name]
            iqr = op.map(st["iqr"]).fillna(pooled_iqr).to_numpy(dtype=float)
            med = op.map(st["med"]).fillna(pooled_med).to_numpy(dtype=float)
            g = np.asarray(self._gap(df, col), dtype=float)
            out[f"ocs_{name}_iqr"] = iqr.astype("float32")
            out[f"ocs_{name}_z"] = ((g - med) / np.maximum(iqr, 1.0)).astype("float32")
        return out


@register
class StandTurnaround(FeatureFamily):

    name = "stand_turnaround"
    needs_context = True

    def __init__(self, min_count: int = 20, **params):
        super().__init__(min_count=min_count, **params)
        self.min_count = min_count

    @staticmethod
    def _local_airport(df: pd.DataFrame) -> pd.Series:
        is_dep = (df["PHASE_mvt"] == "DEP").to_numpy()
        return pd.Series(
            np.where(is_dep, df["ADEP_mvt"].to_numpy(), df["ADES_mvt"].to_numpy()),
            index=df.index,
        ).astype("str")

    @classmethod
    def _stand_key(cls, df: pd.DataFrame) -> pd.Series:
        return cls._local_airport(df).str.cat(df["STAND_mvt"].astype("str"), sep="|")

    @classmethod
    def _arrivals(cls, ctx: pd.DataFrame) -> pd.DataFrame:
        arr = ctx[(ctx["PHASE_mvt"] == "ARR") & ctx["BLOCK_TIME_UTC_mvt"].notna()]
        ev = pd.DataFrame(
            {
                "key": cls._stand_key(arr),
                "ib": arr["BLOCK_TIME_UTC_mvt"],
                "ldg": arr["MVT_TIME_UTC_mvt"],
                "taxiin": arr[TARGET].astype("float64"),
                "arr_type": arr["AIRCRAFT_TYPE_mvt"].astype("str"),
                "sched": (
                    arr["SCHED_TIME_UTC_mvt"]
                    if "SCHED_TIME_UTC_mvt" in arr
                    else pd.Series(
                        pd.NaT, index=arr.index, dtype=arr["BLOCK_TIME_UTC_mvt"].dtype
                    )
                ),
            }
        ).reset_index(drop=True)
        ev = ev[ev["key"].notna()].sort_values("ib", kind="stable")
        ev["ib_prev"] = ev.groupby("key", sort=False, observed=True)["ib"].shift(1)
        return ev

    @staticmethod
    def _query(df: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
        q = pd.DataFrame(
            {
                "_ix": np.arange(len(df)),
                "key": key.fillna("\x00"),
                "t": df["MVT_TIME_UTC_mvt"],
            }
        ).reset_index(drop=True)
        return q.sort_values("t", kind="stable")

    def _join(self, df: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
        q = self._query(df, self._stand_key(df))
        stamps = ("ib", "ldg", "ib_prev", "ib_next", "sched")
        if ev.empty:
            out = pd.DataFrame({c: pd.Series(pd.NaT, index=q.index, dtype=q["t"].dtype)
                                for c in stamps})
            out["taxiin"] = np.nan
            out["arr_type"] = pd.Series(pd.NA, index=q.index, dtype="str")
            return out
        prev = pd.merge_asof(q, ev, left_on="t", right_on="ib", by="key", direction="backward")
        nxt = pd.merge_asof(
            q,
            ev[["key", "ib"]].rename(columns={"ib": "ib_next"}),
            left_on="t",
            right_on="ib_next",
            by="key",
            direction="forward",
            allow_exact_matches=False,
        )
        prev["ib_next"] = nxt["ib_next"].to_numpy()
        return prev.sort_values("_ix", kind="stable").reset_index(drop=True)

    def _fit(self, train: pd.DataFrame) -> None:
        dep = train[(train["PHASE_mvt"] == "DEP") & train[TARGET].notna()]
        joined = self._join(dep, self._arrivals(self.context))
        turn = _sec(dep["BLOCK_TIME_UTC_mvt"].reset_index(drop=True), joined["ib"]).to_numpy()
        frame = pd.DataFrame(
            {
                "apt": self._local_airport(dep).to_numpy(),
                "type": dep["AIRCRAFT_TYPE_mvt"].astype("str").to_numpy(),
                "turn": turn,
            }
        ).dropna(subset=["turn"])
        self.prior_tables = []
        for keys in (["apt", "type"], ["apt"]):
            g = frame.groupby(keys, observed=True)["turn"]
            stats = pd.DataFrame({"median": g.median(), "n": g.size()})
            self.prior_tables.append(stats[stats["n"] >= self.min_count]["median"])
        self.global_prior = float(frame["turn"].median()) if len(frame) else 0.0

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        joined = self._join(df, self._arrivals(self.context))
        t = df["MVT_TIME_UTC_mvt"].reset_index(drop=True)
        out = pd.DataFrame(index=df.index)
        out["turn_prev_ib_gap"] = _sec(t, joined["ib"]).to_numpy()
        out["turn_prev_ldg_gap"] = _sec(t, joined["ldg"]).to_numpy()
        out["turn_prev2_ib_gap"] = _sec(t, joined["ib_prev"]).to_numpy()
        out["turn_next_ib_gap"] = _sec(joined["ib_next"], t).to_numpy()
        out["turn_prev_taxiin"] = joined["taxiin"].to_numpy(dtype="float64")

        dep_type = df["AIRCRAFT_TYPE_mvt"].astype("str").to_numpy()
        arr_type = joined["arr_type"].to_numpy()
        same = np.where(pd.isna(arr_type), -1, (arr_type == dep_type).astype("int8"))
        out["turn_prev_same_type"] = same.astype("int8")

        out["turn_prior"] = self._prior(df)
        out["turn_est_taxi"] = out["turn_prev_ib_gap"] - out["turn_prior"]
        return out

    def _prior(self, df: pd.DataFrame) -> pd.Series:
        apt = self._local_airport(df)
        dep_type = df["AIRCRAFT_TYPE_mvt"].astype("str").to_numpy()
        pair = pd.MultiIndex.from_arrays([apt.to_numpy(), dep_type])
        prior = pd.Series(pair.map(self.prior_tables[0]), index=df.index, dtype="float64")
        fallback = pd.Series(apt.map(self.prior_tables[1]).to_numpy(), index=df.index)
        return prior.fillna(fallback).fillna(self.global_prior)


@register
class Reactionary(StandTurnaround):

    name = "reactionary"
    needs_context = True

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        joined = self._join(df, self._arrivals(self.context))
        out = pd.DataFrame(index=df.index)
        out["turn_inb_late"] = _sec(joined["ib"], joined["sched"]).to_numpy()
        slack = _sec(df["SCHED_TIME_UTC_mvt"].reset_index(drop=True), joined["ib"]).to_numpy()
        out["turn_slack"] = slack
        out["turn_slack_excess"] = slack - self._prior(df).to_numpy()
        return out


@register
class RunwayConfig(FeatureFamily):

    name = "runway_config"
    needs_context = True

    COLUMN_ORDER = (
        "rcfg_n_dep_rwy", "rcfg_n_arr_rwy", "rcfg_mixed_n",
        "rcfg_rwy_share", "rcfg_rwy_rank", "rcfg_rwy_arr_n", "rcfg_rwy_arr_share",
        "rcfg_since_change", "rcfg_until_change", "rcfg_switch_4h",
        "rcfg_cat_config", "rcfg_cat_depset", "rcfg_dev", "rcfg_lvl",
    )

    def __init__(self, freq: str = "h", min_n: int = 2, min_share: float = 0.10,
                 min_count: int = 30, **params):
        super().__init__(freq=freq, min_n=min_n, min_share=min_share,
                         min_count=min_count, **params)
        self.freq = freq
        self.min_n = min_n
        self.min_share = min_share
        self.min_count = min_count

    def _buckets(self, ctx: pd.DataFrame) -> pd.DataFrame:
        apt = _local_airport(ctx)
        rwy = ctx["RUNWAY_mvt"].astype("str")
        frame = pd.DataFrame(
            {
                "apt": apt.to_numpy(),
                "b": ctx["MVT_TIME_UTC_mvt"].dt.floor(self.freq).to_numpy(),
                "k": np.where((ctx["PHASE_mvt"] == "DEP").to_numpy(), "D", "A"),
                "r": rwy.to_numpy(),
            }
        )
        frame = frame[pd.notna(frame["r"]) & pd.notna(frame["b"]) & pd.notna(frame["apt"])]
        return frame.groupby(["apt", "b", "k", "r"], observed=True).size().rename("n").reset_index()

    def _config_table(self, ctx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        counts = self._buckets(ctx)
        if counts.empty:
            empty = pd.DataFrame(
                columns=["apt", "b", "dep_set", "arr_set", "config", "n_dep_rwy", "n_arr_rwy",
                         "mixed_n", "dep_tot", "arr_tot", "run_start", "next_start", "is_switch"]
            )
            return empty, counts
        total = counts.groupby(["apt", "b", "k"], observed=True)["n"].transform("sum")
        counts = counts.assign(share=counts["n"] / total, total=total)
        active = counts[(counts["n"] >= self.min_n) & (counts["share"] >= self.min_share)]

        sets = (
            active.sort_values("r", kind="stable")
            .groupby(["apt", "b", "k"], observed=True)["r"]
            .agg("+".join)
            .unstack("k")
        )
        for c in ("D", "A"):
            if c not in sets.columns:
                sets[c] = np.nan
        sets = sets.rename(columns={"D": "dep_set", "A": "arr_set"}).fillna("")
        sets["config"] = sets["dep_set"].str.cat(sets["arr_set"], sep=">")
        sets["n_dep_rwy"] = _n_parts(sets["dep_set"])
        sets["n_arr_rwy"] = _n_parts(sets["arr_set"])
        sets["mixed_n"] = [
            len(set(d.split("+")) & set(a.split("+"))) if d and a else 0
            for d, a in zip(sets["dep_set"], sets["arr_set"], strict=True)
        ]
        totals = (
            counts.drop_duplicates(["apt", "b", "k"])
            .pivot_table(index=["apt", "b"], columns="k", values="total", aggfunc="first")
            .rename(columns={"D": "dep_tot", "A": "arr_tot"})
        )
        for c in ("dep_tot", "arr_tot"):
            if c not in totals.columns:
                totals[c] = np.nan
        table = sets.join(totals, how="left").reset_index()
        table = self._changes(table)
        return table, counts

    def _changes(self, table: pd.DataFrame) -> pd.DataFrame:
        table = table.sort_values(["apt", "b"], kind="stable").reset_index(drop=True)
        cfg = table["config"].to_numpy()
        apt = table["apt"].to_numpy()
        b = _ns_values(table["b"])
        nat = np.iinfo("int64").min

        new_run = np.empty(len(table), dtype=bool)
        new_run[0] = True
        new_run[1:] = (cfg[1:] != cfg[:-1]) | (apt[1:] != apt[:-1])
        run = np.cumsum(new_run) - 1

        starts = b[new_run]
        run_apt = apt[new_run]
        first_of_apt = np.r_[True, run_apt[1:] != run_apt[:-1]]
        last_of_apt = np.r_[run_apt[:-1] != run_apt[1:], True]

        run_start = np.where(first_of_apt, nat, starts)
        next_start = np.where(last_of_apt, nat, np.r_[starts[1:], nat])
        table["run_start"] = run_start[run]
        table["next_start"] = next_start[run]
        table["is_switch"] = new_run & ~first_of_apt[run]
        return table

    def _assign(self, df: pd.DataFrame) -> pd.DataFrame:
        ctx = self.context if self.context is not None else df
        table, counts = self._config_table(ctx)
        self._table = table
        q = pd.DataFrame(
            {
                "apt": df["ADEP_mvt"].astype("str").to_numpy(),
                "b": df["MVT_TIME_UTC_mvt"].dt.floor(self.freq).to_numpy(),
                "r": df["RUNWAY_mvt"].astype("str").to_numpy(),
                "t": _ns(df["MVT_TIME_UTC_mvt"]),
            }
        )
        out = q.merge(table, on=["apt", "b"], how="left")

        dep_counts = counts[counts["k"] == "D"][["apt", "b", "r", "n"]].rename(columns={"n": "nd"})
        arr_counts = counts[counts["k"] == "A"][["apt", "b", "r", "n"]].rename(columns={"n": "na"})
        out = out.merge(dep_counts, on=["apt", "b", "r"], how="left")
        out = out.merge(arr_counts, on=["apt", "b", "r"], how="left")

        rank = dep_counts.sort_values(["apt", "b", "nd", "r"], ascending=[True, True, False, True])
        rank["rank"] = rank.groupby(["apt", "b"], observed=True).cumcount()
        out = out.merge(rank[["apt", "b", "r", "rank"]], on=["apt", "b", "r"], how="left")
        return out

    def _keys(self, assigned: pd.DataFrame, df: pd.DataFrame):
        apt = assigned["apt"]
        cfg = assigned["config"]
        stand = df["STAND_mvt"].astype("str").reset_index(drop=True)
        rwy = assigned["r"].where(assigned["r"].notna() & (assigned["r"] != "nan"))
        k_cfg3 = apt.str.cat([cfg, stand, rwy], sep="|")
        k_cfg2 = apt.str.cat([cfg, rwy], sep="|")
        k_base3 = apt.str.cat([stand, rwy], sep="|")
        k_base2 = apt.str.cat(rwy, sep="|")
        return k_cfg3, k_cfg2, k_base3, k_base2

    def _fit(self, train: pd.DataFrame) -> None:
        dep = train[(train["PHASE_mvt"] == "DEP") & train[TARGET].notna()]
        assigned = self._assign(dep)
        y = dep[TARGET].reset_index(drop=True)
        self.config_levels = pd.Index(assigned["config"].dropna().unique())
        self.depset_levels = pd.Index(assigned["dep_set"].dropna().unique())
        self.tables = []
        for key in self._keys(assigned, dep):
            g = y.groupby(key.to_numpy())
            stats = pd.DataFrame({"median": g.median(), "n": g.size()})
            self.tables.append(stats[stats["n"] >= self.min_count]["median"])

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assigned = self._assign(df)
        out = pd.DataFrame(index=df.index)
        out["rcfg_n_dep_rwy"] = assigned["n_dep_rwy"].to_numpy(dtype="float64")
        out["rcfg_n_arr_rwy"] = assigned["n_arr_rwy"].to_numpy(dtype="float64")
        out["rcfg_mixed_n"] = assigned["mixed_n"].to_numpy(dtype="float64")

        dep_tot = assigned["dep_tot"].to_numpy(dtype="float64")
        arr_tot = assigned["arr_tot"].to_numpy(dtype="float64")
        nd = assigned["nd"].to_numpy(dtype="float64")
        na = assigned["na"].to_numpy(dtype="float64")
        out["rcfg_rwy_share"] = np.where(dep_tot > 0, nd / np.where(dep_tot > 0, dep_tot, 1.0),
                                         np.nan)
        out["rcfg_rwy_rank"] = assigned["rank"].to_numpy(dtype="float64")
        out["rcfg_rwy_arr_n"] = np.where(np.isfinite(na), na, 0.0)
        out["rcfg_rwy_arr_share"] = np.where(
            arr_tot > 0, np.where(np.isfinite(na), na, 0.0) / np.where(arr_tot > 0, arr_tot, 1.0),
            np.nan,
        )

        t = assigned["t"].to_numpy()
        nat = np.iinfo("int64").min
        run_start = assigned["run_start"].to_numpy(dtype="float64")
        next_start = assigned["next_start"].to_numpy(dtype="float64")
        known = np.isfinite(run_start) & (run_start != nat)
        out["rcfg_since_change"] = np.where(known, (t - run_start) / 1e9, np.nan)
        known = np.isfinite(next_start) & (next_start != nat)
        out["rcfg_until_change"] = np.where(known, (next_start - t) / 1e9, np.nan)
        out["rcfg_switch_4h"] = self._switch_count(assigned)

        out["rcfg_cat_config"] = self.config_levels.get_indexer(
            assigned["config"].where(assigned["config"].notna(), None)
        ).astype("int32")
        out["rcfg_cat_depset"] = self.depset_levels.get_indexer(
            assigned["dep_set"].where(assigned["dep_set"].notna(), None)
        ).astype("int32")

        cfg3, cfg2, base3, base2 = self._keys(assigned, df)
        t3, t2, b3, b2 = (k.map(table).to_numpy(dtype="float64")
                          for k, table in zip((cfg3, cfg2, base3, base2), self.tables,
                                              strict=True))
        dev3 = t3 - b3
        dev2 = t2 - b2
        dev = np.where(np.isfinite(dev3), dev3, dev2)
        level = np.where(np.isfinite(dev3), 3, np.where(np.isfinite(dev2), 2, 0))
        out["rcfg_dev"] = dev
        out["rcfg_lvl"] = level.astype("int8")
        return out[list(self.COLUMN_ORDER)]

    def _switch_count(self, assigned: pd.DataFrame) -> np.ndarray:
        table = self._table
        if table.empty:
            return np.full(len(assigned), np.nan)
        switch = table[table["is_switch"]]
        ev_t = _ns_values(switch["b"])
        q_t = assigned["t"].to_numpy()
        window = np.int64(pd.Timedelta("2h").value)
        return _window_count(
            assigned["apt"], q_t + window, switch["apt"], ev_t, pd.Timedelta("4h")
        ).astype("float64")


def _n_parts(s: pd.Series) -> np.ndarray:
    v = s.to_numpy()
    return np.array([0 if not x else x.count("+") + 1 for x in v], dtype="float64")


def _ns_values(t) -> np.ndarray:
    s = pd.Series(t).reset_index(drop=True)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.to_numpy("datetime64[ns]").astype("int64")


@register
class SourceDisagreement(FeatureFamily):

    name = "disagreement"

    PLAN_COLS = ("SCHED_TIME_UTC_mvt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt")
    BLOCKS = ("plan", "aobt", "arvt")

    def __init__(self, blocks: Iterable[str] = BLOCKS, **params):
        super().__init__(blocks=list(blocks), **params)
        unknown = set(blocks) - set(self.BLOCKS)
        if unknown:
            raise ValueError(f"unknown disagreement block(s) {sorted(unknown)}")
        self.blocks = tuple(blocks)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        sched, eobt, iobt, lobt = (df[c] for c in self.PLAN_COLS)
        aobt = df["AOBT_3_flt"]

        plan = np.column_stack(
            [_sec(df[c], sched).to_numpy(dtype="float64") for c in self.PLAN_COLS]
        )
        present = np.isfinite(plan)
        n_sources = present.sum(axis=1)
        any_source = n_sources > 0

        if "plan" in self.blocks:
            out["dis_lobt_eobt"] = _sec(lobt, eobt)
            out["dis_iobt_eobt"] = _sec(iobt, eobt)
            out["dis_lobt_iobt"] = _sec(lobt, iobt)
            out["dis_sched_eobt"] = _sec(sched, eobt)
            with np.errstate(invalid="ignore"):
                hi = np.where(
                    any_source, np.nanmax(np.where(present, plan, -np.inf), axis=1), np.nan
                )
                lo = np.where(
                    any_source, np.nanmin(np.where(present, plan, np.inf), axis=1), np.nan
                )
            out["dis_plan_range"] = np.where(any_source, hi - lo, np.nan)
            out["dis_plan_std"] = np.nanstd(np.where(present, plan, np.nan), axis=1)
            out["dis_plan_distinct"] = _n_distinct(plan, present).astype("int8")
            out["dis_n_sources"] = n_sources.astype("int8")
            has_plan = (lobt.notna() & eobt.notna()).to_numpy()
            out["dis_eobt_updated"] = np.where(
                has_plan, (_sec(lobt, eobt).to_numpy() != 0).astype("int8"), -1
            ).astype("int8")

        if "aobt" in self.blocks:
            out["dis_aobt_iobt"] = _sec(aobt, iobt)
            med = np.where(
                any_source, np.nanmedian(np.where(present, plan, np.nan), axis=1), np.nan
            )
            out["dis_aobt_plan_med"] = _sec(aobt, sched).to_numpy(dtype="float64") - med

        if "arvt" in self.blocks:
            out["dis_arvt3_arvt1"] = _sec(df["ARVT_3_flt"], df["ARVT_1_flt"])
            out["dis_eet_filed"] = _sec(df["ARVT_1_flt"], eobt)
            out["dis_eet_actual"] = _sec(df["ARVT_3_flt"], aobt)
            out["dis_eet_diff"] = out["dis_eet_actual"] - out["dis_eet_filed"]

        out["dis_no_flt"] = lobt.isna().astype("int8")
        return out[[c for c in self.COLUMN_ORDER if c in out.columns]]

    COLUMN_ORDER = (
        "dis_lobt_eobt", "dis_iobt_eobt", "dis_lobt_iobt", "dis_sched_eobt", "dis_aobt_iobt",
        "dis_plan_range", "dis_plan_std", "dis_plan_distinct", "dis_n_sources",
        "dis_aobt_plan_med", "dis_eobt_updated",
        "dis_arvt3_arvt1", "dis_eet_filed", "dis_eet_actual", "dis_eet_diff", "dis_no_flt",
    )


def _n_distinct(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    filled = np.where(present, values, np.nan)
    ordered = np.sort(filled, axis=1)
    same = ordered[:, 1:] == ordered[:, :-1]
    return np.where(present.any(axis=1), present.sum(axis=1) - same.sum(axis=1), 0)


@register
class SubMinute(FeatureFamily):

    name = "subminute"
    JITTER_BAND = 7

    def __init__(self, min_count: int = 200, **params):
        super().__init__(min_count=min_count, **params)
        self.min_count = min_count

    @staticmethod
    def _fold(sec) -> np.ndarray:
        return np.mod(np.asarray(sec, dtype="float64") + 30.0, 60.0) - 30.0

    def _fit(self, train: pd.DataFrame) -> None:
        dep = train[(train["PHASE_mvt"] == "DEP") & train[TARGET].notna()]
        apt = dep["ADEP_mvt"].astype("str")
        phase_mvt = self._fold(dep["MVT_TIME_UTC_mvt"].dt.second.to_numpy())
        frame = pd.DataFrame(
            {
                "apt": apt.to_numpy(),
                "near": (np.abs(phase_mvt) <= self.JITTER_BAND).astype("float64"),
                "cos": np.cos(2 * np.pi * dep[TARGET].to_numpy(dtype="float64") / 60.0),
                "sin": np.sin(2 * np.pi * dep[TARGET].to_numpy(dtype="float64") / 60.0),
            }
        )
        g = frame.groupby("apt", observed=True)
        stats = g.agg(near=("near", "mean"), cos=("cos", "mean"), sin=("sin", "mean"), n=("near", "size"))
        stats = stats[stats["n"] >= self.min_count]
        self.jitter = stats["near"]
        self.phase_mu = np.arctan2(stats["sin"], stats["cos"]) * 60.0 / (2 * np.pi)
        self.phase_r = np.hypot(stats["sin"], stats["cos"])
        self.global_jitter = float(frame["near"].mean()) if len(frame) else np.nan
        self.global_mu = 0.0
        self.global_r = 0.0

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        mvt_phase = self._fold(df["MVT_TIME_UTC_mvt"].dt.second.to_numpy())
        out["sub_mvt_phase"] = mvt_phase
        aobt_sec = df["AOBT_3_flt"].dt.second.to_numpy(dtype="float64")
        out["sub_aobt_phase"] = np.where(
            df["AOBT_3_flt"].notna().to_numpy(), self._fold(aobt_sec), np.nan
        )
        base = _sec(df["MVT_TIME_UTC_mvt"], df["AOBT_3_flt"]).to_numpy(dtype="float64")
        base_phase = self._fold(base)
        out["sub_base_phase"] = np.where(np.isfinite(base), base_phase, np.nan)

        apt = df["ADEP_mvt"].astype("str")
        out["sub_apt_jitter"] = apt.map(self.jitter).fillna(self.global_jitter).to_numpy()
        mu = apt.map(self.phase_mu).fillna(self.global_mu).to_numpy(dtype="float64")
        out["sub_apt_phase_mu"] = mu
        out["sub_apt_phase_r"] = apt.map(self.phase_r).fillna(self.global_r).to_numpy()
        out["sub_phase_dev"] = np.where(
            np.isfinite(base), self._fold(base_phase - mu), self._fold(mvt_phase - mu)
        )
        return out


def _group_slices(codes_sorted: np.ndarray):
    cuts = np.flatnonzero(np.diff(codes_sorted)) + 1
    starts = np.concatenate(([0], cuts))
    stops = np.concatenate((cuts, [len(codes_sorted)]))
    return zip(starts, stops, strict=True)


def _sequence_stats(key: pd.Series, t: np.ndarray, window: np.int64):
    codes = pd.factorize(key, use_na_sentinel=False)[0]
    order = np.lexsort((t, codes))
    ts, cs = t[order], codes[order]
    n = len(ts)

    prev_gap = np.full(n, np.nan)
    next_gap = np.full(n, np.nan)
    gap_mean5 = np.full(n, np.nan)
    pos = np.zeros(n, dtype="int16")
    after = np.zeros(n, dtype="int16")

    for start, stop in _group_slices(cs):
        g = ts[start:stop]
        gaps = np.diff(g) / 1e9
        prev_gap[start + 1 : stop] = gaps
        next_gap[start : stop - 1] = gaps
        if len(gaps):
            csum = np.concatenate(([0.0], np.cumsum(gaps)))
            k = np.minimum(np.arange(1, len(g)), 5)
            idx = np.arange(1, len(g))
            gap_mean5[start + 1 : stop] = (csum[idx] - csum[idx - k]) / k
        i = np.arange(len(g))
        pos[start:stop] = i - np.searchsorted(g, g - window, side="left")
        after[start:stop] = np.searchsorted(g, g + window, side="right") - i - 1

    inverse = np.empty(n, dtype="int64")
    inverse[order] = np.arange(n)
    return (
        prev_gap[inverse], next_gap[inverse], gap_mean5[inverse], pos[inverse], after[inverse],
    )


def _taxiing_now(key: pd.Series, t: np.ndarray, aobt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    codes = pd.factorize(key, use_na_sentinel=False)[0]
    result = np.zeros(len(t), dtype="int16")
    order = np.lexsort((t, codes))
    cs = codes[order]
    for start, stop in _group_slices(cs):
        rows = order[start:stop]
        here = valid[rows]
        started = np.sort(aobt[rows][here])
        airborne = np.sort(t[rows][here])
        q = t[rows]
        result[rows] = (
            np.searchsorted(started, q, side="right") - np.searchsorted(airborne, q, side="right")
        ).astype("int16")
    return result


def _ns(t: pd.Series) -> np.ndarray:
    return t.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64")


def _last_sunday_utc(year: int, month: int) -> pd.Timestamp:
    last = pd.Timestamp(year=int(year), month=month, day=1, tz="UTC") + pd.offsets.MonthEnd(0)
    return last - pd.Timedelta(days=(last.dayofweek + 1) % 7)


def _eu_summer_time(t: pd.Series) -> np.ndarray:
    year = t.dt.year.to_numpy()
    out = np.zeros(len(t), dtype=bool)
    ts = _ns(t)
    for y in np.unique(year[~pd.isna(year)]):
        start = (_last_sunday_utc(y, 3) + pd.Timedelta(hours=1)).value
        end = (_last_sunday_utc(y, 10) + pd.Timedelta(hours=1)).value
        m = year == y
        out[m] = (ts[m] >= start) & (ts[m] < end)
    return out


def _centroid_hour(t: pd.Series) -> float:
    secs = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).to_numpy(dtype="float64")
    ang = 2 * np.pi * secs / 86400.0
    mean = np.angle(np.mean(np.exp(1j * ang)))
    return float((mean % (2 * np.pi)) * 24.0 / (2 * np.pi))


@register
class LocalTime(FeatureFamily):

    name = "local_time"
    needs_context = True

    BLOCKS = ("hour", "enc")

    ENC_KEYS = ("hour", "op", "stand", "rwy", "typ", "dow", "fltno")

    def __init__(
        self,
        blocks: Iterable[str] = BLOCKS,
        clock: str = "local",
        min_count: int = 100,
        enc_keys: Iterable[str] = ("hour",),
        fltno_m: float = 50.0,
        m: float | None = None,
        **params,
    ):
        super().__init__(blocks=list(blocks), clock=clock, min_count=min_count,
                         enc_keys=list(enc_keys), fltno_m=float(fltno_m), m=m, **params)
        unknown = set(blocks) - set(self.BLOCKS)
        if unknown:
            raise ValueError(f"unknown local_time block(s) {sorted(unknown)}")
        if clock not in ("local", "utc"):
            raise ValueError(f"clock must be 'local' or 'utc', got {clock!r}")
        unknown_keys = set(enc_keys) - set(self.ENC_KEYS)
        if unknown_keys:
            raise ValueError(f"unknown local_time enc_key(s) {sorted(unknown_keys)}")
        if fltno_m <= 0:
            raise ValueError(f"fltno_m must be positive, got {fltno_m!r}")
        if m is not None and m <= 0:
            raise ValueError(f"m must be positive, got {m!r}")
        self.blocks = tuple(blocks)
        self.clock = clock
        self.min_count = min_count
        self.enc_keys = tuple(enc_keys)
        self.fltno_m = float(fltno_m)
        self.m = None if m is None else float(m)

    def _fit_offsets(self, ctx: pd.DataFrame) -> None:
        apt = _local_airport(ctx)
        summer = _eu_summer_time(ctx["MVT_TIME_UTC_mvt"])
        frame = pd.DataFrame(
            {"apt": apt.to_numpy(), "s": summer, "t": ctx["MVT_TIME_UTC_mvt"].to_numpy()},
            index=ctx.index,
        )
        self.network_centroid = _centroid_hour(ctx["MVT_TIME_UTC_mvt"])
        self.offsets: dict[tuple[str, bool], int] = {}
        for (a, s), g in frame.groupby(["apt", "s"], observed=True):
            delta = (self.network_centroid - _centroid_hour(pd.Series(g["t"])) + 12) % 24 - 12
            self.offsets[(str(a), bool(s))] = int(np.round(delta))

    def _offset(self, df: pd.DataFrame) -> np.ndarray:
        summer = _eu_summer_time(df["MVT_TIME_UTC_mvt"])
        apt = df["ADEP_mvt"].astype("str").to_numpy()
        return np.array(
            [self.offsets.get((a, bool(s)), 0) for a, s in zip(apt, summer, strict=True)],
            dtype="int8",
        )

    def _hour_key(self, df: pd.DataFrame, offset: np.ndarray) -> pd.Series:
        utc_hour = df["MVT_TIME_UTC_mvt"].dt.hour.to_numpy()
        hour = (utc_hour + offset) % 24 if self.clock == "local" else utc_hour
        return df["ADEP_mvt"].astype("str").str.cat(pd.Series(hour, index=df.index).astype("str"),
                                                    sep="|")

    def _enc_key(self, name: str, df: pd.DataFrame, offset: np.ndarray) -> pd.Series:
        if name == "hour":
            return self._hour_key(df, offset)
        apt = df["ADEP_mvt"].astype("str")
        t = df["MVT_TIME_UTC_mvt"]
        if name == "fltno":
            return apt.str.cat([self._part(df, "AIRCRAFT_OPERATOR_flt"),
                                self._part(df, "FLIGHT_mvt")], sep="|")
        if name == "dow":
            local_minute = (t.dt.hour * 60 + t.dt.minute).to_numpy() + 60 * offset
            wrap = np.floor_divide(local_minute, 1440) if self.clock == "local" else 0
            part = pd.Series((t.dt.dayofweek.to_numpy() + wrap) % 7, index=df.index).astype("str")
        else:
            col = {"op": "AIRCRAFT_OPERATOR_flt", "stand": "STAND_mvt",
                   "rwy": "RUNWAY_mvt", "typ": "AIRCRAFT_TYPE_mvt"}[name]
            part = self._part(df, col)
        return apt.str.cat(part, sep="|")

    @staticmethod
    def _part(df: pd.DataFrame, col: str) -> pd.Series:
        return df[col].astype("str").where(df[col].notna())

    def _fit_fltno(self, z: pd.Series, dep: pd.DataFrame,
                   key: pd.Series) -> tuple[pd.Series, pd.Series]:
        op_key = self._enc_key("op", dep, np.zeros(len(dep), dtype="int8"))
        ok_op = op_key.notna()
        g_op = z[ok_op].groupby(op_key[ok_op])
        op_med = g_op.median()
        op_apt = pd.Series(op_med.index.str.rsplit("|", n=1).str[0], index=op_med.index)
        d_op = op_med - op_apt.map(self.apt_level).fillna(self.global_level)

        ok = key.notna()
        g = z[ok].groupby(key[ok])
        stats = pd.DataFrame({"median": g.median(), "n": g.size()})
        parent = pd.Series(stats.index.str.rsplit("|", n=1).str[0], index=stats.index)
        apt = pd.Series(parent.str.split("|").str[0].to_numpy(), index=stats.index)
        d_flt = stats["median"] - apt.map(self.apt_level).fillna(self.global_level)
        n = stats["n"].astype("float64")
        dev = n / (n + self.fltno_m) * (d_flt - parent.map(d_op).fillna(0.0))
        return dev, stats["n"]

    def _fit(self, train: pd.DataFrame) -> None:
        ctx = self.context if self.context is not None else train
        self._fit_offsets(ctx)
        if "enc" not in self.blocks:
            return
        dep = train[(train["PHASE_mvt"] == "DEP") & train[TARGET].notna()]
        dep = dep[dep["AOBT_3_flt"].notna()]
        z = dep[TARGET] - _sec(dep["MVT_TIME_UTC_mvt"], dep["AOBT_3_flt"])
        apt = dep["ADEP_mvt"].astype("str")
        self.apt_level = z.groupby(apt).median()
        self.global_level = float(z.median())
        offset = self._offset(dep)
        self.dev: dict[str, pd.Series] = {}
        self.cell_n: dict[str, pd.Series] = {}
        for name in self.enc_keys:
            key = self._enc_key(name, dep, offset)
            if name == "fltno":
                self.dev[name], self.cell_n[name] = self._fit_fltno(z, dep, key)
                continue
            ok = key.notna()
            g = z[ok].groupby(key[ok])
            stats = pd.DataFrame({"median": g.median(), "n": g.size()})
            if self.m is None:
                stats = stats[stats["n"] >= self.min_count]
            base = pd.Series(stats.index.str.rsplit("|", n=1).str[0], index=stats.index)
            dev = stats["median"] - base.map(self.apt_level).fillna(self.global_level)
            if self.m is not None:
                n = stats["n"].astype("float64")
                dev = n / (n + self.m) * dev
            self.dev[name] = dev
            self.cell_n[name] = stats["n"]

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        offset = self._offset(df)
        t = df["MVT_TIME_UTC_mvt"]

        if "hour" in self.blocks:
            out["lt_offset"] = offset
            out["lt_hour"] = ((t.dt.hour.to_numpy() + offset) % 24).astype("int8")
            local_minute = (t.dt.hour * 60 + t.dt.minute).to_numpy() + 60 * offset
            out["lt_minute_of_day"] = (local_minute % 1440).astype("int16")
            wrap = np.floor_divide(local_minute, 1440)
            out["lt_dow"] = ((t.dt.dayofweek.to_numpy() + wrap) % 7).astype("int8")

        if "enc" in self.blocks:
            apt = df["ADEP_mvt"].astype("str")
            out["lt_apt_lvl"] = apt.map(self.apt_level).fillna(self.global_level).astype("float32")
            for name in self.enc_keys:
                key = self._enc_key(name, df, offset)
                out[f"lt_{name}_dev"] = key.map(self.dev[name]).fillna(0.0).astype("float32")
                out[f"lt_{name}_n"] = key.map(self.cell_n[name]).fillna(0).astype("int32")
        return out


@register
class CarrierFill(FeatureFamily):

    name = "carrier_fill"

    BLOCKS = ("cat", "dev")
    PREFIX_RE = r"^([A-Z]{2,4})"

    def __init__(self, blocks: Iterable[str] = BLOCKS, min_count: int = 100,
                 map_min_count: int = 0, **params):
        super().__init__(blocks=list(blocks), min_count=int(min_count),
                         map_min_count=int(map_min_count), **params)
        unknown = set(blocks) - set(self.BLOCKS)
        if unknown:
            raise ValueError(f"unknown carrier_fill block(s) {sorted(unknown)}")
        if min_count < 1:
            raise ValueError(f"min_count must be >= 1, got {min_count!r}")
        if map_min_count < 0:
            raise ValueError(f"map_min_count must be >= 0, got {map_min_count!r}")
        self.blocks = tuple(blocks)
        self.min_count = int(min_count)
        self.map_min_count = int(map_min_count)

    @classmethod
    def prefix(cls, flight: pd.Series) -> pd.Series:
        up = flight.astype("string").str.strip().str.upper()
        return up.str.extract(cls.PREFIX_RE, expand=False)

    def _fit(self, train: pd.DataFrame) -> None:
        op = train["AIRCRAFT_OPERATOR_flt"].astype("string")
        pfx = self.prefix(train["FLIGHT_mvt"])
        both = pd.DataFrame({"pfx": pfx, "op": op}).dropna()
        counts = both.groupby(["pfx", "op"], observed=True).size()
        frame = counts.rename("n").reset_index().sort_values(
            ["pfx", "n", "op"], ascending=[True, False, True])
        keep = frame.groupby("pfx", observed=True).head(1)
        totals = frame.groupby("pfx", observed=True)["n"].sum()
        keep = keep[keep["pfx"].map(totals) >= max(self.map_min_count, 1)]
        self.op_map = pd.Series(keep["op"].to_numpy(), index=pd.Index(keep["pfx"], name="pfx"))
        self.levels = pd.Index(op.dropna().unique())

        dep = train[(train["PHASE_mvt"] == "DEP") & train[TARGET].notna()]
        dep = dep[dep["AOBT_3_flt"].notna()]
        z = dep[TARGET] - _sec(dep["MVT_TIME_UTC_mvt"], dep["AOBT_3_flt"])
        apt = dep["ADEP_mvt"].astype("str")
        self.apt_level = z.groupby(apt).median()
        self.global_level = float(z.median())
        key = apt.str.cat(dep["AIRCRAFT_OPERATOR_flt"].astype("str").where(
            dep["AIRCRAFT_OPERATOR_flt"].notna()), sep="|")
        ok = key.notna()
        g = z[ok].groupby(key[ok])
        stats = pd.DataFrame({"median": g.median(), "n": g.size()})
        stats = stats[stats["n"] >= self.min_count]
        parent = pd.Series(stats.index.str.rsplit("|", n=1).str[0], index=stats.index)
        self.dev = stats["median"] - parent.map(self.apt_level).fillna(self.global_level)
        self.cell_n = stats["n"]

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        op = df["AIRCRAFT_OPERATOR_flt"].astype("string")
        missing = op.isna()
        recovered = self.prefix(df["FLIGHT_mvt"]).map(self.op_map).where(missing)
        out = pd.DataFrame(index=df.index)
        out["car_filled"] = recovered.notna().astype("int8")
        if "cat" in self.blocks:
            values = recovered.where(recovered.notna(), None)
            out["cat_car_fill"] = self.levels.get_indexer(values).astype("int32")
        if "dev" in self.blocks:
            key = df["ADEP_mvt"].astype("str").str.cat(recovered, sep="|")
            out["car_op_dev"] = key.map(self.dev).fillna(0.0).astype("float32")
            out["car_op_n"] = key.map(self.cell_n).fillna(0).astype("int32")
        return out


_STAND_PREFIX = re.compile(r"^([A-Za-z]+)")


@register
class StandClass(FeatureFamily):

    name = "stand_class"

    def __init__(self, k: int = 4, min_count: int = 10, seed: int = 0, **params):
        super().__init__(k=int(k), min_count=int(min_count), seed=int(seed), **params)
        if int(k) < 2:
            raise ValueError(f"stand_class k must be >= 2, got {k!r}")
        if int(min_count) < 1:
            raise ValueError(f"stand_class min_count must be >= 1, got {min_count!r}")
        self.k, self.min_count, self.seed = int(k), int(min_count), int(seed)

    @staticmethod
    def _prefix(stand: pd.Series) -> pd.Series:
        v = stand.astype("string")
        return v.str.extract(_STAND_PREFIX, expand=False).fillna(v.str[:1])

    @staticmethod
    def _as_label(c: pd.Series) -> pd.Series:
        return c.astype("Int16").astype("string")

    def _fit(self, train: pd.DataFrame) -> None:
        from sklearn.cluster import KMeans

        dep = train[(train["PHASE_mvt"] == "DEP") & train[TARGET].notna()]
        apt = dep["ADEP_mvt"].astype("string")
        stand = dep["STAND_mvt"].astype("string")
        key = apt.str.cat(stand, sep="|")
        y = dep[TARGET]

        g = y.groupby(key)
        stats = pd.DataFrame({"median": g.median(), "p10": g.quantile(0.10), "n": g.size()})
        stats["apt"] = pd.Series(stats.index, index=stats.index).str.split("|").str[0]

        self.apt_median = y.groupby(apt).median()
        self.global_median = float(y.median())
        self.global_p10 = float(y.quantile(0.10))

        labels: dict[str, int] = {}
        for _, sub in stats.groupby("apt"):
            big = sub[sub["n"] >= self.min_count]
            if len(big) < self.k:
                labels.update(dict.fromkeys(sub.index, 0))
                continue
            x = np.column_stack(
                [big["median"].to_numpy(float), big["p10"].to_numpy(float),
                 np.log10(big["n"].to_numpy(float))]
            )
            x = (x - x.mean(0)) / (x.std(0) + 1e-9)
            lab = KMeans(n_clusters=self.k, random_state=self.seed, n_init=10).fit_predict(x)
            order = pd.Series(big["median"].to_numpy(float)).groupby(lab).mean().sort_values()
            remap = {int(old): new for new, old in enumerate(order.index)}
            labels.update({i: remap[int(v)] for i, v in zip(big.index, lab, strict=True)})
        self.stand_cluster = pd.Series(labels, dtype="int16")

        ck0 = apt.str.cat(self._as_label(key.map(self.stand_cluster)), sep="|")
        cmed = y.groupby(ck0).median()
        parts = pd.Series(cmed.index, index=cmed.index).str.split("|")
        order = pd.DataFrame(
            {"apt": parts.str[0].to_numpy(), "c": parts.str[1].to_numpy().astype(int),
             "m": cmed.to_numpy()}
        ).sort_values(["apt", "m"], kind="stable")
        order["new"] = order.groupby("apt").cumcount()
        remap = dict(zip(zip(order["apt"], order["c"], strict=True), order["new"], strict=True))
        sc_apt = pd.Series(
            self.stand_cluster.index, index=self.stand_cluster.index
        ).str.split("|").str[0]
        self.stand_cluster = pd.Series(
            [remap[(a, int(c))] for a, c in zip(sc_apt, self.stand_cluster, strict=True)],
            index=self.stand_cluster.index, dtype="int16",
        )

        pref = self._prefix(pd.Series(stats.index, index=stats.index).str.split("|").str[1])
        pkey = stats["apt"].str.cat(pref, sep="|")
        w = pd.DataFrame(
            {"pkey": pkey, "c": stats.index.map(self.stand_cluster), "n": stats["n"]}
        ).dropna()
        modal = w.groupby(["pkey", "c"])["n"].sum().reset_index().sort_values("n")
        self.prefix_cluster = modal.groupby("pkey")["c"].last().astype("int16")
        self.prefix_levels = pd.Index(pkey.dropna().unique())

        ck = apt.str.cat(self._as_label(key.map(self.stand_cluster)), sep="|")
        gc = y.groupby(ck)
        self.class_stats = pd.DataFrame(
            {"median": gc.median(), "p10": gc.quantile(0.10), "n": gc.size()}
        )

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        apt = df["ADEP_mvt"].astype("string")
        stand = df["STAND_mvt"].astype("string")
        pref = self._prefix(stand)
        key = apt.str.cat(stand, sep="|")
        pkey = apt.str.cat(pref, sep="|")

        own = key.map(self.stand_cluster)
        c = own.fillna(pkey.map(self.prefix_cluster))
        ck = apt.str.cat(self._as_label(c), sep="|")
        apt_med = apt.map(self.apt_median).fillna(self.global_median).astype(float)

        out = pd.DataFrame(index=df.index)
        out["sc_cluster"] = c.fillna(-1).astype("int16")
        out["sc_from_prefix"] = (own.isna() & c.notna()).astype("int8")
        out["sc_prefix_code"] = self.prefix_levels.get_indexer(
            pkey.where(pkey.notna(), None)
        ).astype("int32")
        med = ck.map(self.class_stats["median"]).astype(float).fillna(apt_med)
        out["sc_class_median"] = med.astype("float32")
        out["sc_class_p10"] = (
            ck.map(self.class_stats["p10"]).astype(float)
            .fillna(apt.map(self.apt_median).astype(float)).fillna(self.global_p10)
        ).astype("float32")
        out["sc_class_n"] = ck.map(self.class_stats["n"]).fillna(0).astype("int32")
        out["sc_class_dev"] = (med - apt_med).astype("float32")
        return out


@register
class Destination(FeatureFamily):

    name = "destination"
    BLOCKS = ("region", "count", "reroute")
    PAIRS = (
        ("des_reroute", "ADES_mvt", "ADES_flt"),
        ("des_refile", "ADES_mvt", "ADES_FILED_flt"),
        ("des_flt_refiled", "ADES_flt", "ADES_FILED_flt"),
    )

    def __init__(self, blocks: Iterable[str] = BLOCKS, **params):
        super().__init__(blocks=list(blocks), **params)
        unknown = set(blocks) - set(self.BLOCKS)
        if unknown:
            raise ValueError(f"unknown destination block(s) {sorted(unknown)}")
        self.blocks = tuple(blocks)

    @staticmethod
    def _txt(df: pd.DataFrame, col: str) -> pd.Series:
        return df[col].astype("string") if col in df else pd.Series(
            pd.NA, index=df.index, dtype="string"
        )

    def _fit(self, train: pd.DataFrame) -> None:
        dep = train[train["PHASE_mvt"] == "DEP"]
        ades = self._txt(dep, "ADES_mvt")
        self.region_levels = pd.Index(ades.str[:1].dropna().unique())
        self.country_levels = pd.Index(ades.str[:2].dropna().unique())
        key = self._txt(dep, "ADEP_mvt").str.cat(ades, sep="|")
        self.pair_n = key.dropna().value_counts()

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        ades = self._txt(df, "ADES_mvt")
        adep = self._txt(df, "ADEP_mvt")

        if "region" in self.blocks:
            region, country = ades.str[:1], ades.str[:2]
            out["des_region_code"] = self.region_levels.get_indexer(
                region.where(region.notna(), None)
            ).astype("int32")
            out["des_country_code"] = self.country_levels.get_indexer(
                country.where(country.notna(), None)
            ).astype("int32")
            known = (ades.notna() & adep.notna()).to_numpy()
            same = (country == adep.str[:2]).fillna(False).to_numpy()
            out["des_domestic"] = np.where(known, same.astype("int8"), -1).astype("int8")

        if "count" in self.blocks:
            out["des_pair_n"] = (
                adep.str.cat(ades, sep="|").map(self.pair_n).fillna(0).astype("int32")
            )

        if "reroute" in self.blocks:
            for name, a, b in self.PAIRS:
                left, right = self._txt(df, a), self._txt(df, b)
                both = (left.notna() & right.notna()).to_numpy()
                differs = (left != right).fillna(False).to_numpy().astype("int8")
                out[name] = np.where(both, differs, -1).astype("int8")
        return out
