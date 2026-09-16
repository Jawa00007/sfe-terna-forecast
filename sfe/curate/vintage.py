"""Point-in-time ("as-of") views over the vintaged landing store (PRD P-1 / DQ-2).

Rule: a value is visible at decision time ``as_of`` only if its ``_publication_ts`` is
*strictly before* ``as_of``. Among the visible vintages for a given
``(_area, _mtu_start, field)`` the one with the latest ``_publication_ts`` wins - so a
definitive value automatically supersedes its preliminary once it has been published, and
never before.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

_KEYS = ["_area", "_mtu_start", "field"]


def _as_utc(ts: datetime | str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def as_of_view(landing: pd.DataFrame, as_of: datetime | str | pd.Timestamp) -> pd.DataFrame:
    """Latest visible vintage per ``(_area, _mtu_start, field)`` as of *as_of*."""
    if landing.empty:
        return landing.copy()
    as_of = _as_utc(as_of)
    df = landing.copy()
    df["_publication_ts"] = pd.to_datetime(df["_publication_ts"], utc=True)

    visible = df[df["_publication_ts"] < as_of]
    if visible.empty:
        return visible

    visible = visible.sort_values("_publication_ts")
    latest = visible.groupby(_KEYS, as_index=False, dropna=False).tail(1)
    return latest.reset_index(drop=True)


def latest_view(landing: pd.DataFrame) -> pd.DataFrame:
    """Most recent vintage per key with no as-of cutoff - for assembling training labels."""
    return as_of_view(landing, pd.Timestamp("2999-01-01", tz="UTC"))


def pivot_fields(view: pd.DataFrame, value_col: str = "value_num") -> pd.DataFrame:
    """One row per ``(_area, _mtu_start)``, one column per ``field``."""
    if view.empty:
        return pd.DataFrame(columns=["_area", "_mtu_start"])
    wide = view.pivot_table(
        index=["_area", "_mtu_start"], columns="field", values=value_col, aggfunc="last"
    )
    wide.columns.name = None
    return wide.reset_index()


def assert_no_leakage(view: pd.DataFrame, as_of: datetime | str | pd.Timestamp) -> None:
    """Raise if *view* contains any row published at or after *as_of* (test helper)."""
    if view.empty:
        return
    as_of = _as_utc(as_of)
    bad = pd.to_datetime(view["_publication_ts"], utc=True) >= as_of
    if bad.any():
        n = int(bad.sum())
        raise AssertionError(
            f"point-in-time leak: {n} row(s) published >= as_of {as_of.isoformat()}"
        )
