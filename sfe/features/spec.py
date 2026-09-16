"""Point-in-time feature-store contracts (PRD F-4, F-5, P-1).

A feature row is keyed ``(_area, _mtu_start, _as_of)``. ``_as_of`` is the decision timestamp;
a builder may only use source values whose ``_publication_ts`` is strictly before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Protocol

import pandas as pd

from sfe.config import load_yaml
from sfe.curate import regimes
from sfe.curate.timebase import ROME, day_grid

KEY = ["_area", "_mtu_start"]

# Decision offset in hours (Europe/Rome) before delivery-day 00:00, per horizon label.
# Negative = the decision is taken during the delivery day (later MI sessions).
HORIZON_OFFSET_H = {"D-1": 15, "MI1": 3, "MI2": -9}


def to_utc(ts: object) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def as_of_for_horizon(delivery_day: date, horizon: str = "D-1") -> pd.Timestamp:
    """Decision timestamp (UTC) for *horizon* relative to *delivery_day*."""
    local_midnight = datetime(delivery_day.year, delivery_day.month, delivery_day.day, tzinfo=ROME)
    offset = HORIZON_OFFSET_H.get(horizon, 15)
    return to_utc(pd.Timestamp(local_midnight - timedelta(hours=offset)))


@dataclass(frozen=True)
class DecisionContext:
    """One forecasting decision point."""

    as_of: pd.Timestamp
    delivery_day: date
    horizon: str = "D-1"                      # label only; models are per-horizon (F-9)
    areas: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", to_utc(self.as_of))

    @property
    def resolution(self) -> str:
        return regimes.mtu_resolution(self.delivery_day)


def _default_areas() -> tuple[str, ...]:
    z = load_yaml("zones")
    return tuple(z["levels"][z["target_level"]]["areas"])


def target_skeleton(
    delivery_day: date, areas: tuple[str, ...] = (), resolution: str = ""
) -> pd.DataFrame:
    """The ``(_area, _mtu_start)`` grid a forecast must cover for *delivery_day*."""
    areas = areas or _default_areas()
    resolution = resolution or regimes.mtu_resolution(delivery_day)
    grid = day_grid(delivery_day, resolution)
    return pd.DataFrame(
        [{"_area": a, "_mtu_start": pd.Timestamp(g)} for a in areas for g in grid]
    )


class FeatureBuilder(Protocol):
    """Produces a frame keyed on ``KEY`` plus its feature columns, PIT-correct for ``ctx``."""

    name: str

    def build(self, ctx: DecisionContext) -> pd.DataFrame:  # pragma: no cover - protocol
        ...
