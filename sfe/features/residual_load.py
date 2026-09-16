"""Residual-load feature family (PRD F-5): load forecast net of variable renewables.

``residual_load = forecast_load - wind_forecast`` (solar forecast to be added once a source
is identified - the PRD feature list has wind but no explicit solar-forecast endpoint).

Point-in-time: uses the as-of view of the forecast datasets, so only forecast vintages
published before ``ctx.as_of`` are visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfe.curate.vintage import as_of_view, pivot_fields
from sfe.features.spec import DecisionContext, target_skeleton
from sfe.io.storage import StorageBackend, get_storage

LOAD_DATASET = "landing/terna/market/load-forecast"
WIND_DATASET = "landing/terna/generation/wind-production-forecast"


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


class ResidualLoadFeatures:
    name = "residual_load"

    def __init__(self, storage: StorageBackend | None = None):
        self.storage = storage or get_storage()

    def _asof_wide(self, dataset: str, as_of: object) -> pd.DataFrame:
        raw = self.storage.read_dataset(dataset)
        if raw.empty:
            return pd.DataFrame(columns=["_area", "_mtu_start"])
        return pivot_fields(as_of_view(raw, as_of))

    def build(self, ctx: DecisionContext) -> pd.DataFrame:
        out = target_skeleton(ctx.delivery_day, ctx.areas, ctx.resolution)
        load = self._asof_wide(LOAD_DATASET, ctx.as_of)
        wind = self._asof_wide(WIND_DATASET, ctx.as_of)

        out = out.merge(load, on=["_area", "_mtu_start"], how="left")
        out = out.merge(wind, on=["_area", "_mtu_start"], how="left")

        load_col = _col(out, "forecast_load_MW")
        wind_col = _col(out, "wind_forecast_MW")
        out["rl_load_forecast_MW"] = load_col
        out["rl_wind_forecast_MW"] = wind_col
        out["rl_residual_load_MW"] = load_col - wind_col
        out["rl_wind_penetration"] = wind_col / load_col
        drop = [c for c in ("forecast_load_MW", "wind_forecast_MW") if c in out]
        return out.drop(columns=drop)
