"""Lagged imbalance outcome features with realistic publication lags (PRD F-5, P-1).

At a decision timestamp ``ctx.as_of`` the most recent *published* imbalance outcome is
typically several days old (preliminary D+1..D+2, definitive later). This builder therefore
works only from :func:`assemble_targets` evaluated as-of ``ctx.as_of`` - it never sees a
value published at or after the decision - and also emits a *staleness* feature so the model
knows how old its most recent observation is.
"""

from __future__ import annotations

import pandas as pd

from sfe.curate.vintage import assert_no_leakage
from sfe.features.spec import DecisionContext, target_skeleton
from sfe.features.targets import assemble_targets

_DEFAULT_LAGS_DAYS = (1, 2, 7)
_DEFAULT_FIELDS = ("imb_sign", "spread_S", "unbalance_price_EURxMWh")


class LagTargetFeatures:
    name = "lag_target"

    def __init__(
        self,
        lags_days: tuple[int, ...] = _DEFAULT_LAGS_DAYS,
        fields: tuple[str, ...] = _DEFAULT_FIELDS,
    ):
        self.lags_days = lags_days
        self.fields = fields

    def build(self, ctx: DecisionContext) -> pd.DataFrame:
        skel = target_skeleton(ctx.delivery_day, ctx.areas, ctx.resolution)
        pit = assemble_targets(as_of=ctx.as_of)
        if not pit.empty:
            assert_no_leakage(pit, ctx.as_of)

        out = skel.copy()
        if pit.empty:
            for d in self.lags_days:
                for f in self.fields:
                    out[f"lag_target_{f}_d{d}"] = pd.NA
            out["lag_target_staleness_h"] = pd.NA
            for f in self.fields:
                out[f"lag_target_last_{f}"] = pd.NA
            return out

        pit = pit.assign(_mtu_start=pd.to_datetime(pit["_mtu_start"], utc=True))

        # exact same-time-of-day lag, d calendar days back
        for d in self.lags_days:
            shifted = pit.assign(_mtu_start=pit["_mtu_start"] + pd.Timedelta(days=d))
            cols = ["_area", "_mtu_start", *self.fields]
            ren = {f: f"lag_target_{f}_d{d}" for f in self.fields}
            shifted = shifted[cols].rename(columns=ren)
            out = out.merge(shifted, on=["_area", "_mtu_start"], how="left")

        # most recent published observation per area + how stale it is
        last = (
            pit.sort_values("_mtu_start")
            .groupby("_area", as_index=False)
            .tail(1)
            .set_index("_area")
        )
        out["lag_target_staleness_h"] = out["_area"].map(
            (ctx.as_of - last["_mtu_start"]).dt.total_seconds() / 3600.0
        )
        for f in self.fields:
            out[f"lag_target_last_{f}"] = out["_area"].map(last[f])
        return out
