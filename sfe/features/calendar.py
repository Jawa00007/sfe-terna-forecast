"""Calendar feature family (PRD F-5). Always point-in-time safe: known far in advance."""

from __future__ import annotations

import pandas as pd

from sfe.curate.calendar_it import calendar_features
from sfe.features.spec import DecisionContext, target_skeleton


class CalendarFeatures:
    name = "calendar"

    def build(self, ctx: DecisionContext) -> pd.DataFrame:
        skel = target_skeleton(ctx.delivery_day, ctx.areas, ctx.resolution)
        feats = calendar_features(skel["_mtu_start"])
        return skel.merge(feats, on="_mtu_start", how="left")
