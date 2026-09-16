"""Point-in-time feature store (PRD F-4).

Assembles registered feature builders into one frame keyed ``(_area, _mtu_start, _as_of)``
for a decision context, and materialises it to Parquet. Each builder is responsible for its
own point-in-time correctness; :func:`attach_labels` is the only place the *latest* (non
as-of) target view is used, and it is applied after feature assembly, for training only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sfe.config import get_settings
from sfe.features.spec import KEY, DecisionContext, FeatureBuilder, target_skeleton
from sfe.features.targets import assemble_targets


class FeatureStore:
    def __init__(self, builders: Sequence[FeatureBuilder] = ()):
        self.builders: list[FeatureBuilder] = list(builders)

    def register(self, builder: FeatureBuilder) -> FeatureStore:
        self.builders.append(builder)
        return self

    def assemble(self, ctx: DecisionContext) -> pd.DataFrame:
        base = target_skeleton(ctx.delivery_day, ctx.areas, ctx.resolution)
        base["_as_of"] = ctx.as_of
        base["_horizon"] = ctx.horizon

        for b in self.builders:
            f = b.build(ctx)
            missing = [k for k in KEY if k not in f.columns]
            if missing:
                raise ValueError(f"builder {b.name!r} returned frame missing {missing}")
            dup = f.duplicated(subset=KEY).sum()
            if dup:
                raise ValueError(f"builder {b.name!r} produced {dup} duplicate {KEY} rows")
            overlap = (set(f.columns) - set(KEY)) & set(base.columns)
            if overlap:
                raise ValueError(f"builder {b.name!r} column name clash: {sorted(overlap)}")
            base = base.merge(f, on=KEY, how="left")
        return base

    def assemble_many(self, contexts: Iterable[DecisionContext]) -> pd.DataFrame:
        frames = [self.assemble(c) for c in contexts]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def materialize(
        self, contexts: Iterable[DecisionContext], *, dataset: str = "pit"
    ) -> dict:
        df = self.assemble_many(contexts)
        out_dir = get_settings().storage.features_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = out_dir / f"part-{stamp}.parquet"
        df.to_parquet(path, index=False)
        return {"rows": len(df), "cols": df.shape[1], "path": str(path)}


def attach_labels(features: pd.DataFrame, *, storage=None) -> pd.DataFrame:
    """Left-join the latest-known targets as ``label_*`` columns (training only)."""
    tgt = assemble_targets(as_of=None, storage=storage)
    if tgt.empty:
        for c in ("label_sign", "label_spread_S", "label_unbalance_price_EURxMWh"):
            features[c] = pd.NA
        return features
    lab = tgt.rename(
        columns={
            "imb_sign": "label_sign",
            "spread_S": "label_spread_S",
            "unbalance_price_EURxMWh": "label_unbalance_price_EURxMWh",
        }
    )[["_area", "_mtu_start", "label_sign", "label_spread_S", "label_unbalance_price_EURxMWh"]]
    lab["_mtu_start"] = pd.to_datetime(lab["_mtu_start"], utc=True)
    features = features.copy()
    features["_mtu_start"] = pd.to_datetime(features["_mtu_start"], utc=True)
    return features.merge(lab, on=["_area", "_mtu_start"], how="left")


def read_features(dataset: str = "pit") -> pd.DataFrame:
    d = get_settings().storage.features_dir / dataset
    parts = sorted(Path(d).glob("part-*.parquet")) if Path(d).exists() else []
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def default_store() -> FeatureStore:
    """The baseline feature set (PRD F-5). Extended as families are implemented."""
    from sfe.features.calendar import CalendarFeatures
    from sfe.features.lags import LagTargetFeatures
    from sfe.features.residual_load import ResidualLoadFeatures

    return FeatureStore([CalendarFeatures(), LagTargetFeatures(), ResidualLoadFeatures()])
