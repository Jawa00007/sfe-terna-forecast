"""Retraining policy (PRD F-10): rolling-window retrain with drift detection.

Two independent triggers, either one is sufficient:
- **age**: the registered model is older than ``max_age_days`` (rolling-window retrain).
- **drift**: a large-enough fraction of features have moved (PSI) between a reference
  window and the most recent window (see :mod:`sfe.monitoring.drift`).

A missing model is always a trigger. ``force`` (e.g. after a regulatory change - PRD F-10's
"manual override") bypasses both checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from sfe.features.store import FeatureStore
from sfe.models.dataset import build_training_frame
from sfe.models.registry import load_model_meta, model_exists
from sfe.monitoring.drift import feature_drift

_MODEL_KINDS = ("sign", "spread")


@dataclass
class RetrainDecision:
    horizon: str
    retrain: bool
    reasons: list[str]
    model_age_days: float | None
    drift: pd.DataFrame = field(default_factory=pd.DataFrame)


def model_age_days(
    horizon: str, *, kind: str = "sign", now: pd.Timestamp | None = None
) -> float | None:
    meta = load_model_meta(f"{kind}_{horizon}")
    if not meta:
        return None
    ref = meta.get("train_end") or meta.get("saved_at")
    if not ref:
        return None
    trained = pd.Timestamp(ref)
    trained = trained.tz_localize("UTC") if trained.tzinfo is None else trained.tz_convert("UTC")
    now = now or pd.Timestamp.now(tz="UTC")
    return (now - trained).total_seconds() / 86400.0


def check_drift(
    horizon: str,
    *,
    reference_days: int = 60,
    current_days: int = 14,
    end: date | None = None,
    store: FeatureStore | None = None,
    storage=None,
) -> pd.DataFrame:
    """PSI per feature between a reference window and the most recent window."""
    end = end or (date.today() - timedelta(days=1))
    cur_start = end - timedelta(days=current_days - 1)
    ref_end = cur_start - timedelta(days=1)
    ref_start = ref_end - timedelta(days=reference_days - 1)

    ref = build_training_frame(ref_start, ref_end, horizon=horizon, store=store, storage=storage)
    cur = build_training_frame(cur_start, end, horizon=horizon, store=store, storage=storage)
    if len(ref) < 30 or len(cur) < 30:
        return pd.DataFrame()
    return feature_drift(ref.X, cur.X, ref.feature_names)


def should_retrain(
    horizon: str,
    *,
    max_age_days: float = 14.0,
    drift_alert_fraction: float = 0.2,
    reference_days: int = 60,
    current_days: int = 14,
    end: date | None = None,
    store: FeatureStore | None = None,
    storage=None,
    force: bool = False,
    now: pd.Timestamp | None = None,
) -> RetrainDecision:
    """*now* overrides the wall clock used for the age check (mainly for tests / replay)."""
    if force:
        return RetrainDecision(horizon, True, ["forced"], model_age_days(horizon, now=now))

    reasons: list[str] = []
    if not all(model_exists(f"{k}_{horizon}") for k in _MODEL_KINDS):
        reasons.append("no registered model")

    age = model_age_days(horizon, now=now)
    if age is not None and age > max_age_days:
        reasons.append(f"model age {age:.1f}d > max {max_age_days}d")

    drift = pd.DataFrame()
    if not reasons:  # skip the (expensive) drift check if we're retraining anyway
        drift = check_drift(
            horizon,
            reference_days=reference_days,
            current_days=current_days,
            end=end,
            store=store,
            storage=storage,
        )
        if not drift.empty:
            alert_frac = (drift["level"] == "alert").mean()
            if alert_frac > drift_alert_fraction:
                n = int((drift["level"] == "alert").sum())
                reasons.append(f"{n}/{len(drift)} features at PSI alert level")

    return RetrainDecision(horizon, bool(reasons), reasons, age, drift)
