"""Retrain policy against landed mock data (PRD F-10)."""

from datetime import date

import pandas as pd
import pytest

from sfe.models.registry import model_exists
from sfe.models.retrain import check_drift, should_retrain
from sfe.serve.inference import ensure_models

pytestmark = pytest.mark.contract

_TINY = {"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 15, "verbose": -1, "n_jobs": 1}
_END = date(2025, 1, 14)


def test_check_drift_shape(landed_six_weeks):
    drift = check_drift("D-1", reference_days=20, current_days=10, end=_END)
    assert not drift.empty
    assert {"feature", "psi", "level"} <= set(drift.columns)
    assert set(drift["level"]) <= {"ok", "warn", "alert"}


def test_should_retrain_end_to_end(landed_six_weeks):
    # nothing registered yet -> must retrain
    before = should_retrain("D-1", end=_END, reference_days=20, current_days=10)
    assert before.retrain is True
    assert "no registered model" in before.reasons

    ensure_models("D-1", end=_END, lookback_days=40, force_train=True, params=_TINY)
    assert model_exists("sign_D-1") and model_exists("spread_D-1")

    # "now" pinned right after training: freshly trained + generous thresholds -> no retrain
    just_after = pd.Timestamp(_END, tz="UTC") + pd.Timedelta(hours=6)
    after = should_retrain(
        "D-1", end=_END, max_age_days=9999, drift_alert_fraction=0.95,
        reference_days=20, current_days=10, now=just_after,
    )
    assert after.retrain is False
    assert after.model_age_days is not None and after.model_age_days < 1.0

    # tiny age budget flips it back on
    stale = should_retrain("D-1", end=_END, max_age_days=0.0, now=just_after)
    assert stale.retrain is True
    assert any("model age" in r for r in stale.reasons)
