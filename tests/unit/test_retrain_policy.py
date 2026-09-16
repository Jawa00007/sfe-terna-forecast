import pandas as pd

from sfe.models.registry import save_model
from sfe.models.retrain import model_age_days, should_retrain


def test_age_none_when_no_meta(tmp_settings):
    assert model_age_days("D-1") is None


def test_age_computed_from_train_end(tmp_settings):
    save_model({}, "sign_D-1", meta={"train_end": "2025-01-01T00:00:00+00:00"})
    now = pd.Timestamp("2025-01-15T00:00:00", tz="UTC")
    assert abs(model_age_days("D-1", now=now) - 14.0) < 1e-6


def test_should_retrain_missing_model(tmp_settings):
    d = should_retrain("D-1", max_age_days=9999)
    assert d.retrain is True
    assert "no registered model" in d.reasons


def test_should_retrain_forced(tmp_settings):
    save_model({}, "sign_D-1", meta={"train_end": "2025-01-01T00:00:00+00:00"})
    save_model({}, "spread_D-1", meta={"train_end": "2025-01-01T00:00:00+00:00"})
    d = should_retrain("D-1", force=True)
    assert d.retrain is True and d.reasons == ["forced"]


def test_should_retrain_age_trigger(tmp_settings):
    old = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)).isoformat()
    save_model({}, "sign_D-1", meta={"train_end": old})
    save_model({}, "spread_D-1", meta={"train_end": old})
    d = should_retrain("D-1", max_age_days=14)
    assert d.retrain is True
    assert any("model age" in r for r in d.reasons)
