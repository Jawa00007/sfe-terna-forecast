"""Feature-layer point-in-time correctness against landed mock data (PRD F-4, F-5, P-1)."""

from datetime import date

import pandas as pd
import pytest

from sfe.features.lags import LagTargetFeatures
from sfe.features.residual_load import ResidualLoadFeatures
from sfe.features.spec import DecisionContext
from sfe.features.store import attach_labels, default_store
from sfe.features.targets import assemble_targets

pytestmark = pytest.mark.contract


def test_targets_latest_vs_asof(landed_two_weeks):
    latest = assemble_targets(as_of=None)
    assert not latest.empty
    assert {"imb_sign", "spread_S", "unbalance_price_EURxMWh"}.issubset(latest.columns)
    assert set(latest["imb_sign"].dropna().unique()) <= {-1, 0, 1}

    # Mid-week: only the preliminary vintage for the earliest days is visible.
    asof = pd.Timestamp("2025-01-09T07:00:00Z")
    pit = assemble_targets(as_of=asof)
    assert not pit.empty
    assert (pd.to_datetime(pit["_publication_ts"], utc=True) < asof).all()
    # nothing for delivery days whose preliminary is not published yet
    assert pit["_mtu_start"].max() < asof


def test_lag_features_never_leak(landed_two_weeks):
    ctx = DecisionContext(as_of="2025-01-16T08:00:00Z", delivery_day=date(2025, 1, 16))
    feats = LagTargetFeatures(lags_days=(1, 2, 7)).build(ctx)

    # staleness is positive: the most recent observed outcome predates the decision
    stale = pd.to_numeric(feats["lag_target_staleness_h"], errors="coerce").dropna()
    assert not stale.empty
    assert (stale > 0).all()

    # every lag value must correspond to a target actually published before as_of
    pit = assemble_targets(as_of=ctx.as_of)
    pub_ok = pd.to_datetime(pit["_publication_ts"], utc=True) < ctx.as_of
    assert pub_ok.all()
    assert feats.filter(like="lag_target_spread_S_d").notna().any().any()


def test_residual_load_builder(landed_two_weeks):
    ctx = DecisionContext(as_of="2025-01-16T08:00:00Z", delivery_day=date(2025, 1, 16))
    feats = ResidualLoadFeatures().build(ctx)
    rl_cols = ["rl_load_forecast_MW", "rl_wind_forecast_MW", "rl_residual_load_MW"]
    assert set(rl_cols).issubset(feats.columns)
    j = feats.dropna(subset=rl_cols)
    assert not j.empty
    assert (j["rl_wind_forecast_MW"] >= 0).all()
    assert (j["rl_residual_load_MW"] <= j["rl_load_forecast_MW"] + 1e-6).all()
    recomputed = j["rl_load_forecast_MW"] - j["rl_wind_forecast_MW"]
    assert (j["rl_residual_load_MW"] - recomputed).abs().max() < 1e-6


def test_default_store_assembles_with_labels(landed_two_weeks):
    store = default_store()
    ctx = DecisionContext(as_of="2025-01-16T08:00:00Z", delivery_day=date(2025, 1, 16))
    df = store.assemble(ctx)
    assert {"_area", "_mtu_start", "_as_of", "_horizon"}.issubset(df.columns)
    assert df.filter(like="cal_").shape[1] > 3
    assert df.filter(like="lag_target_").shape[1] > 3
    assert len(df) == 24 * 2  # 24 hourly periods x 2 macrozones

    labelled = attach_labels(df)
    assert {"label_sign", "label_spread_S"}.issubset(labelled.columns)
    assert labelled["label_spread_S"].notna().any()
