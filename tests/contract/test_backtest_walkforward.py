"""End-to-end economic backtest on landed mock data (PRD §9, B-1..B-6)."""

import numpy as np
import pandas as pd
import pytest

from sfe.backtest.economic import go_no_go
from sfe.backtest.sensitivity import charge_sensitivity
from sfe.backtest.walkforward import make_folds, run_walkforward
from sfe.decision.policy import recommend_frame

pytestmark = pytest.mark.contract

_TINY = {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 15, "verbose": -1, "n_jobs": 1}


def test_make_folds_are_consecutive_and_expanding():
    from datetime import date, timedelta

    folds = make_folds(date(2024, 1, 1), date(2024, 4, 30), min_train_days=60, test_days=30)
    assert folds[0].train_start == date(2024, 1, 1)
    assert folds[0].test_start == date(2024, 1, 1) + timedelta(days=60)
    for a, b in zip(folds, folds[1:], strict=False):
        assert b.test_start == a.test_end + timedelta(days=1)   # consecutive, no gap/overlap
        assert b.train_start == a.train_start                   # expanding window
        assert b.train_end > a.train_end
    assert folds[-1].test_end <= date(2024, 4, 30)


def test_recommend_frame_shapes(landed_two_weeks):
    from datetime import date

    from sfe.features.spec import DecisionContext, as_of_for_horizon
    from sfe.features.store import default_store
    from sfe.models.dataset import build_training_frame, feature_matrix, time_split
    from sfe.models.sign import SignModel
    from sfe.models.spread import SpreadQuantileModel

    tf = build_training_frame(*landed_two_weeks, horizon="D-1")
    tr, va, _ = time_split(tf, valid_frac=0.2, test_frac=0.0)
    sign = SignModel(params={**SignModel().params, **_TINY}).fit(tr, va)
    spread = SpreadQuantileModel(params={**SpreadQuantileModel().params, **_TINY}).fit(tr, va)

    store = default_store()
    dd = date(2025, 1, 16)
    ctx = DecisionContext(as_of=as_of_for_horizon(dd), delivery_day=dd)
    feats = store.assemble(ctx)
    X = feature_matrix(feats, tf.feature_names)
    frame = feats[["_area", "_mtu_start"]].copy()
    frame["sign_prob"] = sign.predict_proba(X)
    for q, arr in spread.predict(X).items():
        frame[f"S_p{int(q * 100)}"] = arr

    rec = recommend_frame(frame, as_of=ctx.as_of)
    assert {"delta_mwh", "abstain", "reason", "expected_net_margin_eur", "cvar_eur_per_mwh"} <= set(
        rec.columns
    )
    assert len(rec) == len(frame)
    assert rec["abstain"].dtype == bool
    # daily aggregate cap respected
    day = rec["_mtu_start"].map(lambda t: pd.Timestamp(t).date())
    assert rec.groupby(day)["delta_mwh"].apply(lambda s: s.abs().sum()).max() <= 600.0 + 1e-6


def test_walkforward_runs_and_scores(landed_six_weeks):
    wf = run_walkforward(
        *landed_six_weeks,
        horizon="D-1",
        sign_params=_TINY,
        spread_params=_TINY,
        min_train_days=25,
        test_days=10,
    )
    assert not wf["recs"].empty
    assert "pnl_eur" in wf["pnl"].columns
    rep = wf["report"]
    assert 0.0 <= rep["abstention_fraction"] <= 1.0
    assert rep["n_scored"] > 0
    assert np.isfinite(rep["net_margin_eur_per_mwh"])
    ok, msg = go_no_go(rep)
    assert isinstance(ok, bool) and isinstance(msg, str)

    sens = charge_sensitivity(wf["recs"])
    assert list(sens["charge_multiplier"]) == [0.5, 1.0, 1.5]
    # higher charge never improves net margin
    m = sens.set_index("charge_multiplier")["net_margin_eur_per_mwh"]
    assert m[1.5] <= m[0.5] + 1e-6
