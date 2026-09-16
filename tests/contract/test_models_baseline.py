"""Phase 2 baseline smoke: build a training frame from landed mock data and fit tiny models."""

import numpy as np
import pandas as pd
import pytest

from sfe.models.baselines import persistence, seasonal_climatology
from sfe.models.dataset import build_training_frame, time_split
from sfe.models.metrics import sign_report, spread_report
from sfe.models.price import derive_price
from sfe.models.sign import SignModel
from sfe.models.spread import QUANTILES, SpreadQuantileModel

pytestmark = pytest.mark.contract

_TINY = {"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 15, "verbose": -1, "n_jobs": 1}


def test_build_training_frame_shape(landed_six_weeks):
    tf = build_training_frame(*landed_six_weeks, horizon="D-1")
    assert len(tf) > 500
    assert "area_code" in tf.feature_names
    assert tf.X.select_dtypes(exclude="number").empty  # all-numeric matrix
    assert set(tf.y_sign_pos.dropna().unique()) <= {0.0, 1.0}
    assert tf.y_spread.notna().mean() > 0.5


def test_time_split_is_chronological(landed_six_weeks):
    tf = build_training_frame(*landed_six_weeks, horizon="D-1")
    tr, va, te = time_split(tf, valid_frac=0.2, test_frac=0.2)
    assert len(tr) > len(va) and len(te) > 0
    assert tr.meta["_mtu_start"].max() <= va.meta["_mtu_start"].min()
    assert va.meta["_mtu_start"].max() <= te.meta["_mtu_start"].min()


def test_sign_and_spread_models_fit_predict(landed_six_weeks):
    tf = build_training_frame(*landed_six_weeks, horizon="D-1")
    tr, va, te = time_split(tf)

    sign = SignModel(params={**SignModel().params, **_TINY}, calibration="isotonic").fit(tr, va)
    p = sign.predict_proba(te.X)
    assert p.shape == (len(te),)
    assert ((p >= 0) & (p <= 1)).all()
    rep = sign_report(te.y_sign_pos, p)
    assert 0.0 <= rep["brier"] <= 1.0

    spread = SpreadQuantileModel(
        quantiles=QUANTILES, params={**SpreadQuantileModel().params, **_TINY}
    ).fit(tr, va)
    q = spread.predict(te.X)
    stacked = np.column_stack([q[k] for k in QUANTILES])
    assert (np.diff(stacked, axis=1) >= -1e-9).all()  # non-crossing
    srep = spread_report(te.y_spread, q)
    assert np.isfinite(srep["mean_pinball"])

    price_q = derive_price((te.y_price - te.y_spread).to_numpy(), q)
    assert price_q[0.5].shape == (len(te),)


def test_baselines_produce_predictions(landed_six_weeks):
    tf = build_training_frame(*landed_six_weeks, horizon="D-1")
    tr, va, te = time_split(tf)
    ref_mask = pd.Series(~tf.X.index.isin(te.X.index), index=tf.X.index)
    per = persistence(tf.loc(ref_mask), te)
    clim = seasonal_climatology(tr, te)
    for b in (per, clim):
        assert b["p_sign_pos"].shape == (len(te),)
        assert np.isfinite(b["spread"]).mean() > 0.9
