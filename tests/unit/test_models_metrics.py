import numpy as np

from sfe.models.metrics import (
    auc_roc,
    brier_score,
    picp,
    pinball_loss,
    spread_report,
)


def test_brier_perfect_and_worst():
    y = np.array([1, 0, 1, 0], dtype=float)
    assert brier_score(y, y) == 0.0
    assert brier_score(y, 1 - y) == 1.0


def test_auc_perfect_separation():
    y = np.array([0, 0, 1, 1], dtype=float)
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc_roc(y, s) == 1.0
    assert auc_roc(y, 1 - s) == 0.0


def test_pinball_known_value():
    # single point: y=10, pred=8, q=0.9 -> 0.9 * 2 = 1.8
    assert abs(pinball_loss(np.array([10.0]), np.array([8.0]), 0.9) - 1.8) < 1e-9
    # over-prediction: y=8, pred=10, q=0.9 -> (0.9-1) * (-2) = 0.2
    assert abs(pinball_loss(np.array([8.0]), np.array([10.0]), 0.9) - 0.2) < 1e-9


def test_picp_coverage():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert picp(y, np.array([0, 0, 0, 0]), np.array([5, 5, 5, 5])) == 1.0
    assert picp(y, np.array([2, 2, 2, 2]), np.array([3, 3, 3, 3])) == 0.5


def test_spread_report_keys():
    y = np.random.default_rng(0).normal(size=200)
    preds = {0.1: y - 1.2, 0.25: y - 0.6, 0.5: y, 0.75: y + 0.6, 0.9: y + 1.2}
    rep = spread_report(y, preds)
    assert {"pinball_p10", "pinball_p50", "mae_median", "picp_10_90", "mean_pinball"} <= set(rep)
    assert rep["mae_median"] < 1e-9  # median == y here
