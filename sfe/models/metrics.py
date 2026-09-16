"""Scoring for the baseline models (PRD §8 statistical block).

Sign: balanced accuracy, AUC, Brier, calibration error (ECE).
Spread: MAE, RMSE, pinball loss per quantile, PICP of the central interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(y_true, y_score):
    y_true = np.asarray(y_true, dtype="float64")
    y_score = np.asarray(y_score, dtype="float64")
    m = np.isfinite(y_true) & np.isfinite(y_score)
    return y_true[m], y_score[m]


# --- sign -----------------------------------------------------------------------
def brier_score(y_true_pos: np.ndarray, p_pos: np.ndarray) -> float:
    yt, p = _clean(y_true_pos, p_pos)
    return float(np.mean((p - yt) ** 2)) if yt.size else float("nan")


def balanced_accuracy(y_true_pos: np.ndarray, p_pos: np.ndarray, thr: float = 0.5) -> float:
    yt, p = _clean(y_true_pos, p_pos)
    if yt.size == 0:
        return float("nan")
    pred = (p >= thr).astype("float64")
    tpr = np.mean(pred[yt == 1] == 1) if np.any(yt == 1) else 0.0
    tnr = np.mean(pred[yt == 0] == 0) if np.any(yt == 0) else 0.0
    return float((tpr + tnr) / 2)


def auc_roc(y_true_pos: np.ndarray, p_pos: np.ndarray) -> float:
    yt, p = _clean(y_true_pos, p_pos)
    n_pos, n_neg = int(np.sum(yt == 1)), int(np.sum(yt == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype="float64")
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    sum_pos = np.sum(ranks[yt == 1])
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(y_true_pos: np.ndarray, p_pos: np.ndarray, bins: int = 10) -> float:
    yt, p = _clean(y_true_pos, p_pos)
    if yt.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if not np.any(sel):
            continue
        ece += np.mean(sel) * abs(np.mean(p[sel]) - np.mean(yt[sel]))
    return float(ece)


def sign_report(y_true_pos: np.ndarray, p_pos: np.ndarray) -> dict[str, float]:
    return {
        "n": int(np.sum(np.isfinite(np.asarray(y_true_pos, dtype="float64")))),
        "brier": brier_score(y_true_pos, p_pos),
        "balanced_accuracy": balanced_accuracy(y_true_pos, p_pos),
        "auc": auc_roc(y_true_pos, p_pos),
        "ece": expected_calibration_error(y_true_pos, p_pos),
        "base_rate_pos": float(np.nanmean(np.asarray(y_true_pos, dtype="float64"))),
    }


def reliability_table(y_true_pos, p_pos, bins: int = 10) -> pd.DataFrame:
    yt, p = _clean(y_true_pos, p_pos)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        rows.append(
            {
                "bin": b,
                "p_lo": edges[b],
                "p_hi": edges[b + 1],
                "n": int(np.sum(sel)),
                "mean_pred": float(np.mean(p[sel])) if np.any(sel) else float("nan"),
                "frac_pos": float(np.mean(yt[sel])) if np.any(sel) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# --- spread / quantiles -------------------------------------------------------
def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    yt, yp = _clean(y_true, y_pred_q)
    if yt.size == 0:
        return float("nan")
    d = yt - yp
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def mae(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp))) if yt.size else float("nan")


def rmse(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2))) if yt.size else float("nan")


def picp(y_true, lower, upper) -> float:
    """Prediction-interval coverage probability."""
    yt = np.asarray(y_true, dtype="float64")
    lo = np.asarray(lower, dtype="float64")
    hi = np.asarray(upper, dtype="float64")
    m = np.isfinite(yt) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(m):
        return float("nan")
    return float(np.mean((yt[m] >= lo[m]) & (yt[m] <= hi[m])))


def spread_report(
    y_true: np.ndarray, preds_by_q: dict[float, np.ndarray]
) -> dict[str, float]:
    qs = sorted(preds_by_q)
    out: dict[str, float] = {"n": int(np.sum(np.isfinite(np.asarray(y_true, dtype="float64"))))}
    for q in qs:
        out[f"pinball_p{int(q * 100)}"] = pinball_loss(y_true, preds_by_q[q], q)
    if 0.5 in preds_by_q:
        out["mae_median"] = mae(y_true, preds_by_q[0.5])
        out["rmse_median"] = rmse(y_true, preds_by_q[0.5])
    lo, hi = qs[0], qs[-1]
    out[f"picp_{int(lo * 100)}_{int(hi * 100)}"] = picp(y_true, preds_by_q[lo], preds_by_q[hi])
    out["mean_pinball"] = float(
        np.nanmean([out[f"pinball_p{int(q * 100)}"] for q in qs])
    )
    return out
