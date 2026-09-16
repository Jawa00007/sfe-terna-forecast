"""Train + evaluate the Phase 2 baselines (PRD F-6, F-7, §8, B-1/B-6).

Chronological split, no shuffling. Reports statistical metrics for the sign and spread
models against persistence and seasonal climatology, and whether the models beat them.

    python -m sfe.scripts.train_baseline --from 2024-09-01 --to 2025-02-28
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import date

from sfe.models.baselines import persistence, seasonal_climatology
from sfe.models.dataset import build_training_frame, time_split
from sfe.models.metrics import sign_report, spread_report
from sfe.models.price import derive_price
from sfe.models.registry import log_run, save_model
from sfe.models.sign import SignModel
from sfe.models.spread import QUANTILES, SpreadQuantileModel


def _fmt(d: dict) -> str:
    return ", ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()
    )


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sfe-train-baseline", description=__doc__)
    p.add_argument("--from", dest="date_from", required=True, type=date.fromisoformat)
    p.add_argument("--to", dest="date_to", required=True, type=date.fromisoformat)
    p.add_argument("--horizon", default="D-1")
    p.add_argument("--valid-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--calibration", default="isotonic", choices=["isotonic", "platt", "none"])
    p.add_argument("--save", action="store_true", help="persist fitted models under data/models/")
    args = p.parse_args(argv)

    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(encoding="utf-8")

    tf = build_training_frame(args.date_from, args.date_to, horizon=args.horizon)
    if len(tf) == 0:
        print("no training rows - land some fees data and build features first")
        return 1
    train, valid, test = time_split(tf, valid_frac=args.valid_frac, test_frac=args.test_frac)

    sign = SignModel(calibration=args.calibration).fit(train, valid)
    spread = SpreadQuantileModel(quantiles=QUANTILES).fit(train, valid)

    # --- sign ---
    p_model = sign.predict_proba(test.X)
    m_sign = sign_report(test.y_sign_pos, p_model)
    b_persist = persistence(tf.loc(_train_valid_mask(tf, test)), test)
    b_clim = seasonal_climatology(train, test)
    m_persist = sign_report(test.y_sign_pos, b_persist["p_sign_pos"])
    m_clim = sign_report(test.y_sign_pos, b_clim["p_sign_pos"])

    # --- spread ---
    q_model = spread.predict(test.X)
    m_spread = spread_report(test.y_spread, q_model)
    # persistence as a degenerate fan (all quantiles = point forecast)
    q_persist = {q: b_persist["spread"] for q in QUANTILES}
    m_spread_persist = spread_report(test.y_spread, q_persist)

    # --- price (derived) ---
    p_mkt_realized = (test.y_price - test.y_spread).to_numpy()
    price_q = derive_price(p_mkt_realized, q_model)
    m_price = spread_report(test.y_price, price_q)

    beats_sign = (m_sign["brier"] < m_persist["brier"]) and (m_sign["brier"] < m_clim["brier"])
    beats_spread = m_spread["mean_pinball"] < m_spread_persist["mean_pinball"]

    lines = [
        "# Baseline training report",
        "",
        f"rows: total={len(tf)} train={len(train)} valid={len(valid)} test={len(test)}",
        f"features={len(tf.feature_names)} horizon={args.horizon} calib={args.calibration}",
        "",
        "## Sign  (P(area long))",
        f"- model:        {_fmt(m_sign)}",
        f"- persistence:  {_fmt(m_persist)}",
        f"- climatology:  {_fmt(m_clim)}",
        f"- **beats both baselines on Brier: {beats_sign}**",
        "",
        "## Spread S  (pinball across quantiles)",
        f"- model:        {_fmt(m_spread)}",
        f"- persistence:  {_fmt(m_spread_persist)}",
        f"- **beats persistence on mean pinball: {beats_spread}**",
        "",
        "## Imbalance price  (derived: P_mkt_realized + S_forecast)",
        f"- {_fmt(m_price)}",
        "",
        "## Top sign-model features",
        *(f"- {n}: {v:.0f}" for n, v in sign.feature_importance().head(12).items()),
    ]
    report = "\n".join(lines)
    print(report)

    log_run(
        "baseline",
        params={
            "date_from": str(args.date_from),
            "date_to": str(args.date_to),
            "horizon": args.horizon,
            "calibration": args.calibration,
            "n_features": len(tf.feature_names),
        },
        metrics={
            "sign_brier": m_sign["brier"],
            "sign_auc": m_sign["auc"],
            "sign_ece": m_sign["ece"],
            "spread_mean_pinball": m_spread["mean_pinball"],
            "spread_mae_median": m_spread.get("mae_median", float("nan")),
            "persist_sign_brier": m_persist["brier"],
            "persist_spread_mean_pinball": m_spread_persist["mean_pinball"],
        },
        tags={"beats_sign": str(beats_sign), "beats_spread": str(beats_spread)},
    )

    if args.save:
        save_model(sign, f"sign_{args.horizon}")
        save_model(spread, f"spread_{args.horizon}")

    return 0 if (beats_sign and beats_spread) else 2


def _train_valid_mask(tf, test):
    import pandas as pd

    test_keys = set(zip(test.meta["_area"], test.meta["_mtu_start"], strict=True))
    keys = list(zip(tf.meta["_area"], tf.meta["_mtu_start"], strict=True))
    return pd.Series([k not in test_keys for k in keys], index=tf.X.index)


def main(argv: list[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
