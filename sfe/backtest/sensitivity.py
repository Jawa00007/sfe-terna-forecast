"""Sensitivity of the economic result to the non-arbitrage charge assumption (PRD B-5)."""

from __future__ import annotations

import pandas as pd

from sfe.backtest.economic import economic_report, go_no_go, realize_pnl
from sfe.backtest.walkforward import build_actuals
from sfe.config import load_yaml


def charge_sensitivity(recs: pd.DataFrame, *, storage=None) -> pd.DataFrame:
    """Re-score a fixed set of recommendations under each charge multiplier."""
    mults = (
        load_yaml("charges")
        .get("non_arbitrage_charge", {})
        .get("sensitivity_multipliers", [0.5, 1.0, 1.5])
    )
    keys = recs[["_area", "_mtu_start"]].drop_duplicates()
    rows = []
    for mlt in mults:
        actual = build_actuals(keys, storage=storage, charge_multiplier=float(mlt))
        rep = economic_report(realize_pnl(recs, actual))
        ok, _ = go_no_go(rep)
        rows.append(
            {
                "charge_multiplier": float(mlt),
                "net_margin_eur_per_mwh": rep.get("net_margin_eur_per_mwh"),
                "net_margin_eur": rep.get("net_margin_eur"),
                "value_weighted_hit_rate": rep.get("value_weighted_hit_rate"),
                "go": ok,
            }
        )
    return pd.DataFrame(rows)
