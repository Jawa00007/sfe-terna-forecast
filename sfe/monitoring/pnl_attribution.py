"""Daily P&L attribution: realised vs predicted (PRD F-16).

Exact additive decomposition of the error between realised and expected P&L:

    pnl_realised  = Delta * S_real      - |Delta| * charge_real
    pnl_expected  = Delta * E[S_pred]   - |Delta| * charge_pred
    error         = pnl_realised - pnl_expected
                  = Delta * (S_real - E[S_pred])          # spread / magnitude error
                    - |Delta| * (charge_real - charge_pred)  # charge error

Sign error is surfaced separately as a value-weighted rate (weight |S_real|).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfe.curate.timebase import utc_to_rome

KEY = ["_area", "_mtu_start"]


def attribute(recs: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    r = recs.copy()
    r["_mtu_start"] = pd.to_datetime(r["_mtu_start"], utc=True)
    a = actual.copy()
    a["_mtu_start"] = pd.to_datetime(a["_mtu_start"], utc=True)
    m = r.merge(
        a[[*KEY, "spread_S", "charge_realized_eur_per_mwh"]], on=KEY, how="left"
    )

    d = pd.to_numeric(m["delta_mwh"], errors="coerce").fillna(0.0)
    s_real = pd.to_numeric(m["spread_S"], errors="coerce")
    s_pred = pd.to_numeric(m.get("e_spread"), errors="coerce")
    c_real = pd.to_numeric(m["charge_realized_eur_per_mwh"], errors="coerce").fillna(0.0)
    c_pred = pd.to_numeric(m.get("charge_eur_per_mwh"), errors="coerce").fillna(0.0)

    m["pnl_realized_eur"] = d * s_real - d.abs() * c_real
    m["pnl_expected_eur"] = d * s_pred - d.abs() * c_pred
    m["err_spread_eur"] = d * (s_real - s_pred)
    m["err_charge_eur"] = -d.abs() * (c_real - c_pred)
    m["sign_miss"] = (np.sign(d) != np.sign(s_real)) & (d != 0.0)
    m["abs_spread"] = s_real.abs()
    m["delivery_day"] = m["_mtu_start"].map(lambda t: utc_to_rome(t.to_pydatetime()).date())
    return m


def daily_attribution(attributed: pd.DataFrame) -> pd.DataFrame:
    scored = attributed.dropna(subset=["pnl_realized_eur"])
    if scored.empty:
        return pd.DataFrame()
    g = scored.groupby("delivery_day")
    out = g.agg(
        pnl_realized_eur=("pnl_realized_eur", "sum"),
        pnl_expected_eur=("pnl_expected_eur", "sum"),
        err_spread_eur=("err_spread_eur", "sum"),
        err_charge_eur=("err_charge_eur", "sum"),
        n_positions=("delta_mwh", lambda s: int((s != 0).sum())),
    )
    vw = g.apply(
        lambda d: np.average(d["sign_miss"], weights=d["abs_spread"])
        if d["abs_spread"].sum() > 0
        else np.nan,
        include_groups=False,
    )
    out["value_weighted_sign_miss"] = vw
    out["unexplained_eur"] = out["pnl_realized_eur"] - out["pnl_expected_eur"] - (
        out["err_spread_eur"] + out["err_charge_eur"]
    )
    return out.reset_index()
