"""Economic scoring of a set of recommendations (PRD §8 economic block, B-4/B-5).

Primary metric: net margin in EUR/MWh of portfolio volume versus a perfect-hedge
(Delta = 0) baseline, after all charges. Plus value-weighted sign hit rate, realised
Sharpe of the daily P&L, max drawdown, worst MTU / day, abstention fraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfe.curate.timebase import utc_to_rome

KEY = ["_area", "_mtu_start"]


def realize_pnl(
    recs: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    spread_col: str = "spread_S",
    charge_col: str = "charge_realized_eur_per_mwh",
    load_col: str = "load_realized_mwh",
) -> pd.DataFrame:
    """Join recommendations to realised outcomes and compute per-MTU P&L.

    ``pnl = Delta * S_realised - |Delta| * charge_realised``  (penalty already folded into
    the charge). The perfect-hedge baseline is 0, so ``pnl`` *is* the incremental margin.
    """
    r = recs.copy()
    r["_mtu_start"] = pd.to_datetime(r["_mtu_start"], utc=True)
    a = actual.copy()
    a["_mtu_start"] = pd.to_datetime(a["_mtu_start"], utc=True)

    m = r.merge(a[[*KEY, spread_col, charge_col, load_col]], on=KEY, how="left")
    s = pd.to_numeric(m[spread_col], errors="coerce")
    c = pd.to_numeric(m[charge_col], errors="coerce").fillna(0.0)
    d = pd.to_numeric(m["delta_mwh"], errors="coerce").fillna(0.0)

    m["pnl_eur"] = d * s - d.abs() * c
    m["pnl_eur"] = m["pnl_eur"].where(s.notna(), np.nan)  # can't score unknown outcomes
    m["abstained"] = m["reason"].isin(
        ["edge_below_min", "wide_distribution", "sign_indeterminate", "cvar_limit", "no_forecast"]
    ) | (d == 0.0)
    m["hit"] = np.sign(d) == np.sign(s)
    m["delivery_day"] = m["_mtu_start"].map(lambda t: utc_to_rome(t.to_pydatetime()).date())
    return m


def _max_drawdown(cum: np.ndarray) -> float:
    if cum.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(cum)
    return float(np.min(cum - peak))


def economic_report(pnl: pd.DataFrame, *, load_col: str = "load_realized_mwh") -> dict:
    scored = pnl.dropna(subset=["pnl_eur"]).copy()
    out: dict[str, float] = {
        "n_mtu": int(len(pnl)),
        "n_scored": int(len(scored)),
        "abstention_fraction": float(pnl["abstained"].mean()) if len(pnl) else float("nan"),
    }
    if scored.empty:
        return out

    vol = pd.to_numeric(scored[load_col], errors="coerce").abs().fillna(0.0)
    total_vol = float(vol.sum())
    out["net_margin_eur"] = float(scored["pnl_eur"].sum())
    out["net_margin_eur_per_mwh"] = (
        out["net_margin_eur"] / total_vol if total_vol > 0 else float("nan")
    )

    active = scored[~scored["abstained"]]
    if not active.empty:
        w = pd.to_numeric(active["spread_S"], errors="coerce").abs()
        out["value_weighted_hit_rate"] = float(
            np.average(active["hit"].astype(float), weights=w) if w.sum() > 0 else np.nan
        )
        out["n_positions"] = int(len(active))
        out["worst_mtu_eur"] = float(active["pnl_eur"].min())

    daily = scored.groupby("delivery_day")["pnl_eur"].sum().sort_index()
    out["worst_day_eur"] = float(daily.min()) if not daily.empty else float("nan")
    if daily.std(ddof=0) and daily.std(ddof=0) > 0:
        out["daily_sharpe_ann"] = float(
            daily.mean() / daily.std(ddof=0) * np.sqrt(252)
        )
    out["max_drawdown_eur"] = _max_drawdown(daily.cumsum().to_numpy())
    return out


def go_no_go(report: dict, *, min_margin_eur_per_mwh: float = 0.0) -> tuple[bool, str]:
    m = report.get("net_margin_eur_per_mwh", float("nan"))
    if not np.isfinite(m):
        return False, "no scored positions"
    if m <= min_margin_eur_per_mwh:
        return False, f"net margin {m:.3f} EUR/MWh <= threshold {min_margin_eur_per_mwh}"
    return True, f"net margin {m:.3f} EUR/MWh after charges"
