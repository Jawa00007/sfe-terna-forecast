"""Market-impact / reflexivity model (PRD P-7, B-4).

At meaningful volume the portfolio's own deviation ``Delta`` moves the aggregate imbalance
it is trying to exploit, and therefore the spread. Model it linearly:

    S_effective(Delta) = E[S] - eta * Delta

``eta`` (EUR/MWh of spread move per MWh of own deviation) is estimated empirically from the
historical sensitivity of the spread to the aggregate imbalance volume, and falls back to
the configured constant when the fit is weak or wrong-signed. Without this term the sized
position is unbounded and the backtest overstates capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sfe.config import load_yaml


@dataclass(frozen=True)
class MarketImpact:
    eta_eur_per_mwh2: float
    source: str = "config"

    @classmethod
    def from_config(cls) -> MarketImpact:
        v = load_yaml("decision").get("sizing", {}).get("elasticity_eur_per_mwh2", 0.02)
        return cls(float(v), source="config")

    @classmethod
    def estimate(cls, targets: pd.DataFrame, *, min_rows: int = 500) -> MarketImpact:
        """OLS of spread_S on the signed aggregate imbalance volume, pooled across areas.

        Own ``Delta`` adds ~1:1 to the aggregate long position, so ``eta = -slope`` (a more
        area-long system depresses the imbalance price and lowers S).
        """
        cfg = cls.from_config()
        need = {"spread_S", "zonal_aggregate_unbalance_MWh"}
        if targets.empty or not need.issubset(targets.columns):
            return cfg
        d = targets[list(need)].apply(pd.to_numeric, errors="coerce").dropna()
        if len(d) < min_rows or d["zonal_aggregate_unbalance_MWh"].std() < 1e-6:
            return cfg
        slope = np.polyfit(d["zonal_aggregate_unbalance_MWh"], d["spread_S"], 1)[0]
        eta = -float(slope)
        if not np.isfinite(eta) or eta <= 0:
            return cfg
        return cls(eta, source="empirical")

    def effective_spread(self, expected_S, delta_mwh):
        return np.asarray(expected_S, dtype="float64") - self.eta_eur_per_mwh2 * np.asarray(
            delta_mwh, dtype="float64"
        )

    def unconstrained_optimum(self, expected_S_net) -> np.ndarray:
        """argmax_Delta  Delta*(E[S_net] - eta*Delta)  =>  Delta* = E[S_net] / (2*eta)."""
        return np.asarray(expected_S_net, dtype="float64") / (2.0 * self.eta_eur_per_mwh2)
