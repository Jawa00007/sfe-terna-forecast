"""Turn a spread distribution into a target deviation (PRD F-11).

Closed form: maximise  Delta * (E[S] - eta*Delta) - |Delta| * charge
=> interior optimum  Delta* = (E[S] - dir*charge) / (2*eta),  dir = sign(E[S]).
If the charge flips the sign of the net edge, the optimum is 0 (the charge ate the trade).
The result here is unclipped; :mod:`sfe.decision.limits` applies the caps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sfe.config import load_yaml
from sfe.decision.market_impact import MarketImpact

# p90 - p10 of a normal covers 2.5631 sigma.
_FAN_TO_SIGMA = 2.5631


@dataclass(frozen=True)
class Sizer:
    aggressiveness: float
    rounding_mwh: float
    impact: MarketImpact

    @classmethod
    def from_config(cls, impact: MarketImpact | None = None) -> Sizer:
        s = load_yaml("decision").get("sizing", {})
        return cls(
            aggressiveness=float(s.get("aggressiveness", 0.5)),
            rounding_mwh=float(s.get("rounding_mwh", 1.0)),
            impact=impact or MarketImpact.from_config(),
        )

    @staticmethod
    def expected_spread(quantiles: dict[float, float]) -> float:
        qs = sorted(quantiles)
        return float(np.mean([quantiles[q] for q in qs]))

    @staticmethod
    def sigma_spread(quantiles: dict[float, float]) -> float:
        lo, hi = min(quantiles), max(quantiles)
        return float(abs(quantiles[hi] - quantiles[lo]) / _FAN_TO_SIGMA)

    def net_edge(self, e_spread: float, charge: float) -> float:
        """E[S] net of the charge in the profitable direction; 0 if the charge kills it."""
        direction = int(np.sign(e_spread)) or 1
        net = e_spread - direction * charge
        return net if np.sign(net) == direction else 0.0

    def target_delta(self, quantiles: dict[float, float], charge: float) -> float:
        e_spread = self.expected_spread(quantiles)
        edge = self.net_edge(e_spread, charge)
        if edge == 0.0:
            return 0.0
        raw = self.aggressiveness * self.impact.unconstrained_optimum(edge)
        step = self.rounding_mwh
        return float(np.round(raw / step) * step) if step > 0 else float(raw)
