"""Abstention rule (PRD F-13): refusing to trade is a valid, often correct, output."""

from __future__ import annotations

import math
from dataclasses import dataclass

from sfe.config import load_yaml


@dataclass(frozen=True)
class AbstainRule:
    min_edge_eur_per_mwh: float
    uncertainty_mult: float
    sign_prob_deadband: float

    @classmethod
    def from_config(cls) -> AbstainRule:
        a = load_yaml("decision").get("abstain", {})
        return cls(
            min_edge_eur_per_mwh=float(a.get("min_edge_eur_per_mwh", 1.5)),
            uncertainty_mult=float(a.get("uncertainty_mult", 0.35)),
            sign_prob_deadband=float(a.get("sign_prob_deadband", 0.05)),
        )

    def check(self, expected_net_edge: float, sigma_S: float, sign_prob: float) -> str | None:
        """Return an abstention reason, or ``None`` to proceed."""
        if math.isnan(expected_net_edge):
            return "no_forecast"
        if abs(expected_net_edge) < self.min_edge_eur_per_mwh:
            return "edge_below_min"
        if sigma_S == sigma_S and abs(expected_net_edge) < self.uncertainty_mult * sigma_S:
            return "wide_distribution"
        if sign_prob == sign_prob and abs(sign_prob - 0.5) < self.sign_prob_deadband:
            return "sign_indeterminate"
        return None
