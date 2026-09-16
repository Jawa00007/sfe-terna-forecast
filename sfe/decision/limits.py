"""Risk limits on the recommended deviation (PRD F-11).

Per-MTU: hard cap as a fraction of forecast load, an absolute MW cap, and a CVaR floor on
the P&L distribution implied by the spread quantiles. Per-day: an aggregate |Delta| cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sfe.config import load_yaml


@dataclass(frozen=True)
class Limits:
    pct_load_cap: float
    per_mtu_mwh_cap: float | None
    daily_mwh_cap: float
    cvar_alpha: float
    cvar_limit_eur_per_mwh: float

    @classmethod
    def from_config(cls) -> Limits:
        limg = load_yaml("decision").get("limits", {})
        cap = limg.get("per_mtu_mwh_cap")
        return cls(
            pct_load_cap=float(limg.get("pct_load_cap", 0.15)),
            per_mtu_mwh_cap=(None if cap is None else float(cap)),
            daily_mwh_cap=float(limg.get("daily_mwh_cap", 600.0)),
            cvar_alpha=float(limg.get("cvar_alpha", 0.95)),
            cvar_limit_eur_per_mwh=float(limg.get("cvar_limit_eur_per_mwh", 25.0)),
        )

    def per_mtu_cap(self, load_mwh: float) -> float:
        caps = [abs(load_mwh) * self.pct_load_cap]
        if self.per_mtu_mwh_cap is not None:
            caps.append(self.per_mtu_mwh_cap)
        return float(min(caps))

    def clip(self, delta_mwh: float, load_mwh: float) -> tuple[float, str | None]:
        cap = self.per_mtu_cap(load_mwh)
        if abs(delta_mwh) <= cap:
            return delta_mwh, None
        return float(np.sign(delta_mwh) * cap), "per_mtu_cap"

    def cvar_per_mwh(
        self, quantiles: dict[float, float], charge: float, direction: int
    ) -> float:
        """CVaR of per-|Delta| P&L using the quantile points as equiprobable scenarios.

        pnl_per_unit = direction * S_scenario - charge. Returns the mean of the worst
        ``1 - alpha`` tail (with few scenarios this is the single worst point).
        """
        qs = np.array(sorted(quantiles))
        s = np.array([quantiles[q] for q in qs], dtype="float64")
        pnl = direction * s - charge
        pnl.sort()
        k = max(1, int(np.ceil((1 - self.cvar_alpha) * len(pnl))))
        return float(np.mean(pnl[:k]))

    def cvar_ok(self, quantiles: dict[float, float], charge: float, direction: int) -> bool:
        return self.cvar_per_mwh(quantiles, charge, direction) >= -self.cvar_limit_eur_per_mwh

    def apply_daily_cap(self, delta: pd.Series, group: pd.Series | None = None) -> pd.Series:
        """Scale each day's deviations down proportionally if the aggregate cap is exceeded."""
        out = delta.astype("float64").copy()
        if group is None:
            group = pd.Series(0, index=delta.index)
        for _, idx in group.groupby(group).groups.items():
            total = out.loc[idx].abs().sum()
            if total > self.daily_mwh_cap and total > 0:
                out.loc[idx] *= self.daily_mwh_cap / total
        return out
