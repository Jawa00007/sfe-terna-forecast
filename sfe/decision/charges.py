"""Regulatory charges for the net-margin calculation (PRD R-1, F-12).

The decision layer subtracts the macrozonal non-arbitrage charge and expected penalty
components from the gross spread before sizing a position. Gross spread is never used as
P&L. Authoritative source is Terna's ``macrozonal-no-arbitrage-prices`` endpoint; the YAML
fallback + sensitivity multipliers are for periods with no published value and for the
backtest sensitivity band (B-5).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sfe.config import load_yaml
from sfe.curate.vintage import as_of_view, latest_view, pivot_fields
from sfe.io.storage import StorageBackend, get_storage

_NO_ARB_DATASET = "landing/terna/fees/macrozonal-no-arbitrage-prices"
_NO_ARB_FIELD_CANDIDATES = ("no_arbitrage_price_EURxMWh", "no_arb_price_EURxMWh", "value")


@dataclass(frozen=True)
class ChargeModel:
    fallback_eur_per_mwh: float
    penalty_eur_per_mwh: float
    multiplier: float = 1.0

    @classmethod
    def from_config(cls, *, multiplier: float = 1.0) -> ChargeModel:
        c = load_yaml("charges")
        na = c.get("non_arbitrage_charge", {})
        pen = c.get("dispatch_order_penalty", {})
        return cls(
            fallback_eur_per_mwh=float(na.get("fallback_eur_per_mwh", 0.0)),
            penalty_eur_per_mwh=(
                float(pen.get("expected_eur_per_mwh", 0.0)) if pen.get("enabled") else 0.0
            ),
            multiplier=multiplier,
        )

    def charge_per_mwh(
        self,
        keys: pd.DataFrame,
        *,
        as_of: object | None = None,
        storage: StorageBackend | None = None,
    ) -> pd.Series:
        """Non-arbitrage charge + expected penalty per MWh of deviation, indexed like *keys*.

        *keys* must have ``_area`` and ``_mtu_start``. Uses the as-of view when *as_of* is
        given (point-in-time), else the latest vintage.
        """
        storage = storage or get_storage()
        raw = storage.read_dataset(_NO_ARB_DATASET)
        charge = pd.Series(self.fallback_eur_per_mwh, index=keys.index, dtype="float64")

        if not raw.empty:
            view = as_of_view(raw, as_of) if as_of is not None else latest_view(raw)
            if not view.empty:
                wide = pivot_fields(view)
                field = next(
                    (f for f in _NO_ARB_FIELD_CANDIDATES if f in wide.columns), None
                )
                if field is not None:
                    wide["_mtu_start"] = pd.to_datetime(wide["_mtu_start"], utc=True)
                    lut = wide.set_index(["_area", "_mtu_start"])[field]
                    k = keys.copy()
                    k["_mtu_start"] = pd.to_datetime(k["_mtu_start"], utc=True)
                    mapped = lut.reindex(list(zip(k["_area"], k["_mtu_start"], strict=True)))
                    mapped.index = keys.index
                    charge = pd.to_numeric(mapped, errors="coerce").fillna(
                        self.fallback_eur_per_mwh
                    )

        return charge.abs() * self.multiplier + self.penalty_eur_per_mwh
