"""Structural-break / regime markers (PRD DQ-4, A-1, A-4, A-5).

Resolved from ``conf/regimes.yaml``. Exposed to models as categorical features and used as
hard split boundaries in the walk-forward backtest. Values are placeholders until A-1..A-5
are confirmed in Phase 0 - ``unconfirmed_regimes()`` lists what still needs sign-off.
"""

from __future__ import annotations

import bisect
from datetime import date, datetime
from functools import lru_cache

from sfe.config import load_yaml


@lru_cache(maxsize=1)
def _cfg() -> dict:
    return load_yaml("regimes")


def _pick(timeline_key: str, when: date) -> dict:
    entries = sorted(_cfg().get(timeline_key, []), key=lambda e: e["from"])
    if not entries:
        return {}
    starts = [date.fromisoformat(e["from"]) for e in entries]
    idx = bisect.bisect_right(starts, when) - 1
    return entries[max(idx, 0)]


def _as_date(when: date | datetime) -> date:
    return when.date() if isinstance(when, datetime) else when


def mtu_resolution(when: date | datetime) -> str:
    return _pick("mtu", _as_date(when)).get("resolution", "PT1H")


def terna_data_type(when: date | datetime) -> str:
    return _pick("mtu", _as_date(when)).get("terna_dataType", "Orario")


def pricing_rule(when: date | datetime) -> str:
    return _pick("pricing_rule", _as_date(when)).get("rule", "single_price")


def tide_phase(when: date | datetime) -> str:
    return _pick("tide_phase", _as_date(when)).get("phase", "unknown")


def zone_config(when: date | datetime) -> list[str]:
    return _pick("zone_config", _as_date(when)).get("zones", [])


def regime_features(when: date | datetime) -> dict[str, str]:
    """Flat dict of regime markers for feature assembly."""
    return {
        "regime_mtu_resolution": mtu_resolution(when),
        "regime_pricing_rule": pricing_rule(when),
        "regime_tide_phase": tide_phase(when),
    }


def unconfirmed_regimes() -> list[str]:
    out: list[str] = []
    for key, entries in _cfg().items():
        for e in entries:
            if not e.get("confirmed", False):
                out.append(f"{key}:{e.get('from')}")
    return out
