"""Synthetic Terna payloads for offline development (no sandbox exists - PRD 12).

Deliberately reproduces the awkward bits of the real feed so the parsing/curate layers are
exercised: numbers as strings with a decimal comma, ``dd/mm/yyyy`` dates, a text sign
field, an explicit ``Publication_Date``, and separate preliminary vs definitive vintages
with different publication lags.

These are *plausible-shaped fakes*, not a schema of record. Replace with recorded real
payloads once portal access lands.
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta

from sfe.curate.timebase import periods_in_day

MACROZONES = ("NORD", "SUD")

# Envelope key each endpoint wraps its record list in (best-effort guess; the client's
# extractor is tolerant of this varying).
ENVELOPE_KEY = {
    "daily-prices": "prezzi",
    "preliminary-prices": "prezzi",
    "daily-macrozonal-imbalance": "sbilanciamento",
    "preliminary-macrozonal-imbalance": "sbilanciamento",
    "macrozonal-no-arbitrage-prices": "corrispettivi",
    "load-forecast": "result",
    "wind-production-forecast": "result",
}

# Endpoints published ahead of delivery (D-1 morning) rather than settled afterwards.
_FORECAST_ENDPOINTS = {"load-forecast", "wind-production-forecast"}


def _it_num(x: float, decimals: int = 2) -> str:
    return f"{x:.{decimals}f}".replace(".", ",")


def _seeded(day: date, zone: str, idx: int) -> random.Random:
    return random.Random(f"{day.isoformat()}|{zone}|{idx}")


def _period_shape(day: date, idx: int, n: int) -> float:
    """Smooth intra-day curve in [-1, 1]-ish, peaking morning & evening."""
    frac = idx / max(n - 1, 1)
    return math.sin(frac * 2 * math.pi) + 0.4 * math.sin(frac * 4 * math.pi)


def _forecast_records_for_day(endpoint: str, day: date, resolution: str) -> list[dict]:
    n = periods_in_day(day, resolution)
    # Published ~18:00 on D-2, comfortably before any D-1 decision.
    pub_str = (datetime(day.year, day.month, day.day) - timedelta(days=1, hours=6)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    out: list[dict] = []
    for zone in MACROZONES:
        load_base = 30000.0 if zone == "NORD" else 18000.0
        wind_base = 900.0 if zone == "NORD" else 2600.0
        for i in range(n):
            rng = _seeded(day, f"fc|{zone}", i)
            shape = _period_shape(day, i, n)
            rec = {
                "Date": day.strftime("%d/%m/%Y"),
                "Hour": str(i + 1),
                "Zone": zone,
                "Publication_Date": pub_str,
            }
            if endpoint == "load-forecast":
                rec["forecast_load_MW"] = _it_num(load_base + 6000 * shape + rng.gauss(0, 400), 1)
            else:  # wind-production-forecast
                rec["wind_forecast_MW"] = _it_num(
                    max(0.0, wind_base * (1 + 0.6 * shape) + rng.gauss(0, 250)), 1
                )
            out.append(rec)
    return out


def _records_for_day(endpoint: str, day: date, resolution: str) -> list[dict]:
    if endpoint in _FORECAST_ENDPOINTS:
        return _forecast_records_for_day(endpoint, day, resolution)
    n = periods_in_day(day, resolution)
    prelim = endpoint.startswith("preliminary")
    # Publication lag: preliminary ~ next day 06:00 Rome; definitive ~ day + 3 at 12:00.
    pub = datetime(day.year, day.month, day.day) + (
        timedelta(days=1, hours=6) if prelim else timedelta(days=3, hours=12)
    )
    pub_str = pub.strftime("%Y-%m-%dT%H:%M:%S")
    out: list[dict] = []

    for zone in MACROZONES:
        base_level = 95.0 if zone == "NORD" else 105.0
        for i in range(n):
            rng = _seeded(day, zone, i)
            shape = _period_shape(day, i, n)
            mkt = base_level + 18 * shape + rng.gauss(0, 6)
            spread = 9 * shape + rng.gauss(0, 14) + (0 if prelim else rng.gauss(0, 3))
            imb = mkt + spread
            sign_txt = "+" if spread < 0 else "-"  # area long when imbalance is cheaper
            vol = abs(spread) * 4.0 * (1 if sign_txt == "+" else -1)
            rec = {
                "Date": day.strftime("%d/%m/%Y"),
                "Hour": str(i + 1),
                "Macrozone": zone,
                "Publication_Date": pub_str,
            }
            if "no-arbitrage" in endpoint:
                rec |= {"no_arbitrage_price_EURxMWh": _it_num(abs(rng.gauss(3.5, 1.2)))}
            elif "prices" in endpoint:
                rec |= {
                    "base_price_EURxMWh": _it_num(mkt),
                    "incentive_component_EURxMWh": _it_num(rng.uniform(0, 2)),
                    "unbalance_price_EURxMWh": _it_num(imb),
                }
            else:  # imbalance sign / volume
                rec |= {
                    "zonal_aggregate_sign": sign_txt,
                    "zonal_aggregate_unbalance_MWh": _it_num(vol, 1),
                    "exchanges_MWh": _it_num(rng.gauss(0, 200), 1),
                    "foreign_MWh": _it_num(rng.gauss(0, 120), 1),
                }
            out.append(rec)
    return out


def generate(endpoint: str, date_from: date, date_to: date, resolution: str = "PT1H") -> list[dict]:
    recs: list[dict] = []
    d = date_from
    while d <= date_to:
        recs.extend(_records_for_day(endpoint, d, resolution))
        d += timedelta(days=1)
    return recs


def envelope(endpoint: str, records: list[dict]) -> dict:
    return {ENVELOPE_KEY.get(endpoint, "result"): records}
