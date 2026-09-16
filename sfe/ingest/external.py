"""Non-Terna sources (PRD 6.2): GME (P_mkt leg), ENTSO-E, weather.

Each provider yields *observations* in the shape :func:`sfe.ingest.landing.build_cells_long`
expects, so everything lands in the same vintaged long-format store.

Only the GME leg has a synthetic implementation for offline work; ENTSO-E and weather are
interface stubs until credentials / a vendor are chosen (open questions #7, #8).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Protocol

from sfe.config import load_yaml
from sfe.curate.timebase import ROME, day_grid
from sfe.ingest.landing import build_cells_long, land
from sfe.io.storage import StorageBackend, get_storage


class Provider(Protocol):
    source: str
    dataset: str

    def fetch(self, date_from: date, date_to: date) -> list[dict[str, Any]]:  # pragma: no cover
        ...


def _target_area_level() -> str:
    return load_yaml("zones").get("target_level", "macrozone")


def land_provider(
    provider: Provider,
    date_from: date,
    date_to: date,
    *,
    storage: StorageBackend | None = None,
) -> dict:
    obs = provider.fetch(date_from, date_to)
    cells = build_cells_long(
        provider.source, provider.dataset.split("/")[-1], provider.dataset, obs,
        area_level=_target_area_level(),
    )
    stats = land(provider.dataset, cells, backend=storage or get_storage())
    return stats | {"source": provider.source, "observations": len(obs)}


class SyntheticGME:
    """Stand-in for GME MGP zonal day-ahead prices - the ``P_mkt`` reference leg.

    Published on D-1 around the MGP results time (~12:45 Europe/Rome).
    """

    source = "gme"
    dataset = "landing/gme/mgp-zonal-prices"

    def __init__(self, areas: tuple[str, ...] | None = None):
        z = load_yaml("zones")
        self.areas = areas or tuple(z["levels"][z["target_level"]]["areas"])

    def fetch(self, date_from: date, date_to: date) -> list[dict[str, Any]]:
        import random

        out: list[dict[str, Any]] = []
        d = date_from
        while d <= date_to:
            pub = (datetime(d.year, d.month, d.day, 12, 45, tzinfo=ROME) - timedelta(days=1))
            for area in self.areas:
                base = 95.0 if area.startswith("N") else 105.0
                for g in day_grid(d, "PT1H"):
                    rng = random.Random(f"gme|{area}|{g.isoformat()}")
                    hour_frac = g.astimezone(ROME).hour / 24.0
                    price = base + 20 * (0.5 - abs(hour_frac - 0.5)) * 2 + rng.gauss(0, 5)
                    out.append(
                        {
                            "area": area,
                            "mtu_start": g,
                            "resolution": "PT1H",
                            "field": "mgp_price_EURxMWh",
                            "value": round(price, 2),
                            "publication_ts": pub,
                            "vintage": "definitive",
                        }
                    )
            d += timedelta(days=1)
        return out


class EntsoeProvider:  # pragma: no cover - stub
    source = "entsoe"
    dataset = "landing/entsoe/cross-border-flows"

    def fetch(self, date_from: date, date_to: date) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "ENTSO-E Transparency needs an API token; implement against "
            "https://transparency.entsoe.eu/ once obtained (open question #7)."
        )


class WeatherProvider:  # pragma: no cover - stub
    source = "weather"
    dataset = "landing/weather/zone-weighted"

    def fetch(self, date_from: date, date_to: date) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "Wire ECMWF Open Data / ICON-EU (or a commercial vendor) here; must also emit "
            "NWP run-to-run spread as a forecast-error proxy (open question #8)."
        )
