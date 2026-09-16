"""One-shot startup seeding for a public/demo deployment with no real Terna access.

Runs entirely in-process against the synthetic Terna payloads (:mod:`sfe.testing.synth`)
via an ASGI-wired ``TernaClient`` - no network calls, no second server needed. Intended to
run once at container boot, before the dashboard starts (see ``render.yaml``).

Skips the backfill if landing data is already present (e.g. a persistent disk carried it
over); always (re)trains the model set so it reflects whatever the storage root holds.
"""

from __future__ import annotations

from datetime import date, timedelta

from starlette.testclient import TestClient

from sfe.config import TernaSettings
from sfe.features.spec import HORIZON_OFFSET_H
from sfe.ingest.external import SyntheticGME, land_provider
from sfe.ingest.flows import ingest_endpoint
from sfe.ingest.terna_client import TernaClient
from sfe.io.storage import get_storage
from sfe.serve.inference import ensure_models
from sfe.testing.mock_terna import app as mock_app

_ENDPOINTS = (
    "daily-prices",
    "daily-macrozonal-imbalance",
    "preliminary-prices",
    "preliminary-macrozonal-imbalance",
    "macrozonal-no-arbitrage-prices",
    "load-forecast",
    "wind-production-forecast",
)


def _mock_client() -> TernaClient:
    settings = TernaSettings(
        client_id="seed",
        client_secret="seed",
        base_url="http://seed-terna",
        token_url="http://seed-terna/oauth/accessToken",
    )
    http = TestClient(mock_app, base_url="http://seed-terna")
    return TernaClient(settings=settings, client=http)


def already_seeded() -> bool:
    return get_storage().exists("landing/terna/fees/daily-prices")


# Small trees for a fast, low-CPU container boot; still plenty to demo calibration + fans.
# n_jobs is pinned rather than -1: on a CPU-throttled/shared container, os.cpu_count() often
# over-reports the actual quota, and LightGBM then oversubscribes threads and gets slower,
# not faster - triggering exactly the kind of fair-use throttle free tiers apply.
_FAST_PARAMS = {"n_estimators": 60, "num_leaves": 15, "verbose": -1, "n_jobs": 1}


def seed(*, end: date | None = None, months: float = 0.5, lookback_days: int = 12) -> None:
    end = end or (date.today() - timedelta(days=2))
    start = end - timedelta(days=30 * months)

    if already_seeded():
        print("seed: landing data already present, skipping backfill", flush=True)
    else:
        client = _mock_client()
        for ep in _ENDPOINTS:
            stats = ingest_endpoint(ep, start, end, client=client)
            print(f"seed: {ep} -> {stats}", flush=True)
        client.close()
        gme_stats = land_provider(SyntheticGME(), start, end)
        print(f"seed: gme -> {gme_stats}", flush=True)

    for horizon in HORIZON_OFFSET_H:
        try:
            ensure_models(horizon, end=end, lookback_days=lookback_days, params=_FAST_PARAMS)
            print(f"seed: {horizon} model set ready", flush=True)
        except RuntimeError as e:
            print(f"seed: could not prepare {horizon} models: {e}", flush=True)


if __name__ == "__main__":
    seed()
