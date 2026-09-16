from __future__ import annotations

import pytest
from starlette.testclient import TestClient

import sfe.config as config
from sfe.ingest.terna_client import TernaClient
from sfe.testing.mock_terna import app


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """Isolate storage under tmp_path and reset cached settings/yaml."""
    monkeypatch.setenv("SFE_STORAGE__ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("SFE_ENV", "test")
    monkeypatch.delenv("SFE_TERNA__BASE_URL", raising=False)
    config.get_settings.cache_clear()
    config.load_yaml.cache_clear()
    yield config.get_settings()
    config.get_settings.cache_clear()
    config.load_yaml.cache_clear()


@pytest.fixture
def mock_terna_client(tmp_settings):
    """A TernaClient wired to the in-process mock app via ASGI transport (no sockets)."""
    settings = config.TernaSettings(
        client_id="mock",
        client_secret="mock",
        base_url="http://mock-terna",
        token_url="http://mock-terna/oauth/accessToken",
    )
    # Starlette's TestClient is a sync httpx.Client that drives the ASGI app in-process.
    http = TestClient(app, base_url="http://mock-terna")
    client = TernaClient(settings=settings, client=http)
    yield client
    client.close()


@pytest.fixture
def landed_two_weeks(mock_terna_client):
    """Land two weeks of fees + feature endpoints into the isolated tmp store."""
    from datetime import date

    from sfe.ingest.flows import ingest_endpoint

    span = (date(2025, 1, 6), date(2025, 1, 19))
    for ep in _FEED_ENDPOINTS:
        ingest_endpoint(ep, *span, client=mock_terna_client)
    return span


_FEED_ENDPOINTS = (
    "daily-prices",
    "daily-macrozonal-imbalance",
    "preliminary-prices",
    "preliminary-macrozonal-imbalance",
    "macrozonal-no-arbitrage-prices",
    "load-forecast",
    "wind-production-forecast",
)


@pytest.fixture
def landed_six_weeks(mock_terna_client):
    """~6 weeks of feeds - enough for a chronological train/valid/test split."""
    from datetime import date

    from sfe.ingest.flows import ingest_endpoint

    span = (date(2024, 12, 1), date(2025, 1, 15))
    for ep in _FEED_ENDPOINTS:
        ingest_endpoint(ep, *span, client=mock_terna_client)
    return span
