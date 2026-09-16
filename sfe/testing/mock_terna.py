"""Local stand-in for the Terna Developer Portal (PRD Phase 0: no sandbox exists).

Run it, point the client at it, and the whole ingest path works offline::

    python -m sfe.testing.mock_terna            # serves on http://127.0.0.1:8900
    SFE_TERNA__BASE_URL=http://127.0.0.1:8900 \\
    SFE_TERNA__TOKEN_URL=http://127.0.0.1:8900/oauth/accessToken \\
    SFE_TERNA__CLIENT_ID=mock SFE_TERNA__CLIENT_SECRET=mock \\
    python -m sfe.scripts.backfill --endpoint daily-prices --from 2025-01-01 --to 2025-01-07

It also serves as the fixture target for contract tests (via ASGI transport, no socket).
"""

from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, HTTPException, Query

from sfe.ingest.landing import _parse_date_flexible
from sfe.testing import synth

app = FastAPI(title="Mock Terna Developer Portal", version="0.0.0")


@app.post("/oauth/accessToken")
def access_token() -> dict:
    return {"access_token": "mock-token", "token_type": "Bearer", "expires_in": 3600}


def _serve(endpoint: str, date_from: str, date_to: str, data_type: str | None) -> dict:
    d_from = _parse_date_flexible(date_from)
    d_to = _parse_date_flexible(date_to)
    if not d_from or not d_to:
        raise HTTPException(
            400, f"bad date window {date_from!r}..{date_to!r}; expected dd/mm/yyyy"
        )
    if d_to < d_from:
        raise HTTPException(400, "dateTo precedes dateFrom")
    resolution = "PT15M" if data_type == "Quarto Orario" else "PT1H"
    records = synth.generate(endpoint, d_from, d_to, resolution=resolution)
    return synth.envelope(endpoint, records)


def _register(path: str, endpoint: str) -> None:
    @app.get(path, name=endpoint)
    def _handler(  # noqa: ANN202
        dateFrom: str = Query(...),
        dateTo: str = Query(...),
        dataType: str | None = Query(None),
    ) -> dict:
        return _serve(endpoint, dateFrom, dateTo, dataType)


for _ep in (
    "daily-prices",
    "preliminary-prices",
    "daily-macrozonal-imbalance",
    "preliminary-macrozonal-imbalance",
    "macrozonal-no-arbitrage-prices",
):
    _register(f"/fees/v1.0/{_ep}", _ep)

# A couple of feature endpoints so the residual-load family is exercisable offline.
_register("/market/v1.0/load-forecast", "load-forecast")
_register("/generation/v1.0/wind-production-forecast", "wind-production-forecast")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "now": datetime.utcnow().isoformat()}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8900)


if __name__ == "__main__":
    main()
