"""REST API for downstream systems (PRD F-15).

Endpoints
---------
GET  /health                              liveness + which models are loaded
GET  /forecast?date=&horizon=             full bundle: fans, Delta, utilisation, degraded flag
GET  /recommendations?date=&horizon=      non-abstaining positions + a summary
GET  /drivers?date=&horizon=              model feature importances
GET  /staleness                           upstream feed ages
POST /audit?date=&horizon=                record the current bundle to the audit log
GET  /audit/{audit_id}                    fetch one audit row

Models are loaded from the registry on first use (trained on recent history if absent).
"""

from __future__ import annotations

from datetime import date

from fastapi import FastAPI, HTTPException, Query

from sfe.features.store import default_store
from sfe.monitoring.audit import AuditLog
from sfe.monitoring.drift import staleness_report
from sfe.serve.inference import ensure_models, generate_bundle

_FEEDS = [
    "landing/terna/fees/daily-prices",
    "landing/terna/fees/daily-macrozonal-imbalance",
    "landing/terna/fees/macrozonal-no-arbitrage-prices",
    "landing/terna/market/load-forecast",
    "landing/terna/generation/wind-production-forecast",
]


def create_app() -> FastAPI:
    app = FastAPI(title="Sbilanciamento Forecast Engine", version="0.1.0")
    state: dict = {"store": None, "models": {}}

    def store():
        if state["store"] is None:
            state["store"] = default_store()
        return state["store"]

    def models(horizon: str):
        if horizon not in state["models"]:
            try:
                state["models"][horizon] = ensure_models(horizon, store=store())
            except RuntimeError as e:
                raise HTTPException(503, str(e)) from e
        return state["models"][horizon]

    def _bundle(d: str, horizon: str):
        try:
            dd = date.fromisoformat(d)
        except ValueError as e:
            raise HTTPException(422, f"bad date {d!r}; expected YYYY-MM-DD") from e
        sign, spread = models(horizon)
        return generate_bundle(
            dd, horizon, sign_model=sign, spread_model=spread, store=store()
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "models_loaded": sorted(state["models"])}

    @app.get("/forecast")
    def forecast(date: str = Query(...), horizon: str = "D-1") -> dict:  # noqa: A002
        return _bundle(date, horizon).to_dict()

    @app.get("/recommendations")
    def recommendations(date: str = Query(...), horizon: str = "D-1") -> dict:  # noqa: A002
        b = _bundle(date, horizon)
        active = b.table[~b.table["abstain"]] if not b.table.empty else b.table
        return {
            "delivery_day": b.delivery_day.isoformat(),
            "horizon": b.horizon,
            "as_of": b.as_of.isoformat(),
            "degraded": b.degraded,
            "n_total": int(len(b.table)),
            "n_active": int(len(active)),
            "abstention_fraction": (
                float(b.table["abstain"].mean()) if len(b.table) else None
            ),
            "expected_net_margin_eur": (
                float(active["expected_net_margin_eur"].sum()) if len(active) else 0.0
            ),
            "positions": active.assign(_mtu_start=active["_mtu_start"].astype(str)).to_dict(
                orient="records"
            )
            if len(active)
            else [],
        }

    @app.get("/drivers")
    def drivers(date: str = Query(...), horizon: str = "D-1") -> dict:  # noqa: A002
        return {"drivers": _bundle(date, horizon).drivers.to_dict(orient="records")}

    @app.get("/staleness")
    def staleness() -> dict:
        return {"feeds": staleness_report(_FEEDS).to_dict(orient="records")}

    @app.post("/audit")
    def audit(date: str = Query(...), horizon: str = "D-1") -> dict:  # noqa: A002
        b = _bundle(date, horizon)
        ids = AuditLog().record(b)
        return {"recorded": len(ids), "audit_ids": ids}

    @app.get("/audit/{audit_id}")
    def audit_get(audit_id: str) -> dict:
        row = AuditLog().get(audit_id)
        if row is None:
            raise HTTPException(404, f"no audit row {audit_id!r}")
        return row

    return app


app = create_app()
