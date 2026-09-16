"""Phase 4 shadow-mode: serve pipeline, audit log, and REST API on landed mock data."""

from datetime import date

import pytest
from starlette.testclient import TestClient

from sfe.monitoring.audit import AuditLog
from sfe.serve.inference import ensure_models, generate_bundle

pytestmark = pytest.mark.contract

_TINY = {"n_estimators": 50, "learning_rate": 0.1, "num_leaves": 15, "verbose": -1, "n_jobs": 1}
_DAY = date(2025, 1, 14)


@pytest.fixture
def registered_models(landed_six_weeks):
    return ensure_models(
        "D-1", end=date(2025, 1, 13), lookback_days=40, force_train=True, params=_TINY
    )


def test_generate_bundle_shape(registered_models):
    sign, spread = registered_models
    b = generate_bundle(_DAY, "D-1", sign_model=sign, spread_model=spread)
    assert not b.table.empty
    cols = set(b.table.columns)
    assert {"sign_prob", "S_p10", "S_p90", "Pimb_p50", "delta_mwh", "abstain",
            "util_pct_load_cap", "util_daily_cap", "cvar_eur_per_mwh"} <= cols
    assert ((b.table["sign_prob"] >= 0) & (b.table["sign_prob"] <= 1)).all()
    # non-crossing fan
    assert (b.table["S_p90"] >= b.table["S_p10"] - 1e-9).all()
    assert not b.drivers.empty
    assert len(b.table) == 24 * 2


def test_audit_roundtrip(registered_models, tmp_settings):
    sign, spread = registered_models
    feats_day = generate_bundle(_DAY, "D-1", sign_model=sign, spread_model=spread)
    log = AuditLog()
    ids = log.record(feats_day, features=feats_day.table[["_area", "_mtu_start", "sign_prob"]])
    assert len(ids) == len(feats_day.table)
    allrows = log.read()
    assert len(allrows) == len(ids)
    one = log.get(ids[0])
    assert one["horizon"] == "D-1" and one["model_version"]
    assert one["features_json"]  # captured feature vector


def test_api_endpoints(registered_models):
    from sfe.api.main import create_app

    client = TestClient(create_app())

    assert client.get("/health").json()["status"] == "ok"

    fc = client.get("/forecast", params={"date": _DAY.isoformat(), "horizon": "D-1"})
    assert fc.status_code == 200
    body = fc.json()
    assert body["rows"] and "degraded" in body
    assert {"delta_mwh", "S_p50"} <= set(body["rows"][0])

    rec = client.get("/recommendations", params={"date": _DAY.isoformat()}).json()
    assert rec["n_total"] == 48
    assert 0.0 <= rec["abstention_fraction"] <= 1.0

    drv = client.get("/drivers", params={"date": _DAY.isoformat()}).json()
    assert len(drv["drivers"]) > 0

    stale = client.get("/staleness").json()
    assert any(f["dataset"].endswith("daily-prices") for f in stale["feeds"])

    posted = client.post("/audit", params={"date": _DAY.isoformat()}).json()
    assert posted["recorded"] > 0
    got = client.get(f"/audit/{posted['audit_ids'][0]}")
    assert got.status_code == 200

    assert client.get("/forecast", params={"date": "nope"}).status_code == 422


def test_api_503_without_models(landed_six_weeks):
    # no models registered and no data near "today" -> ensure_models fails -> 503
    from sfe.api.main import create_app

    client = TestClient(create_app())
    r = client.get("/forecast", params={"date": _DAY.isoformat(), "horizon": "MI2"})
    assert r.status_code == 503
