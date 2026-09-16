from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from sfe.monitoring.drift import feature_drift, psi, staleness_report
from sfe.monitoring.pnl_attribution import attribute, daily_attribution


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x.copy()) < 0.02


def test_psi_flags_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(2.5, 1, 5000)
    assert psi(a, b) > 0.25


def test_feature_drift_levels():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"f": rng.normal(0, 1, 3000), "g": rng.normal(0, 1, 3000)})
    cur = pd.DataFrame({"f": rng.normal(0, 1, 3000), "g": rng.normal(3, 1, 3000)})
    d = feature_drift(ref, cur, ["f", "g"]).set_index("feature")
    assert d.loc["g", "level"] == "alert"
    assert d.loc["f", "level"] == "ok"


def _recs():
    mtu = pd.Timestamp("2025-01-10T12:00:00Z")
    return pd.DataFrame(
        [
            {"_area": "NORD", "_mtu_start": mtu, "delta_mwh": -10.0, "e_spread": -5.0,
             "charge_eur_per_mwh": 2.0, "reason": "sized"},
            {"_area": "SUD", "_mtu_start": mtu, "delta_mwh": 8.0, "e_spread": 6.0,
             "charge_eur_per_mwh": 2.0, "reason": "sized"},
        ]
    )


def _actual():
    mtu = pd.Timestamp("2025-01-10T12:00:00Z")
    return pd.DataFrame(
        [
            {"_area": "NORD", "_mtu_start": mtu, "spread_S": -12.0,
             "charge_realized_eur_per_mwh": 3.0},
            {"_area": "SUD", "_mtu_start": mtu, "spread_S": 4.0,
             "charge_realized_eur_per_mwh": 1.0},
        ]
    )


def test_attribution_decomposition_is_exact():
    att = attribute(_recs(), _actual())
    err = att["pnl_realized_eur"] - att["pnl_expected_eur"]
    recomposed = att["err_spread_eur"] + att["err_charge_eur"]
    assert np.allclose(err, recomposed)
    # NORD: realized = (-10)(-12) - 10*3 = 90 ; expected = (-10)(-5) - 10*2 = 30
    nord = att[att._area == "NORD"].iloc[0]
    assert nord["pnl_realized_eur"] == 90.0 and nord["pnl_expected_eur"] == 30.0
    assert nord["err_spread_eur"] == -10.0 * (-12.0 - -5.0)   # 70
    assert nord["err_charge_eur"] == -10.0 * (3.0 - 2.0)      # -10


def test_daily_attribution_rolls_up():
    d = daily_attribution(attribute(_recs(), _actual()))
    assert len(d) == 1
    row = d.iloc[0]
    assert abs(row["unexplained_eur"]) < 1e-9
    assert row["n_positions"] == 2


def test_staleness_report(tmp_settings):
    from sfe.io.storage import META_COLUMNS, get_storage

    now = datetime.now(UTC)
    frag = pd.DataFrame(
        [{
            **{c: None for c in META_COLUMNS},
            "_source": "terna", "_endpoint": "daily-prices", "_area": "NORD",
            "_area_level": "macrozone", "_mtu_start": now, "_resolution": "PT1H",
            "_vintage": "definitive", "_publication_ts": now - timedelta(hours=50),
            "_ingested_at": now, "_payload_hash": "h1", "_raw": "{}",
            "field": "x", "value_raw": "1", "value_num": 1.0,
        }],
        columns=[*META_COLUMNS, "field", "value_raw", "value_num"],
    )
    get_storage().write_fragment("landing/terna/fees/daily-prices", frag)
    rep = staleness_report(
        ["landing/terna/fees/daily-prices", "landing/terna/fees/missing"],
        as_of=now, max_age_hours=30,
    ).set_index("dataset")
    assert rep.loc["landing/terna/fees/daily-prices", "stale"]     # 50h > 30h
    assert rep.loc["landing/terna/fees/missing", "stale"]          # absent
