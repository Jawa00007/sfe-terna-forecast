import numpy as np
import pandas as pd

from sfe.backtest.economic import economic_report, go_no_go, realize_pnl


def _recs():
    mtu = pd.Timestamp("2025-01-10T12:00:00Z")
    return pd.DataFrame(
        [
            # correct direction: Delta<0 into S<0 -> positive pnl
            {"_area": "NORD", "_mtu_start": mtu, "delta_mwh": -10.0, "reason": "sized"},
            # wrong direction
            {"_area": "SUD", "_mtu_start": mtu, "delta_mwh": 10.0, "reason": "sized"},
            # abstained
            {"_area": "NORD", "_mtu_start": mtu + pd.Timedelta(hours=1),
             "delta_mwh": 0.0, "reason": "edge_below_min"},
        ]
    )


def _actual():
    mtu = pd.Timestamp("2025-01-10T12:00:00Z")
    return pd.DataFrame(
        [
            {"_area": "NORD", "_mtu_start": mtu, "spread_S": -20.0,
             "charge_realized_eur_per_mwh": 2.0, "load_realized_mwh": 100.0},
            {"_area": "SUD", "_mtu_start": mtu, "spread_S": -20.0,
             "charge_realized_eur_per_mwh": 2.0, "load_realized_mwh": 100.0},
            {"_area": "NORD", "_mtu_start": mtu + pd.Timedelta(hours=1), "spread_S": 5.0,
             "charge_realized_eur_per_mwh": 2.0, "load_realized_mwh": 100.0},
        ]
    )


def test_realize_pnl_arithmetic():
    pnl = realize_pnl(_recs(), _actual())
    # NORD: (-10)*(-20) - 10*2 = 200 - 20 = 180
    assert pnl.loc[pnl._area == "NORD"].iloc[0]["pnl_eur"] == 180.0
    # SUD: 10*(-20) - 10*2 = -220
    assert pnl.loc[pnl._area == "SUD"].iloc[0]["pnl_eur"] == -220.0
    assert bool(pnl.iloc[2]["abstained"]) is True


def test_economic_report_and_gate():
    rep = economic_report(realize_pnl(_recs(), _actual()))
    assert rep["n_scored"] == 3
    assert abs(rep["abstention_fraction"] - 1 / 3) < 1e-9
    # net margin = (180 - 220 + 0) / (100+100+100) = -40/300
    assert abs(rep["net_margin_eur_per_mwh"] - (-40.0 / 300.0)) < 1e-9
    ok, _ = go_no_go(rep)
    assert ok is False
    ok2, _ = go_no_go({"net_margin_eur_per_mwh": 1.5}, min_margin_eur_per_mwh=1.0)
    assert ok2 is True


def test_value_weighted_hit_rate_weights_big_spreads():
    pnl = realize_pnl(_recs(), _actual())
    rep = economic_report(pnl)
    # one hit (NORD, |S|=20), one miss (SUD, |S|=20) among active -> 0.5
    assert abs(rep["value_weighted_hit_rate"] - 0.5) < 1e-9
    assert np.isfinite(rep["worst_mtu_eur"])
