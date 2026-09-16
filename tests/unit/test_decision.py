import numpy as np

from sfe.decision.abstain import AbstainRule
from sfe.decision.limits import Limits
from sfe.decision.market_impact import MarketImpact
from sfe.decision.sizing import Sizer

FAN = {0.1: -12.0, 0.25: -6.0, 0.5: 2.0, 0.75: 10.0, 0.9: 18.0}


def _sizer(eta=0.02, aggr=1.0):
    return Sizer(aggressiveness=aggr, rounding_mwh=0.0, impact=MarketImpact(eta))


def test_expected_and_sigma_spread():
    s = _sizer()
    assert s.expected_spread(FAN) == np.mean(list(FAN.values()))
    assert abs(s.sigma_spread(FAN) - (18.0 - -12.0) / 2.5631) < 1e-6


def test_net_edge_charge_can_kill_trade():
    s = _sizer()
    # E[S] = 2.4; charge 1.0 -> net 1.4 same sign -> kept
    assert s.net_edge(2.4, 1.0) == 2.4 - 1.0
    # charge 5.0 > 2.4 -> flips sign -> 0
    assert s.net_edge(2.4, 5.0) == 0.0


def test_target_delta_direction_and_scale():
    s = _sizer(eta=0.02, aggr=1.0)
    # E[S] ~ +2.4, charge 0 -> edge 2.4 -> Delta* = 2.4 / (2*0.02) = 60
    d = s.target_delta(FAN, 0.0)
    assert d > 0
    assert abs(d - 2.4 / 0.04) < 1e-6
    # halving aggressiveness halves the size
    assert abs(_sizer(aggr=0.5).target_delta(FAN, 0.0) - d / 2) < 1e-6


def test_market_impact_optimum_and_effective_spread():
    mi = MarketImpact(0.05)
    assert mi.unconstrained_optimum(10.0) == 100.0
    assert mi.effective_spread(10.0, 100.0) == 10.0 - 0.05 * 100.0


def test_limits_cap_is_min_of_rules():
    lim = Limits(pct_load_cap=0.1, per_mtu_mwh_cap=50.0, daily_mwh_cap=1e9,
                 cvar_alpha=0.95, cvar_limit_eur_per_mwh=25.0)
    assert lim.per_mtu_cap(200.0) == 20.0          # 0.1 * 200 < 50
    assert lim.per_mtu_cap(2000.0) == 50.0         # abs cap binds
    clipped, why = lim.clip(70.0, 200.0)
    assert clipped == 20.0 and why == "per_mtu_cap"
    assert lim.clip(5.0, 200.0) == (5.0, None)


def test_limits_cvar_uses_worst_tail():
    lim = Limits(0.1, None, 1e9, 0.95, 25.0)
    # direction +1, charge 3 -> pnl scenarios = S - 3 ; worst = -12 - 3 = -15
    cv = lim.cvar_per_mwh(FAN, charge=3.0, direction=1)
    assert cv == -12.0 - 3.0
    assert lim.cvar_ok(FAN, 3.0, 1) is True
    assert lim.cvar_ok(FAN, 20.0, 1) is False


def test_daily_cap_scales_group_down():
    import pandas as pd

    lim = Limits(0.1, None, daily_mwh_cap=100.0, cvar_alpha=0.95, cvar_limit_eur_per_mwh=25.0)
    delta = pd.Series([60.0, -60.0, 30.0])
    grp = pd.Series(["d1", "d1", "d2"])
    out = lim.apply_daily_cap(delta, grp)
    assert abs(out.iloc[:2].abs().sum() - 100.0) < 1e-9   # d1 scaled 120 -> 100
    assert out.iloc[2] == 30.0                            # d2 untouched


def test_abstain_reasons():
    rule = AbstainRule(min_edge_eur_per_mwh=1.5, uncertainty_mult=0.35, sign_prob_deadband=0.05)
    assert rule.check(float("nan"), 5.0, 0.7) == "no_forecast"
    assert rule.check(0.5, 5.0, 0.7) == "edge_below_min"
    assert rule.check(2.0, 100.0, 0.7) == "wide_distribution"
    assert rule.check(5.0, 1.0, 0.51) == "sign_indeterminate"
    assert rule.check(5.0, 1.0, 0.8) is None
