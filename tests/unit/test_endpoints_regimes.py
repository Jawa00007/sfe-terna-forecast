from datetime import date

from sfe.curate import regimes
from sfe.ingest.endpoints import get_endpoint, iter_endpoints, target_endpoints


def test_endpoint_spec_fields():
    e = get_endpoint("daily-prices")
    assert e.path == "/fees/v1.0/daily-prices"
    assert e.role == "target_price"
    assert e.vintage == "definitive"
    assert e.dataset == "landing/terna/fees/daily-prices"
    assert e.date_param_names == {"from": "dateFrom", "to": "dateTo"}
    assert e.numeric_as_string is True


def test_iter_and_targets():
    fees = list(iter_endpoints(group="fees"))
    assert {e.name for e in fees} >= {"daily-prices", "daily-macrozonal-imbalance"}
    roles = {e.role for e in target_endpoints()}
    assert roles == {"target_sign", "target_price", "charge"}


def test_regime_pricing_rule_break_2022():
    assert regimes.pricing_rule(date(2021, 12, 31)) == "dual_price"
    assert regimes.pricing_rule(date(2022, 4, 1)) == "single_price"


def test_regime_mtu_resolution_and_datatype():
    assert regimes.mtu_resolution(date(2024, 1, 1)) == "PT1H"
    assert regimes.terna_data_type(date(2024, 1, 1)) == "Orario"
    assert regimes.mtu_resolution(date(2026, 6, 1)) == "PT15M"


def test_unconfirmed_regimes_listed():
    # every regime entry ships unconfirmed until Phase 0 sign-off
    assert regimes.unconfirmed_regimes()
