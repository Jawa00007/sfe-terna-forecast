"""Contract tests: parse + land against the mock Terna payloads (PRD Phase 0, DQ-1..DQ-3).

No Terna sandbox exists, so these run against sfe.testing.mock_terna, whose payloads
deliberately mimic the real feed's quirks (string numerics, decimal comma, dd/mm/yyyy,
Publication_Date, preliminary vs definitive lag).
"""

from datetime import date

import pandas as pd
import pytest

from sfe.curate.timebase import period_start_utc
from sfe.curate.vintage import as_of_view, assert_no_leakage
from sfe.ingest.endpoints import get_endpoint
from sfe.ingest.flows import ingest_endpoint
from sfe.ingest.landing import build_cells

pytestmark = pytest.mark.contract

WEEK = (date(2025, 1, 6), date(2025, 1, 12))


def test_raw_payload_has_string_numerics_with_comma(mock_terna_client):
    recs = mock_terna_client.fetch("daily-prices", *WEEK)
    assert recs, "mock returned no records"
    sample = recs[0]
    assert isinstance(sample["unbalance_price_EURxMWh"], str)
    assert "," in sample["unbalance_price_EURxMWh"]
    assert sample["Date"].count("/") == 2  # dd/mm/yyyy


def test_windowing_covers_full_range(mock_terna_client):
    recs = mock_terna_client.fetch("daily-prices", date(2024, 12, 1), date(2025, 2, 15))
    days = {r["Date"] for r in recs}
    assert "01/12/2024" in days and "15/02/2025" in days
    assert len(days) == 77  # inclusive span, chunked across >60-day windows


def test_build_cells_parses_and_shapes(mock_terna_client):
    spec = get_endpoint("daily-prices")
    recs = mock_terna_client.fetch(spec, *WEEK)
    cells = build_cells(
        spec, recs, period_start_resolver=period_start_utc, default_area_level="macrozone"
    )
    assert set(cells["field"].unique()) == {
        "base_price_EURxMWh", "incentive_component_EURxMWh", "unbalance_price_EURxMWh"
    }
    assert cells["value_num"].notna().all()
    assert str(cells["_mtu_start"].dt.tz) == "UTC"
    assert (cells["_publication_ts"] > cells["_mtu_start"]).all()  # published after the period
    assert {"NORD", "SUD"} == set(cells["_area"].unique())


def test_dst_fallback_day_has_25_periods(mock_terna_client):
    spec = get_endpoint("daily-prices")
    recs = mock_terna_client.fetch(spec, date(2025, 10, 26), date(2025, 10, 26))
    cells = build_cells(
        spec, recs, period_start_resolver=period_start_utc, default_area_level="macrozone"
    )
    nord_price = cells[(cells._area == "NORD") & (cells.field == "unbalance_price_EURxMWh")]
    assert len(nord_price) == 25


def test_land_is_idempotent_and_append_only(mock_terna_client, tmp_settings):
    first = ingest_endpoint("daily-prices", *WEEK, client=mock_terna_client)
    assert first["new"] > 0 and first["skipped"] == 0
    second = ingest_endpoint("daily-prices", *WEEK, client=mock_terna_client)
    assert second["new"] == 0 and second["skipped"] == first["new"]


def test_point_in_time_view_excludes_future_vintages(mock_terna_client, tmp_settings):
    from sfe.io.storage import get_storage

    # land both the preliminary (D+1) and definitive (D+3) series for the same week
    ingest_endpoint("preliminary-prices", *WEEK, client=mock_terna_client)
    ingest_endpoint("daily-prices", *WEEK, client=mock_terna_client)

    storage = get_storage()
    landing = pd.concat(
        [
            storage.read_dataset("landing/terna/fees/preliminary-prices"),
            storage.read_dataset("landing/terna/fees/daily-prices"),
        ],
        ignore_index=True,
    )

    # Decision time = 08 Jan 2025 07:00Z: preliminary for 06-07 Jan is out, definitive is not.
    as_of = "2025-01-08T07:00:00Z"
    view = as_of_view(landing, as_of)
    assert_no_leakage(view, as_of)
    assert not view.empty
    assert (view["_vintage"] == "preliminary").all()
