from datetime import UTC, datetime

import pandas as pd
import pytest

from sfe.curate.vintage import as_of_view, assert_no_leakage, pivot_fields


def _landing() -> pd.DataFrame:
    mtu = datetime(2025, 1, 10, 12, 0, tzinfo=UTC)
    rows = [
        # preliminary published D+1, then definitive published D+3 with a revised value
        dict(_area="NORD", _mtu_start=mtu, field="unbalance_price_EURxMWh",
             _vintage="preliminary", _publication_ts=datetime(2025, 1, 11, 6, tzinfo=UTC),
             value_num=100.0),
        dict(_area="NORD", _mtu_start=mtu, field="unbalance_price_EURxMWh",
             _vintage="definitive", _publication_ts=datetime(2025, 1, 13, 12, tzinfo=UTC),
             value_num=118.5),
        dict(_area="SUD", _mtu_start=mtu, field="unbalance_price_EURxMWh",
             _vintage="preliminary", _publication_ts=datetime(2025, 1, 11, 6, tzinfo=UTC),
             value_num=95.0),
    ]
    return pd.DataFrame(rows)


def test_as_of_before_any_publication_is_empty():
    v = as_of_view(_landing(), "2025-01-11T00:00:00Z")
    assert v.empty


def test_as_of_sees_only_preliminary_midweek():
    v = as_of_view(_landing(), "2025-01-12T00:00:00Z")
    assert_no_leakage(v, "2025-01-12T00:00:00Z")
    nord = v[(v._area == "NORD") & (v.field == "unbalance_price_EURxMWh")]
    assert len(nord) == 1
    assert nord.iloc[0]["_vintage"] == "preliminary"
    assert nord.iloc[0]["value_num"] == 100.0


def test_definitive_supersedes_once_published():
    v = as_of_view(_landing(), "2025-01-20T00:00:00Z")
    nord = v[(v._area == "NORD") & (v.field == "unbalance_price_EURxMWh")]
    assert nord.iloc[0]["_vintage"] == "definitive"
    assert nord.iloc[0]["value_num"] == 118.5


def test_leakage_guard_trips():
    v = _landing()  # unfiltered - contains future publications relative to this as_of
    with pytest.raises(AssertionError):
        assert_no_leakage(v, "2025-01-12T00:00:00Z")


def test_pivot_fields_shape():
    v = as_of_view(_landing(), "2025-01-20T00:00:00Z")
    wide = pivot_fields(v)
    assert {"_area", "_mtu_start", "unbalance_price_EURxMWh"} <= set(wide.columns)
    assert len(wide) == 2  # NORD + SUD
