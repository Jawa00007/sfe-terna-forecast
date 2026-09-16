from datetime import UTC, datetime

import pandas as pd
import pytest

from sfe.io.storage import META_COLUMNS, LocalParquetBackend


def _frag(area: str, val: float) -> pd.DataFrame:
    now = datetime.now(UTC)
    return pd.DataFrame(
        [
            {
                "_source": "terna", "_endpoint": "daily-prices", "_area": area,
                "_area_level": "macrozone", "_mtu_start": now, "_resolution": "PT1H",
                "_vintage": "definitive", "_publication_ts": now, "_ingested_at": now,
                "_payload_hash": f"{area}-{val}", "_raw": "{}",
                "field": "unbalance_price_EURxMWh", "value_raw": str(val), "value_num": val,
            }
        ],
        columns=[*META_COLUMNS, "field", "value_raw", "value_num"],
    )


def test_append_only_accumulates(tmp_path):
    b = LocalParquetBackend(tmp_path)
    b.write_fragment("landing/terna/fees/daily-prices", _frag("NORD", 1.0))
    b.write_fragment("landing/terna/fees/daily-prices", _frag("SUD", 2.0))
    df = b.read_dataset("landing/terna/fees/daily-prices")
    assert len(df) == 2
    assert len(b.list_fragments("landing/terna/fees/daily-prices")) == 2


def test_missing_meta_columns_rejected(tmp_path):
    b = LocalParquetBackend(tmp_path)
    bad = _frag("NORD", 1.0).drop(columns=["_publication_ts"])
    with pytest.raises(ValueError, match="metadata columns"):
        b.write_fragment("landing/x", bad)


def test_empty_fragment_rejected(tmp_path):
    b = LocalParquetBackend(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        b.write_fragment("landing/x", _frag("NORD", 1.0).iloc[0:0])


def test_read_missing_dataset_is_empty(tmp_path):
    b = LocalParquetBackend(tmp_path)
    assert b.read_dataset("landing/nope").empty
    assert not b.exists("landing/nope")
