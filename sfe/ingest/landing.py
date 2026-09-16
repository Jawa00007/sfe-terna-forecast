"""Vintaged, append-only landing writer (PRD DQ-1, DQ-2, F-3).

Raw source records are exploded to *long* format - one row per (period, area, field,
vintage) - so endpoints with different and still-unconfirmed schemas can all land in the
same shape. The original record is retained verbatim in ``_raw``.

Idempotency: each cell gets a ``_payload_hash``; re-landing a batch that overlaps existing
data writes only the genuinely new cells (or nothing).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from sfe.curate.normalize import coerce_str, parse_number
from sfe.ingest.endpoints import EndpointSpec
from sfe.io.storage import META_COLUMNS, StorageBackend, get_storage

# Record keys we treat as dimensions / provenance rather than measured values.
_DATE_KEYS = ("Date", "date", "DATE", "Data", "Giorno")
_HOUR_KEYS = ("Hour", "hour", "Ora", "Quarter", "Quarto", "Period", "Periodo")
_AREA_KEYS = (
    "Macrozone", "macrozone", "Zone", "zone", "Area", "area", "Bidding_Zone", "BiddingZone",
)
_PUB_KEYS = (
    "Publication_Date", "PublicationDate", "publicationDate", "Data_Pubblicazione", "Updated",
)
_NON_VALUE_KEYS = set(_DATE_KEYS + _HOUR_KEYS + _AREA_KEYS + _PUB_KEYS) | {
    "_fetched_at", "Year", "Month",
}


def _parse_date_flexible(value: Any) -> date | None:
    s = coerce_str(value)
    if not s:
        return None
    s = s.split("T")[0].split(" ")[0]
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_ts_flexible(value: Any) -> datetime | None:
    s = coerce_str(value)
    if not s:
        return None
    fmts = (
        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
    )
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    d = _parse_date_flexible(s)
    return datetime(d.year, d.month, d.day, tzinfo=UTC) if d else None


def _first(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in record and record[k] not in (None, ""):
            return record[k]
    return None


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def build_cells(
    spec: EndpointSpec,
    records: list[dict[str, Any]],
    *,
    period_start_resolver,
    default_area_level: str,
    ingested_at: datetime | None = None,
) -> pd.DataFrame:
    """Explode Terna *records* into long-format landing rows.

    ``period_start_resolver(day, period_index, resolution) -> datetime`` supplies the UTC
    period start (see :func:`sfe.curate.timebase.period_start_utc`); ``period_index`` is
    ``None`` when the record has no hour/quarter field (daily grain).
    """
    ingested_at = ingested_at or datetime.now(UTC)
    resolution = "PT15M" if "15" in str(spec.raw.get("resolution", "")) else "PT1H"
    rows: list[dict[str, Any]] = []

    for rec in records:
        day = _parse_date_flexible(_first(rec, _DATE_KEYS))
        hour_raw = _first(rec, _HOUR_KEYS)
        area = coerce_str(_first(rec, _AREA_KEYS)) or "UNKNOWN"
        pub = _parse_ts_flexible(_first(rec, _PUB_KEYS))
        pub = pub or _parse_ts_flexible(rec.get("_fetched_at")) or ingested_at

        period_index = None
        if hour_raw is not None:
            try:
                period_index = int(float(str(hour_raw).strip()))
            except ValueError:
                period_index = None

        mtu_start = None
        if day is not None:
            try:
                mtu_start = period_start_resolver(day, period_index, resolution)
            except (ValueError, TypeError):
                mtu_start = None

        value_keys = [k for k in rec if k not in _NON_VALUE_KEYS and not k.startswith("_")]
        if spec.value_fields:
            value_keys = [k for k in value_keys if k in spec.value_fields] or value_keys

        # Hash the source record without our own provenance keys (e.g. _fetched_at), so a
        # re-pull of the same data is recognised as identical (idempotent landing).
        rec_stable = {k: v for k, v in rec.items() if not k.startswith("_")}

        for field in value_keys:
            raw_val = rec[field]
            rows.append(
                {
                    "_source": "terna",
                    "_endpoint": spec.name,
                    "_area": area,
                    "_area_level": default_area_level,
                    "_mtu_start": mtu_start,
                    "_resolution": resolution if period_index is not None else "P1D",
                    "_vintage": spec.vintage or "definitive",
                    "_publication_ts": pub,
                    "_ingested_at": ingested_at,
                    "_payload_hash": hashlib.sha1(
                        f"{spec.dataset}|{field}|{_canonical(rec_stable)}".encode()
                    ).hexdigest(),
                    "_raw": _canonical(rec),
                    "field": field,
                    "value_raw": coerce_str(raw_val),
                    "value_num": parse_number(raw_val),
                }
            )

    return _finalize(rows)


def _finalize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=[*META_COLUMNS, "field", "value_raw", "value_num"])
    for col in ("_mtu_start", "_publication_ts", "_ingested_at"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def build_cells_long(
    source: str,
    endpoint: str,
    dataset: str,
    observations: Iterable[dict[str, Any]],
    *,
    area_level: str,
    ingested_at: datetime | None = None,
) -> pd.DataFrame:
    """Build landing rows from already-normalised observations (non-Terna sources).

    Each observation is a dict with keys: ``area``, ``mtu_start`` (tz-aware / UTC-coercible),
    ``resolution``, ``field``, ``value``; optional ``vintage`` (default ``definitive``),
    ``publication_ts`` (default = ``mtu_start``, i.e. known at period start), ``raw``.
    """
    ingested_at = ingested_at or datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for obs in observations:
        raw = obs.get("raw", {k: v for k, v in obs.items() if k != "raw"})
        pub = obs.get("publication_ts") or obs["mtu_start"]
        rows.append(
            {
                "_source": source,
                "_endpoint": endpoint,
                "_area": coerce_str(obs.get("area")) or "UNKNOWN",
                "_area_level": area_level,
                "_mtu_start": obs["mtu_start"],
                "_resolution": obs.get("resolution", "PT1H"),
                "_vintage": obs.get("vintage", "definitive"),
                "_publication_ts": pub,
                "_ingested_at": ingested_at,
                "_payload_hash": hashlib.sha1(
                    f"{dataset}|{obs['field']}|{obs['area']}|{obs['mtu_start']}|"
                    f"{obs.get('vintage', 'definitive')}|{_canonical(raw)}".encode()
                ).hexdigest(),
                "_raw": _canonical(raw),
                "field": obs["field"],
                "value_raw": coerce_str(obs.get("value")),
                "value_num": parse_number(obs.get("value")),
            }
        )
    return _finalize(rows)


def land(
    dataset: str,
    df: pd.DataFrame,
    *,
    backend: StorageBackend | None = None,
) -> dict[str, int]:
    """Append only the cells of *df* whose ``_payload_hash`` is not already stored."""
    backend = backend or get_storage()
    if df.empty:
        return {"received": 0, "new": 0, "skipped": 0}

    existing: set[str] = set()
    if backend.exists(dataset):
        prior = backend.read_dataset(dataset, columns=["_payload_hash"])
        existing = set(prior["_payload_hash"].tolist())

    fresh = df[~df["_payload_hash"].isin(existing)].drop_duplicates(subset=["_payload_hash"])
    if not fresh.empty:
        backend.write_fragment(dataset, fresh)
    return {"received": len(df), "new": len(fresh), "skipped": len(df) - len(fresh)}
