"""Ingestion orchestration (PRD F-1).

Plain functions so they run without Prefect installed; when the ``orchestration`` extra is
present they are also exposed as Prefect tasks/flows for scheduling, retries and logging.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from sfe.config import load_yaml
from sfe.curate import regimes
from sfe.curate.timebase import period_start_utc
from sfe.ingest.endpoints import EndpointSpec, get_endpoint, iter_endpoints
from sfe.ingest.landing import build_cells, land
from sfe.ingest.terna_client import TernaClient
from sfe.ingest.watermark import get_watermark, set_watermark


def _target_area_level() -> str:
    return load_yaml("zones").get("target_level", "macrozone")


def ingest_endpoint(
    endpoint: str | EndpointSpec,
    date_from: date,
    date_to: date,
    *,
    client: TernaClient | None = None,
    data_type: str | None = None,
    update_watermark: bool = True,
) -> dict:
    """Pull ``[date_from, date_to]`` for one endpoint and land it, vintaged and idempotent."""
    spec = endpoint if isinstance(endpoint, EndpointSpec) else get_endpoint(endpoint)
    if data_type is None and spec.supports_dataType:
        data_type = regimes.terna_data_type(date_from)

    owns_client = client is None
    client = client or TernaClient()
    try:
        records = client.fetch(spec, date_from, date_to, data_type=data_type or None)
    finally:
        if owns_client:
            client.close()

    cells = build_cells(
        spec,
        records,
        period_start_resolver=period_start_utc,
        default_area_level=_target_area_level(),
        ingested_at=datetime.now(UTC),
    )
    stats = land(spec.dataset, cells)
    stats |= {"endpoint": spec.name, "records": len(records), "cells": len(cells)}

    if update_watermark and stats["new"] >= 0:
        set_watermark(f"terna/{spec.group}/{spec.name}", date_to)
    return stats


def backfill_endpoint(
    endpoint: str,
    start: date,
    end: date,
    *,
    client: TernaClient | None = None,
    resume: bool = True,
) -> dict:
    spec = get_endpoint(endpoint)
    if resume:
        wm = get_watermark(f"terna/{spec.group}/{spec.name}")
        if wm and wm >= start:
            start = wm  # re-pull the watermark day; landing dedupes
    return ingest_endpoint(spec, start, end, client=client)


def ingest_targets(
    date_from: date, date_to: date, *, client: TernaClient | None = None
) -> list[dict]:
    """Sign, price and charge endpoints - the label set (PRD 6.1)."""
    owns = client is None
    client = client or TernaClient()
    try:
        out = []
        for spec in iter_endpoints():
            if spec.role in {"target_sign", "target_price", "charge"}:
                out.append(ingest_endpoint(spec, date_from, date_to, client=client))
        return out
    finally:
        if owns:
            client.close()


def as_prefect_flow():  # pragma: no cover - needs the `orchestration` extra
    """Wrap :func:`ingest_targets` in a Prefect flow for scheduled deployment.

    Kept out of import time so the module stays usable without Prefect. Call this from a
    deployment script once ``pip install '.[orchestration]'`` is done.
    """
    from prefect import flow

    return flow(name="sfe-ingest-targets", log_prints=True)(ingest_targets)
