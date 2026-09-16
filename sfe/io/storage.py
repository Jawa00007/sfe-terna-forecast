"""Append-only, vintaged storage (PRD DQ-1, DQ-2, F-3).

A *dataset* is an ordered collection of immutable Parquet *fragments*. Writes only ever
append a new fragment; nothing is updated or deleted. Point-in-time correctness is a
read-time concern (see :mod:`sfe.curate.vintage`), not enforced here - this layer's single
job is to never lose a vintage.

The :class:`StorageBackend` interface is deliberately tiny so an S3/object-store
implementation can drop in behind it later without touching callers.
"""

from __future__ import annotations

import abc
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sfe.config import get_settings

# Metadata columns every landed record carries. Kept as a module constant so the curate
# and feature layers can rely on them and tests can assert the contract.
META_COLUMNS: tuple[str, ...] = (
    "_source",
    "_endpoint",
    "_area",
    "_area_level",
    "_mtu_start",       # tz-aware UTC: start of the settlement period
    "_resolution",      # ISO 8601 duration, e.g. 'PT1H' / 'PT15M'
    "_vintage",         # 'preliminary' | 'definitive'
    "_publication_ts",  # tz-aware UTC: when the source published this value
    "_ingested_at",     # tz-aware UTC: when we pulled it
    "_payload_hash",    # sha1 of the raw source record, for idempotency
    "_raw",             # raw source record, JSON-encoded string (raw payload retention)
)


class StorageBackend(abc.ABC):
    """Minimal append-only fragment store."""

    @abc.abstractmethod
    def write_fragment(self, dataset: str, df: pd.DataFrame) -> str:
        """Append *df* as a new immutable fragment. Returns the fragment id."""

    @abc.abstractmethod
    def read_dataset(self, dataset: str, columns: list[str] | None = None) -> pd.DataFrame:
        """Concatenate every fragment of *dataset*. Empty frame if the dataset is absent."""

    @abc.abstractmethod
    def list_fragments(self, dataset: str) -> list[str]:
        ...

    @abc.abstractmethod
    def dataset_uri(self, dataset: str) -> str:
        """A DuckDB-readable URI/glob covering all fragments of *dataset*."""

    @abc.abstractmethod
    def exists(self, dataset: str) -> bool:
        ...


class LocalParquetBackend(StorageBackend):
    """Fragments as ``<root>/<dataset>/part-<utc-stamp>-<uuid>.parquet``."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _dir(self, dataset: str) -> Path:
        return self.root / dataset

    def write_fragment(self, dataset: str, df: pd.DataFrame) -> str:
        missing = [c for c in META_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"fragment for {dataset!r} is missing metadata columns: {missing}")
        if df.empty:
            raise ValueError(f"refusing to write an empty fragment for {dataset!r}")

        target = self._dir(dataset)
        target.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        frag_id = f"part-{stamp}-{uuid.uuid4().hex[:8]}"
        df.to_parquet(target / f"{frag_id}.parquet", index=False)
        return frag_id

    def list_fragments(self, dataset: str) -> list[str]:
        d = self._dir(dataset)
        if not d.exists():
            return []
        return sorted(p.name for p in d.glob("part-*.parquet"))

    def read_dataset(self, dataset: str, columns: list[str] | None = None) -> pd.DataFrame:
        d = self._dir(dataset)
        parts = sorted(d.glob("part-*.parquet")) if d.exists() else []
        if not parts:
            return pd.DataFrame(columns=list(columns) if columns else None)
        frames = [pd.read_parquet(p, columns=columns) for p in parts]
        return pd.concat(frames, ignore_index=True)

    def dataset_uri(self, dataset: str) -> str:
        return str((self._dir(dataset) / "part-*.parquet").as_posix())

    def exists(self, dataset: str) -> bool:
        return bool(self.list_fragments(dataset))


def get_storage() -> StorageBackend:
    s = get_settings().storage
    if s.backend == "local":
        return LocalParquetBackend(s.root)
    raise ValueError(f"unknown storage backend: {s.backend!r}")
