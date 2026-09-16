"""DuckDB query layer over the Parquet fragment store.

Thin wrapper: registers one view per dataset pointing at its fragment glob, so the curate
and feature layers can express point-in-time joins in SQL without loading everything into
pandas first.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

from sfe.config import get_settings
from sfe.io.storage import StorageBackend, get_storage


class Catalog:
    def __init__(self, backend: StorageBackend | None = None, db_path: Path | None = None):
        self.backend = backend or get_storage()
        self.db_path = db_path or get_settings().storage.catalog_path

    @contextmanager
    def connect(self, read_only: bool = False):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(self.db_path), read_only=read_only)
        try:
            con.execute("SET TimeZone='UTC'")
            yield con
        finally:
            con.close()

    def register(
        self, con: duckdb.DuckDBPyConnection, dataset: str, view: str | None = None
    ) -> str:
        """Create/replace a view named *view* (default: dataset with '/' -> '__')."""
        view = view or dataset.replace("/", "__").replace("-", "_")
        # DuckDB can't bind a prepared parameter inside CREATE VIEW / read_parquet, so the
        # glob is inlined. It is derived from our own storage root, not user input; still,
        # reject the only character that could break out of the quoting.
        uri = self.backend.dataset_uri(dataset)
        if "'" in uri:
            raise ValueError(f"unsupported character in dataset uri: {uri!r}")
        con.execute(
            f"CREATE OR REPLACE VIEW {view} AS "
            f"SELECT * FROM read_parquet('{uri}', union_by_name=true, filename=true)"
        )
        return view

    def query(self, sql: str, datasets: dict[str, str] | None = None) -> pd.DataFrame:
        """Run *sql*, first registering ``{view_name: dataset}`` mappings if given."""
        with self.connect(read_only=False) as con:
            for view, dataset in (datasets or {}).items():
                self.register(con, dataset, view=view)
            return con.execute(sql).df()
