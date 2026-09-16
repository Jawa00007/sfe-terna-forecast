"""Drift + staleness monitoring (PRD F-10 drift detection, NF-3 graceful degradation)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from sfe.io.storage import StorageBackend, get_storage


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples. >0.25 is a strong drift signal."""
    e = np.asarray(expected, dtype="float64")
    a = np.asarray(actual, dtype="float64")
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if e.size < bins or a.size < bins:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    e_pct = np.histogram(e, bins=edges)[0] / e.size
    a_pct = np.histogram(a, bins=edges)[0] / a.size
    eps = 1e-6
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, feature_names: list[str],
    *, warn: float = 0.1, alert: float = 0.25,
) -> pd.DataFrame:
    rows = []
    for f in feature_names:
        if f not in reference.columns or f not in current.columns:
            continue
        val = psi(
            pd.to_numeric(reference[f], errors="coerce").to_numpy(),
            pd.to_numeric(current[f], errors="coerce").to_numpy(),
        )
        level = "ok"
        if val == val:
            level = "alert" if val >= alert else ("warn" if val >= warn else "ok")
        rows.append({"feature": f, "psi": val, "level": level})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")


def staleness_report(
    datasets: list[str],
    *,
    as_of: datetime | None = None,
    max_age_hours: float = 30.0,
    storage: StorageBackend | None = None,
) -> pd.DataFrame:
    storage = storage or get_storage()
    as_of = as_of or datetime.now(UTC)
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    rows = []
    for ds in datasets:
        raw = storage.read_dataset(ds, columns=["_publication_ts"]) if storage.exists(ds) else None
        if raw is None or raw.empty:
            rows.append({"dataset": ds, "latest_publication": None, "age_hours": np.inf,
                         "stale": True})
            continue
        latest = pd.to_datetime(raw["_publication_ts"], utc=True).max()
        age = (as_of - latest).total_seconds() / 3600.0
        rows.append(
            {
                "dataset": ds,
                "latest_publication": latest.isoformat(),
                "age_hours": round(age, 2),
                "stale": bool(age > max_age_hours),
            }
        )
    return pd.DataFrame(rows)
