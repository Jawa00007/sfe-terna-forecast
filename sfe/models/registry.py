"""Model persistence + experiment tracking (PRD F-10, NF-4).

joblib for the artefact; MLflow for run metadata when the ``models`` extra is installed
(otherwise ``log_run`` writes a JSON sidecar so nothing is lost).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from sfe.config import get_settings


def _models_dir() -> Path:
    d = get_settings().storage.root / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_model(obj: Any, name: str, *, meta: dict[str, Any] | None = None) -> Path:
    path = _models_dir() / f"{name}.joblib"
    joblib.dump(obj, path)
    meta = dict(meta or {})
    meta.setdefault("saved_at", datetime.now(UTC).isoformat())
    (_models_dir() / f"{name}.meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return path


def load_model(name: str) -> Any:
    return joblib.load(_models_dir() / f"{name}.joblib")


def load_model_meta(name: str) -> dict[str, Any] | None:
    p = _models_dir() / f"{name}.meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def model_exists(name: str) -> bool:
    return (_models_dir() / f"{name}.joblib").exists()


def log_run(
    name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    *,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    record = {
        "name": name,
        "logged_at": datetime.now(UTC).isoformat(),
        "params": params,
        "metrics": metrics,
        "tags": tags or {},
    }
    try:  # pragma: no cover - only when mlflow installed
        import mlflow

        mlflow.set_experiment("sfe")
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
            for k, v in (tags or {}).items():
                mlflow.set_tag(k, v)
        record["backend"] = "mlflow"
    except ImportError:
        runs = _models_dir() / "runs.jsonl"
        with runs.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        record["backend"] = "jsonl"
    return record


def read_runs(name_prefix: str | None = None) -> pd.DataFrame:
    """The JSONL run log (empty if mlflow is the active backend - query mlflow instead)."""
    path = _models_dir() / "runs.jsonl"
    if not path.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if name_prefix:
        records = [r for r in records if str(r.get("name", "")).startswith(name_prefix)]
    if not records:
        return pd.DataFrame()
    rows = []
    for r in records:
        row = {"name": r["name"], "logged_at": r["logged_at"]}
        row.update({f"param.{k}": v for k, v in r.get("params", {}).items()})
        row.update({f"metric.{k}": v for k, v in r.get("metrics", {}).items()})
        row.update({f"tag.{k}": v for k, v in r.get("tags", {}).items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("logged_at", ascending=False).reset_index(drop=True)
