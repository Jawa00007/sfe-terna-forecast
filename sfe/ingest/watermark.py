"""Per-endpoint high-water marks for incremental pulls (PRD F-1).

Stored as a single JSON file under the storage root. A watermark is the latest settlement
date (Europe/Rome, ``yyyy-mm-dd``) for which a pull has completed; the next pull starts the
day after. It is a scheduling hint only - correctness comes from the append-only,
idempotent landing writer, not from this file.
"""

from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path

from sfe.config import get_settings

_LOCK = threading.Lock()


def _path() -> Path:
    return get_settings().storage.root / "_watermarks.json"


def _load() -> dict[str, str]:
    p = _path()
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def get_watermark(key: str) -> date | None:
    raw = _load().get(key)
    return date.fromisoformat(raw) if raw else None


def set_watermark(key: str, value: date) -> None:
    with _LOCK:
        data = _load()
        cur = data.get(key)
        if cur is None or date.fromisoformat(cur) < value:
            data[key] = value.isoformat()
            p = _path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def all_watermarks() -> dict[str, str]:
    return _load()
