"""Audit log of every recommendation (PRD F-17, NF-4).

Append-only Parquet fragments under ``data/audit/``. One row per (area, MTU, as_of) with
the full input feature vector (JSON), the recommendation, and the model version - enough to
reproduce or explain any past decision for debugging or a regulatory query.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sfe.config import get_settings

_AUDIT_COLS = [
    "audit_id", "recorded_at", "delivery_day", "horizon", "as_of",
    "_area", "_mtu_start", "delta_mwh", "abstain", "reason", "sign_prob",
    "e_spread", "sigma_spread", "charge_eur_per_mwh", "load_mwh",
    "expected_net_margin_eur", "cvar_eur_per_mwh", "limit_binding",
    "util_pct_load_cap", "util_daily_cap", "degraded", "model_version", "features_json",
]


class AuditLog:
    def __init__(self, root: Path | None = None):
        self.dir = Path(root or get_settings().storage.root / "audit")

    def _fragments(self) -> list[Path]:
        return sorted(self.dir.glob("part-*.parquet")) if self.dir.exists() else []

    def record(self, bundle, features: pd.DataFrame | None = None) -> list[str]:
        """Persist one row per recommendation in *bundle*. Returns the audit ids."""
        tbl = bundle.table
        if tbl.empty:
            return []
        feat_lut: dict = {}
        if features is not None:
            f = features.copy()
            f["_key_ts"] = pd.to_datetime(f["_mtu_start"], utc=True).map(lambda t: t.isoformat())
            for _, r in f.iterrows():
                feat_lut[(r["_area"], r["_key_ts"])] = json.dumps(
                    {
                        k: _jsonable(v)
                        for k, v in r.items()
                        if not k.startswith("_") and k != "_key_ts"
                    },
                    sort_keys=True,
                )

        mv = json.dumps(bundle.model_version, sort_keys=True, default=str)
        now = datetime.now(UTC).isoformat()
        rows = []
        for _, r in tbl.iterrows():
            mtu = pd.to_datetime(r["_mtu_start"], utc=True)
            rows.append(
                {
                    "audit_id": uuid.uuid4().hex,
                    "recorded_at": now,
                    "delivery_day": bundle.delivery_day.isoformat(),
                    "horizon": bundle.horizon,
                    "as_of": pd.Timestamp(bundle.as_of).isoformat(),
                    "_area": r["_area"],
                    "_mtu_start": mtu.isoformat(),
                    "delta_mwh": _f(r.get("delta_mwh")),
                    "abstain": bool(r.get("abstain", False)),
                    "reason": r.get("reason", ""),
                    "sign_prob": _f(r.get("sign_prob")),
                    "e_spread": _f(r.get("e_spread")),
                    "sigma_spread": _f(r.get("sigma_spread")),
                    "charge_eur_per_mwh": _f(r.get("charge_eur_per_mwh")),
                    "load_mwh": _f(r.get("load_mwh")),
                    "expected_net_margin_eur": _f(r.get("expected_net_margin_eur")),
                    "cvar_eur_per_mwh": _f(r.get("cvar_eur_per_mwh")),
                    "limit_binding": r.get("limit_binding"),
                    "util_pct_load_cap": _f(r.get("util_pct_load_cap")),
                    "util_daily_cap": _f(r.get("util_daily_cap")),
                    "degraded": bool(bundle.degraded),
                    "model_version": mv,
                    "features_json": feat_lut.get((r["_area"], mtu.isoformat()), ""),
                }
            )
        df = pd.DataFrame(rows, columns=_AUDIT_COLS)
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        df.to_parquet(self.dir / f"part-{stamp}-{uuid.uuid4().hex[:6]}.parquet", index=False)
        return df["audit_id"].tolist()

    def read(self) -> pd.DataFrame:
        parts = self._fragments()
        if not parts:
            return pd.DataFrame(columns=_AUDIT_COLS)
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

    def get(self, audit_id: str) -> dict | None:
        df = self.read()
        hit = df[df["audit_id"] == audit_id]
        return None if hit.empty else hit.iloc[0].to_dict()

    def checksum(self) -> str:
        df = self.read().sort_values("audit_id")
        return hashlib.sha1(
            pd.util.hash_pandas_object(df, index=False).values.tobytes()
        ).hexdigest()


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _jsonable(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if pd.isna(v):
        return None
    return str(v)
