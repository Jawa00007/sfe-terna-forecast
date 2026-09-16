"""Typed access to the Terna endpoint catalogue (``conf/endpoints.yaml``)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from sfe.config import load_yaml


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    group: str
    path: str                       # full path incl. group base_path, e.g. /fees/v1.0/daily-prices
    role: str                       # target_sign | target_price | charge | feature
    grain: tuple[str, ...]          # e.g. ('macrozone', 'period')
    vintage: str | None             # 'preliminary' | 'definitive' | None
    key_fields: tuple[str, ...]
    value_fields: tuple[str, ...]
    date_param_style: str
    date_param_names: dict[str, str]
    numeric_as_string: bool
    supports_dataType: bool
    max_window_days: int
    timezone: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def dataset(self) -> str:
        """Landing-zone dataset path for this endpoint."""
        return f"landing/terna/{self.group}/{self.name}"


def _catalogue() -> dict[str, Any]:
    return load_yaml("endpoints")


def _build(name: str, group: str, gcfg: dict[str, Any], ecfg: dict[str, Any]) -> EndpointSpec:
    d = _catalogue().get("defaults", {})
    base = gcfg.get("base_path", "")
    return EndpointSpec(
        name=name,
        group=group,
        path=f"{base}{ecfg['path']}",
        role=ecfg.get("role", "feature"),
        grain=tuple(ecfg.get("grain", [])),
        vintage=ecfg.get("vintage"),
        key_fields=tuple(ecfg.get("key_fields", [])),
        value_fields=tuple(ecfg.get("value_fields", [])),
        date_param_style=ecfg.get("date_param_style", d.get("date_param_style", "dd_mm_yyyy")),
        date_param_names=ecfg.get("date_param_names", d.get("date_param_names", {})),
        numeric_as_string=ecfg.get("numeric_as_string", d.get("numeric_as_string", True)),
        supports_dataType=ecfg.get("supports_dataType", d.get("supports_dataType", True)),
        max_window_days=ecfg.get("max_window_days", d.get("max_window_days", 60)),
        timezone=ecfg.get("timezone", d.get("timezone", "Europe/Rome")),
        raw=ecfg,
    )


def get_endpoint(name: str) -> EndpointSpec:
    for group, gcfg in _catalogue().get("groups", {}).items():
        endpoints = gcfg.get("endpoints", {})
        if name in endpoints:
            return _build(name, group, gcfg, endpoints[name])
    raise KeyError(f"unknown Terna endpoint: {name!r}")


def iter_endpoints(*, group: str | None = None, role: str | None = None) -> Iterator[EndpointSpec]:
    for gname, gcfg in _catalogue().get("groups", {}).items():
        if group and gname != group:
            continue
        for ename, ecfg in gcfg.get("endpoints", {}).items():
            spec = _build(ename, gname, gcfg, ecfg)
            if role and spec.role != role:
                continue
            yield spec


def target_endpoints() -> list[EndpointSpec]:
    """The endpoints that define the label + the charge (PRD 6.1)."""
    return [e for e in iter_endpoints() if e.role in {"target_sign", "target_price", "charge"}]
