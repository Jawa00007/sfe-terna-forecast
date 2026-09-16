"""Terna Developer Portal client: OAuth2 client-credentials + resilient GET.

Known quirks handled here (PRD 6.1):
- numeric fields arrive as JSON strings -> left as-is; parsing is the curate layer's job
- date params are ``dateFrom`` / ``dateTo`` formatted ``dd/mm/yyyy``
- optional ``dataType`` query param: ``Orario`` | ``Quarto Orario``
- request windows are chunked to ``max_window_days``
- response body is an envelope ``{"<SomeKey>": [ ...records... ]}``; the key varies by
  endpoint, so record extraction is tolerant.

No Terna sandbox exists: point ``SFE_TERNA__BASE_URL`` / ``SFE_TERNA__TOKEN_URL`` at the
local mock server (:mod:`sfe.testing.mock_terna`) for offline development.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from sfe.config import TernaSettings, get_settings
from sfe.ingest.endpoints import EndpointSpec, get_endpoint

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return False


@dataclass
class _Token:
    value: str
    expires_at: float  # unix epoch seconds


class TernaAuth:
    """Caches the client-credentials token and refreshes it before expiry."""

    def __init__(self, settings: TernaSettings, client: httpx.Client):
        self._s = settings
        self._client = client
        self._token: _Token | None = None
        self._lock = threading.Lock()

    def _fetch_token(self) -> _Token:
        resp = self._client.post(
            self._s.token_url,
            data={"grant_type": "client_credentials"},
            auth=(self._s.client_id, self._s.client_secret),
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        body = resp.json()
        ttl = int(body.get("expires_in", 3600))
        return _Token(
            value=body["access_token"],
            expires_at=time.time() + ttl - self._s.token_refresh_margin_s,
        )

    def bearer(self) -> str:
        if not self._s.configured:
            raise RuntimeError(
                "Terna credentials not configured. Set SFE_TERNA__CLIENT_ID / "
                "SFE_TERNA__CLIENT_SECRET, or point SFE_TERNA__* at the mock server."
            )
        with self._lock:
            if self._token is None or time.time() >= self._token.expires_at:
                self._token = self._fetch_token()
            return self._token.value


def _extract_records(body: Any) -> list[dict[str, Any]]:
    """Pull the record list out of Terna's response envelope, tolerantly."""
    if isinstance(body, list):
        return [r for r in body if isinstance(r, dict)]
    if isinstance(body, dict):
        # common wrappers
        for key in ("result", "results", "data", "Data"):
            if isinstance(body.get(key), list):
                return [r for r in body[key] if isinstance(r, dict)]
        # otherwise: the single list-valued entry is the payload
        list_vals = [v for v in body.values() if isinstance(v, list)]
        if len(list_vals) == 1:
            return [r for r in list_vals[0] if isinstance(r, dict)]
    return []


def _windows(dfrom: date, dto: date, max_days: int) -> Iterable[tuple[date, date]]:
    cur = dfrom
    step = timedelta(days=max_days - 1)
    while cur <= dto:
        end = min(cur + step, dto)
        yield cur, end
        cur = end + timedelta(days=1)


class TernaClient:
    def __init__(self, settings: TernaSettings | None = None, client: httpx.Client | None = None):
        self._s = settings or get_settings().terna
        self._client = client or httpx.Client(
            base_url=self._s.base_url, timeout=self._s.request_timeout_s
        )
        self._auth = TernaAuth(self._s, self._client)

    # -- low level -----------------------------------------------------------------
    def _get_raw(self, path: str, params: dict[str, Any]) -> Any:
        @retry(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential_jitter(
                initial=self._s.backoff_base_s, max=self._s.backoff_max_s
            ),
            stop=stop_after_attempt(self._s.max_retries),
            reraise=True,
        )
        def _do() -> Any:
            resp = self._client.get(
                path,
                params=params,
                headers={
                    "Authorization": f"Bearer {self._auth.bearer()}",
                    "Accept": "application/json",
                },
            )
            resp.raise_for_status()
            return resp.json()

        return _do()

    # -- endpoint level ----------------------------------------------------------
    def fetch(
        self,
        endpoint: str | EndpointSpec,
        date_from: date,
        date_to: date,
        *,
        data_type: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch every record for ``[date_from, date_to]``, chunked by the endpoint window.

        Records are returned exactly as received (numerics still strings) plus a
        ``_fetched_at`` UTC ISO timestamp for provenance.
        """
        spec = endpoint if isinstance(endpoint, EndpointSpec) else get_endpoint(endpoint)
        if data_type and not spec.supports_dataType:
            raise ValueError(f"{spec.name!r} does not accept dataType")
        pfrom = spec.date_param_names.get("from", "dateFrom")
        pto = spec.date_param_names.get("to", "dateTo")

        out: list[dict[str, Any]] = []
        for w_from, w_to in _windows(date_from, date_to, spec.max_window_days):
            params: dict[str, Any] = {
                pfrom: _fmt_date(w_from, spec.date_param_style),
                pto: _fmt_date(w_to, spec.date_param_style),
            }
            if data_type:
                params["dataType"] = data_type
            if extra_params:
                params.update(extra_params)
            body = self._get_raw(spec.path, params)
            fetched_at = datetime.now(UTC).isoformat()
            for rec in _extract_records(body):
                rec = dict(rec)
                rec.setdefault("_fetched_at", fetched_at)
                out.append(rec)
        return out

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TernaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _fmt_date(d: date, style: str) -> str:
    if style == "dd_mm_yyyy":
        return d.strftime("%d/%m/%Y")
    if style == "iso":
        return d.isoformat()
    raise ValueError(f"unknown date_param_style: {style!r}")
