"""Italian market calendar (PRD 6.2 "Calendar", F-5 calendar family).

National public holidays + DST structure, computed offline. Regional patron-saint days and
school calendars are a documented TODO (need a per-zone mapping / vendor feed).

Calendar features are known arbitrarily far ahead, so in the point-in-time feature store
their ``publication_ts`` is effectively -infinity - they are always visible.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from functools import lru_cache

import pandas as pd

from sfe.curate.timebase import ROME, utc_to_rome

# Fixed-date national public holidays (month, day).
_FIXED = {
    (1, 1): "capodanno",
    (1, 6): "epifania",
    (4, 25): "liberazione",
    (5, 1): "festa_lavoro",
    (6, 2): "festa_repubblica",
    (8, 15): "ferragosto",
    (11, 1): "ognissanti",
    (12, 8): "immacolata",
    (12, 25): "natale",
    (12, 26): "santo_stefano",
}


def easter_sunday(year: int) -> date:
    """Gregorian Easter (Meeus/Jones/Butcher algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lu = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lu) // 451
    month = (h + lu - 7 * m + 114) // 31
    day = ((h + lu - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@lru_cache(maxsize=64)
def holidays(year: int) -> dict[date, str]:
    out = {date(year, m, d): name for (m, d), name in _FIXED.items()}
    out[easter_sunday(year) + timedelta(days=1)] = "pasquetta"
    return out


def is_holiday(day: date) -> bool:
    return day in holidays(day.year)


def holiday_name(day: date) -> str | None:
    return holidays(day.year).get(day)


def _is_nonworking(day: date) -> bool:
    return day.weekday() >= 5 or is_holiday(day)


def is_bridge_day(day: date) -> bool:
    """A lone working day wedged between two non-working days (Italian *ponte*)."""
    if _is_nonworking(day):
        return False
    return _is_nonworking(day - timedelta(days=1)) and _is_nonworking(day + timedelta(days=1))


def _signed_days_to_holiday(day: date, horizon: int = 7) -> tuple[int, int]:
    fwd = next((k for k in range(horizon + 1) if is_holiday(day + timedelta(days=k))), horizon + 1)
    back = next((k for k in range(horizon + 1) if is_holiday(day - timedelta(days=k))), horizon + 1)
    return fwd, back


def calendar_features(mtu_starts_utc: pd.Series | list[datetime]) -> pd.DataFrame:
    """One row per distinct UTC period start, with Europe/Rome-local calendar features."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(mtu_starts_utc), utc=True)).unique().sort_values()
    rows = []
    for ts_utc in idx:
        local = utc_to_rome(ts_utc.to_pydatetime())
        day = local.date()
        fwd, back = _signed_days_to_holiday(day)
        off = local.utcoffset() or timedelta(0)
        rows.append(
            {
                "_mtu_start": ts_utc,
                "cal_hour": local.hour,
                "cal_dow": local.weekday(),
                "cal_month": local.month,
                "cal_doy": int(local.timetuple().tm_yday),
                "cal_is_weekend": int(local.weekday() >= 5),
                "cal_is_holiday": int(is_holiday(day)),
                "cal_is_nonworking": int(_is_nonworking(day)),
                "cal_is_bridge": int(is_bridge_day(day)),
                "cal_days_to_holiday": fwd,
                "cal_days_from_holiday": back,
                "cal_hour_sin": math.sin(2 * math.pi * local.hour / 24),
                "cal_hour_cos": math.cos(2 * math.pi * local.hour / 24),
                "cal_dow_sin": math.sin(2 * math.pi * local.weekday() / 7),
                "cal_dow_cos": math.cos(2 * math.pi * local.weekday() / 7),
                "cal_is_dst": int(off == timedelta(hours=2)),  # CEST
            }
        )
    return pd.DataFrame(rows)


def dst_transition_days(year: int) -> tuple[date, date]:
    """(spring-forward, fall-back) Sundays for Europe/Rome in *year*."""
    march = [date(year, 3, d) for d in range(25, 32)]
    october = [date(year, 10, d) for d in range(25, 32)]
    spring = next(d for d in march if d.weekday() == 6)
    autumn = next(d for d in october if d.weekday() == 6)
    return spring, autumn


def _assert_rome_loaded() -> None:  # pragma: no cover - import-time sanity
    if datetime(2025, 1, 1, tzinfo=ROME).utcoffset() != timedelta(hours=1):
        raise RuntimeError("Europe/Rome tz data not available")
