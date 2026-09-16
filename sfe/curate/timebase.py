"""Europe/Rome <-> UTC settlement-period arithmetic (PRD DQ-3).

Store UTC internally, present Europe/Rome. DST days are handled explicitly:
- 25-hour day (autumn fall-back): 25 hourly / 100 quarter-hour periods
- 23-hour day (spring forward): 23 hourly / 92 quarter-hour periods

The safe construction: local midnight of day D and of day D+1 are both unambiguous
instants. Convert each to UTC; the settlement grid for day D is
``[utc_midnight_D, utc_midnight_D + res, ... , utc_midnight_{D+1})``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

ROME = ZoneInfo("Europe/Rome")

_RESOLUTIONS = {
    "PT1H": timedelta(hours=1),
    "PT15M": timedelta(minutes=15),
}


def resolution_delta(resolution: str) -> timedelta:
    try:
        return _RESOLUTIONS[resolution]
    except KeyError:
        raise ValueError(
            f"unsupported resolution {resolution!r}; expected one of {list(_RESOLUTIONS)}"
        ) from None


def _local_midnight_utc(day: date) -> datetime:
    """UTC instant of 00:00 Europe/Rome on *day* (unambiguous)."""
    return datetime(day.year, day.month, day.day, tzinfo=ROME).astimezone(UTC)


def day_grid(day: date, resolution: str) -> list[datetime]:
    """UTC period-start instants covering the Europe/Rome calendar *day*."""
    step = resolution_delta(resolution)
    start = _local_midnight_utc(day)
    end = _local_midnight_utc(day + timedelta(days=1))
    out: list[datetime] = []
    cur = start
    while cur < end:
        out.append(cur)
        cur += step
    return out


def periods_in_day(day: date, resolution: str) -> int:
    """23/24/25 for PT1H, 92/96/100 for PT15M, depending on DST."""
    return len(day_grid(day, resolution))


def period_start_utc(day: date, period_index: int | None, resolution: str) -> datetime:
    """UTC start of the ``period_index``-th (1-based) settlement period of *day*.

    ``period_index=None`` means daily grain and returns 00:00 Europe/Rome of *day* in UTC.
    """
    if period_index is None:
        return _local_midnight_utc(day)
    grid = day_grid(day, resolution)
    if not 1 <= period_index <= len(grid):
        raise ValueError(
            f"period_index {period_index} out of range 1..{len(grid)} for {day} @ {resolution}"
        )
    return grid[period_index - 1]


def rome_wall_to_utc(wall: datetime, *, fold: int = 0) -> datetime:
    """Interpret a naive wall-clock datetime as Europe/Rome and return UTC.

    During the autumn fall-back hour the wall time is ambiguous; ``fold=0`` selects the
    first (pre-transition, still summer-time) occurrence, ``fold=1`` the second.
    """
    if wall.tzinfo is not None:
        raise ValueError("expected a naive datetime")
    return wall.replace(tzinfo=ROME, fold=fold).astimezone(UTC)


def utc_to_rome(instant: datetime) -> datetime:
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    return instant.astimezone(ROME)
