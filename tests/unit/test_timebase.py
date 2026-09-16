from datetime import date, datetime, timedelta

import pytest

from sfe.curate.timebase import (
    day_grid,
    period_start_utc,
    periods_in_day,
    rome_wall_to_utc,
    utc_to_rome,
)

# Italy 2025 DST: spring forward 30 Mar (23h day), fall back 26 Oct (25h day).
SPRING = date(2025, 3, 30)
AUTUMN = date(2025, 10, 26)
NORMAL = date(2025, 6, 15)


@pytest.mark.parametrize(
    "day, hourly, quarterly",
    [(NORMAL, 24, 96), (SPRING, 23, 92), (AUTUMN, 25, 100)],
)
def test_periods_in_day_handles_dst(day, hourly, quarterly):
    assert periods_in_day(day, "PT1H") == hourly
    assert periods_in_day(day, "PT15M") == quarterly


def test_day_grid_is_contiguous_and_sorted():
    grid = day_grid(AUTUMN, "PT1H")
    assert grid == sorted(grid)
    assert all(b - a == timedelta(hours=1) for a, b in zip(grid, grid[1:], strict=False))
    # 25 hourly slots => spans 25h of real time
    assert grid[-1] + timedelta(hours=1) - grid[0] == timedelta(hours=25)


def test_period_start_index_bounds():
    assert period_start_utc(NORMAL, 1, "PT1H") == day_grid(NORMAL, "PT1H")[0]
    with pytest.raises(ValueError):
        period_start_utc(SPRING, 24, "PT1H")  # only 23 periods that day
    # daily grain
    assert period_start_utc(NORMAL, None, "PT1H") == day_grid(NORMAL, "PT1H")[0]


def test_rome_wall_ambiguous_hour_fold():
    # 02:30 local occurs twice on the fall-back night.
    wall = datetime(2025, 10, 26, 2, 30)
    first = rome_wall_to_utc(wall, fold=0)
    second = rome_wall_to_utc(wall, fold=1)
    assert second - first == timedelta(hours=1)


def test_roundtrip_utc_rome():
    inst = period_start_utc(NORMAL, 10, "PT1H")
    assert utc_to_rome(inst).tzinfo is not None
    assert utc_to_rome(inst).hour == 9  # 10th hourly period starts at 09:00 local
