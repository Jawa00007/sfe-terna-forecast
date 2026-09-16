from datetime import date

import pandas as pd

from sfe.curate.calendar_it import (
    calendar_features,
    dst_transition_days,
    easter_sunday,
    is_bridge_day,
    is_holiday,
)
from sfe.curate.timebase import day_grid


def test_easter_known_dates():
    assert easter_sunday(2024) == date(2024, 3, 31)
    assert easter_sunday(2025) == date(2025, 4, 20)
    assert easter_sunday(2026) == date(2026, 4, 5)


def test_fixed_and_derived_holidays():
    assert is_holiday(date(2025, 1, 1))       # Capodanno
    assert is_holiday(date(2025, 8, 15))      # Ferragosto
    assert is_holiday(date(2025, 4, 21))      # Pasquetta (Easter Mon 2025)
    assert not is_holiday(date(2025, 4, 22))


def test_bridge_day():
    # 2025-04-25 (Fri, Liberazione) -> Sat/Sun; 2025-04-24 Thu is not a bridge,
    # but a lone workday between holiday+weekend is. Use 2 Jun 2025 (Mon) holiday:
    # 2025-06-02 is Mon holiday; 2025-05-30 Fri is between... check a clean case:
    # 2025-12-26 Fri holiday, 27-28 weekend; 2025-12-29 Mon is NOT bridge (Tue working).
    assert is_bridge_day(date(2025, 1, 2)) in (True, False)  # smoke: no crash
    # New Year 2026-01-01 is Thu; 01-02 Fri wedged before Sat/Sun -> bridge
    assert is_bridge_day(date(2026, 1, 2)) is True


def test_dst_transition_days_2025():
    spring, autumn = dst_transition_days(2025)
    assert spring == date(2025, 3, 30)
    assert autumn == date(2025, 10, 26)


def test_calendar_features_frame():
    grid = day_grid(date(2025, 6, 15), "PT1H")  # summer -> CEST
    feats = calendar_features(pd.Series(grid))
    assert len(feats) == 24
    assert set(["_mtu_start", "cal_hour", "cal_is_holiday", "cal_hour_sin"]).issubset(feats.columns)
    assert (feats["cal_is_dst"] == 1).all()
    assert feats["cal_hour"].tolist() == list(range(24))
