"""Tests for AU FY week-numbering conventions."""
from __future__ import annotations

import pandas as pd
import pytest

from app.core.fy_calendar import (
    fy_week_table, map_date_to_fy_week,
)


def test_date_in_july_is_new_fy():
    fw = map_date_to_fy_week(pd.Timestamp("2025-08-15"))
    assert fw.fy_label == "FY2026"


def test_date_in_january_belongs_to_current_fy():
    fw = map_date_to_fy_week(pd.Timestamp("2026-02-15"))
    assert fw.fy_label == "FY2026"


def test_calendar_convention_week_1_at_fy_start():
    fw = map_date_to_fy_week(pd.Timestamp("2025-07-01"), convention="calendar")
    assert fw.fy_week == 1
    fw = map_date_to_fy_week(pd.Timestamp("2025-07-08"), convention="calendar")
    assert fw.fy_week == 2


def test_iso_convention_produces_52_weeks():
    table = fy_week_table(fy_start_year=2026, convention="iso")
    assert len(table) == 52
    assert table["fy_week"].iloc[0] == 1
    assert table["fy_week"].iloc[51] == 52


def test_retail_454_produces_52_weeks():
    table = fy_week_table(fy_start_year=2026, convention="retail_454")
    assert len(table) == 52


def test_unknown_convention_raises():
    with pytest.raises(ValueError, match="Unknown FY convention"):
        map_date_to_fy_week(pd.Timestamp("2025-08-15"), convention="bogus")  # type: ignore[arg-type]
