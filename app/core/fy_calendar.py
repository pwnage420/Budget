"""AU Financial Year calendar helpers.

The AU FY runs July-June by default; the v4 Excel may use a different
convention. This module centralises the mapping so the GUI's Settings tab
can swap convention without touching the rest of the code.

Three conventions supported:
- 'iso'        — FY weeks 1..52 are ISO weeks starting from the ISO week
                 containing July 1 (default; aligned with statsmodels'
                 weekly index).
- 'calendar'   — FY weeks 1..52 are calendar weeks counted from July 1
                 (ignores ISO week 53 entirely).
- 'retail_454' — 4-4-5 retail calendar: 13 fiscal months of 4-4-5 weeks
                 (52 weeks total). Common in AU retail accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

FYConvention = Literal["iso", "calendar", "retail_454"]


@dataclass
class FYWeek:
    fy_label: str       # e.g. "FY2026"
    fy_week: int        # 1..52 (or 53 if convention allows)
    iso_year: int
    iso_week: int
    week_start: pd.Timestamp


def map_date_to_fy_week(date: pd.Timestamp, fy_start_month: int = 7,
                         convention: FYConvention = "iso") -> FYWeek:
    """Map a calendar date to an AU FY week.

    Examples (fy_start_month=7, convention='iso'):
        2025-07-07 → FY2026, week 1
        2026-06-29 → FY2026, week 52
        2026-07-06 → FY2027, week 1
    """
    date = pd.Timestamp(date).normalize()
    fy_label, fy_start = _fy_for(date, fy_start_month)
    iso = date.isocalendar()
    if convention == "iso":
        # FY week 1 = ISO week of fy_start; +1 per ISO week thereafter
        fy_start_iso = pd.Timestamp(fy_start).isocalendar()
        diff_weeks = ((iso.year - fy_start_iso.year) * 52
                       + (iso.week - fy_start_iso.week))
        fy_week = ((diff_weeks % 52) + 1) if diff_weeks >= 0 else 53 + (diff_weeks % 52)
    elif convention == "calendar":
        days_since_fy_start = (date - pd.Timestamp(fy_start)).days
        fy_week = (days_since_fy_start // 7) + 1
    elif convention == "retail_454":
        # 4-4-5 pattern: months of 4, 4, 5 weeks repeating; 13 months × 4 = 52.
        days_since_fy_start = (date - pd.Timestamp(fy_start)).days
        weeks = days_since_fy_start // 7
        fy_week = (weeks % 52) + 1
    else:
        raise ValueError(f"Unknown FY convention: {convention}")
    return FYWeek(
        fy_label=fy_label,
        fy_week=int(fy_week),
        iso_year=int(iso.year),
        iso_week=int(iso.week),
        week_start=date - pd.Timedelta(days=date.weekday()),
    )


def _fy_for(date: pd.Timestamp, fy_start_month: int) -> tuple[str, pd.Timestamp]:
    """Return (fy_label, fy_start_date) for the FY this date belongs to.

    For fy_start_month=7, dates in Jan-Jun belong to FY of that calendar year,
    dates in Jul-Dec belong to FY of next calendar year. e.g. 2025-08-15 → FY2026.
    """
    year = date.year
    fy_year = year + 1 if date.month >= fy_start_month else year
    fy_start = pd.Timestamp(year=fy_year - 1, month=fy_start_month, day=1)
    return f"FY{fy_year}", fy_start


def fy_week_table(fy_start_year: int, fy_start_month: int = 7,
                   convention: FYConvention = "iso", horizon: int = 52
                   ) -> pd.DataFrame:
    """Return a horizon-row DataFrame for the FY beginning in `fy_start_year`.

    Always starts at fy_week 1; each convention picks its own anchor Monday:
    - 'iso'        — Monday of the ISO week containing fy-start (e.g. 2025-06-30)
    - 'calendar'   — first Monday on/after fy-start (e.g. 2025-07-07)
    - 'retail_454' — first Monday on/after fy-start (4-4-5 retail anchor)
    """
    fy_start = pd.Timestamp(year=fy_start_year - 1, month=fy_start_month, day=1)
    if convention == "iso":
        anchor = fy_start - pd.Timedelta(days=fy_start.weekday())
    else:
        anchor = fy_start + pd.Timedelta(days=(7 - fy_start.weekday()) % 7)
    rows = []
    for h in range(horizon):
        d = anchor + pd.Timedelta(weeks=h)
        iso = d.isocalendar()
        rows.append({
            "fy_label": f"FY{fy_start_year}",
            "fy_week": h + 1,
            "iso_year": int(iso.year),
            "iso_week": int(iso.week),
            "week_start": d,
        })
    return pd.DataFrame(rows)
