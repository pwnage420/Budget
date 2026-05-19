"""Wide-format CSV exporters designed for direct paste-back into the v4 Excel model.

Two writers:
- write_budget_wide(): one row per FY week, columns: Year, Week, P20, P50, P80.
  Pastes straight into v4's existing column layout.
- write_history_wide(): the existing 'Source Data' formatted back into the same
  Week,2021,...,2025 layout the v4 model uses.

Same formula-injection guard as excel_writer.py.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .excel_writer import _sanitize  # share the injection-defuse logic


def write_budget_wide(forecast: pd.DataFrame, path: str | Path,
                       columns: tuple[str, ...] = ("P20", "P50", "P80")
                       ) -> Path:
    """Write a long-then-wide budget CSV.

    `forecast` columns expected: date, mean, p20, p30, p70, p80.
    Output: one row per week with the requested band columns. Layout suits
    direct paste into the v4 Excel budget where each row is one ISO week.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fc = forecast.copy()
    fc["Year"] = pd.to_datetime(fc["date"]).dt.isocalendar().year.astype(int)
    fc["Week"] = pd.to_datetime(fc["date"]).dt.isocalendar().week.astype(int)
    rename = {"mean": "P50", "p20": "P20", "p30": "P30", "p70": "P70", "p80": "P80"}
    fc = fc.rename(columns=rename)
    out_cols = ["Year", "Week", *columns]
    fc = fc[out_cols].round(2)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([_sanitize(c) for c in out_cols])
        for _, row in fc.iterrows():
            writer.writerow([_sanitize(row[c]) for c in out_cols])
    return p


def write_history_wide(history: pd.DataFrame, path: str | Path) -> Path:
    """Write history back into the v4 'Week,2021,...,2025' wide layout.

    `history` columns: date, year, week, price.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    wide = (history.pivot(index="week", columns="year", values="price")
                   .rename_axis(index="Week"))
    wide.columns = [str(int(c)) for c in wide.columns]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Week"] + list(wide.columns))
        for week, row in wide.round(2).iterrows():
            writer.writerow([int(week)] + [_sanitize(row[c]) for c in wide.columns])
    return p
