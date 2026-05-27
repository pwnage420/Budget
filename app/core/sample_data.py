"""Synthetic 2021-2025 weekly retail tomato price series + matching regressors.

Documented as synthetic in the README. Designed so:
- Winter (weeks 22-35, AU calendar) shows higher glasshouse-supported prices
- Summer (weeks 1-12, 45-52) shows lower trough prices
- Mild upward trend across 5 years
- Two plausible outliers (e.g. a heatwave spike + a promotional crash)
- Synthetic weather carries a *signed* effect on price so the Regressors demo
  can show measurable backtest improvement when toggled on
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

YEARS = (2021, 2022, 2023, 2024, 2025)
WEEKS = tuple(range(1, 53))
SEED = 42


def _seasonal_shape(week: int) -> float:
    """Aussie tomato seasonality: trough Jan-Mar, peak Jun-Aug."""
    # Phase shift so week 28 (~mid-July) sits at the peak
    return float(0.6 * np.cos(2 * np.pi * (week - 28) / 52))


def _trend(year: int) -> float:
    return 0.05 * (year - 2021)  # +5c/kg/year baseline drift


def generate_price_series(seed: int = SEED) -> pd.DataFrame:
    """Returns long-format DataFrame with columns: date, year, week, price."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in YEARS:
        for week in WEEKS:
            base = 3.40 + _seasonal_shape(week) + _trend(year)
            noise = rng.normal(0, 0.18)
            price = max(0.5, base + noise)
            rows.append({"year": year, "week": week, "price": round(price, 2)})
    df = pd.DataFrame(rows)
    # Plausible outliers
    df.loc[(df["year"] == 2023) & (df["week"] == 8), "price"] = 5.60   # promo crash inverse
    df.loc[(df["year"] == 2024) & (df["week"] == 30), "price"] = 6.40  # winter heatwave spike

    df["date"] = df.apply(
        lambda r: pd.Timestamp.fromisocalendar(int(r["year"]), int(r["week"]), 1), axis=1
    )
    return df[["date", "year", "week", "price"]]


def generate_regressors(seed: int = SEED) -> pd.DataFrame:
    """Synthetic weekly exogenous regressors aligned to the price series.

    - max_temp_c: realistic Tongala seasonal profile + noise, signed price effect
    - rainfall_mm: weak negative effect
    - promo_flag: binary, lifts retail price downward (negative coef)
    - public_holiday_week: binary
    - school_holiday_week: binary (Vic terms approximation)
    - fuel_index: drift + noise (positive small effect)
    """
    rng = np.random.default_rng(seed + 1)
    rows = []
    for year in YEARS:
        for week in WEEKS:
            # Inverse phase to price seasonality (hot in summer = Jan/Feb, low temp in July)
            temp = 22.0 - 10.0 * np.cos(2 * np.pi * (week - 4) / 52) + rng.normal(0, 1.5)
            rain = max(0.0, 12.0 + 6.0 * np.cos(2 * np.pi * (week - 30) / 52) + rng.normal(0, 4))
            promo = 1 if rng.random() < 0.12 else 0
            ph = 1 if week in {1, 13, 16, 17, 18, 24, 25, 41, 52} else 0  # rough AU holidays
            sh = 1 if week in {1, 2, 14, 15, 27, 28, 39, 40, 51, 52} else 0  # VIC terms approx
            fuel = 1.50 + 0.02 * (year - 2021) + 0.05 * np.sin(2 * np.pi * week / 52) + rng.normal(0, 0.04)
            rows.append({
                "year": year,
                "week": week,
                "max_temp_c": round(float(temp), 2),
                "rainfall_mm": round(float(rain), 2),
                "promo_flag": int(promo),
                "public_holiday_week": int(ph),
                "school_holiday_week": int(sh),
                "fuel_index": round(float(fuel), 3),
            })
    df = pd.DataFrame(rows)
    df["date"] = df.apply(
        lambda r: pd.Timestamp.fromisocalendar(int(r["year"]), int(r["week"]), 1), axis=1
    )
    return df


def write_sample_files(out_dir: str | Path) -> dict[str, Path]:
    """Write CSV files for use by the GUI's 'Use sample data' button."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    price = generate_price_series()
    # Save in wide layout, matching the spec's input contract
    wide = price.pivot(index="week", columns="year", values="price")
    wide.columns = [str(c) for c in wide.columns]
    wide.index.name = "Week"
    wide_path = out / "sample_tomato_2021_2025.csv"
    wide.to_csv(wide_path)
    regs = generate_regressors()
    regs_path = out / "sample_regressors_2021_2025.csv"
    regs.drop(columns=["date"]).to_csv(regs_path, index=False)
    return {"prices": wide_path, "regressors": regs_path}
