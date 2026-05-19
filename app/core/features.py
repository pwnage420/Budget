"""Lag / rolling / Fourier / calendar feature engineering for tree-based models.

Features (per spec):
- Lags y_{t-1}, y_{t-2}, y_{t-4}, y_{t-52}
- Rolling mean+std over 4 / 13 / 52 week windows
- Calendar: week-of-year, month, quarter, year
- Fourier: sin/cos at k=1,2,3 for 52-week cycle
- Linear trend t
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 4, 52)
ROLLING_WINDOWS = (4, 13, 52)
FOURIER_K = (1, 2, 3)


def build_features(df: pd.DataFrame, regressors: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a feature frame indexed by `date` from the price + optional regressor inputs.

    Input df must have columns: date, year, week, price (long format).
    Output is one row per date, with all features + the target column `y`.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    out = pd.DataFrame(index=df["date"].values)
    out.index.name = "date"
    y = df["price"].astype(float).reset_index(drop=True)

    out["y"] = y.values
    out["t"] = np.arange(len(df), dtype=float)
    out["week"] = df["week"].values.astype(int)
    out["month"] = df["date"].dt.month.values.astype(int)
    out["quarter"] = df["date"].dt.quarter.values.astype(int)
    out["year"] = df["year"].values.astype(int)

    for lag in LAGS:
        out[f"lag_{lag}"] = y.shift(lag).values
    for w in ROLLING_WINDOWS:
        out[f"roll_mean_{w}"] = y.shift(1).rolling(w, min_periods=1).mean().values
        out[f"roll_std_{w}"] = y.shift(1).rolling(w, min_periods=1).std(ddof=0).values
    for k in FOURIER_K:
        out[f"sin_{k}"] = np.sin(2 * np.pi * k * out["week"].values / 52.0)
        out[f"cos_{k}"] = np.cos(2 * np.pi * k * out["week"].values / 52.0)

    if regressors is not None and not regressors.empty:
        regs = regressors.copy()
        if "date" not in regs.columns:
            regs["date"] = regs.apply(
                lambda r: pd.Timestamp.fromisocalendar(int(r["year"]), int(r["week"]), 1),
                axis=1,
            )
        regs = regs.set_index("date")
        regs = regs.drop(columns=[c for c in ("year", "week") if c in regs.columns],
                         errors="ignore")
        # Align by date; missing future rows will be filled later by the caller
        regs = regs.reindex(out.index)
        for col in regs.columns:
            out[f"reg_{col}"] = regs[col].astype(float).values
    return out


def train_split(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows with NaN lags (warmup); return X, y for sklearn-style fit."""
    fully = features.dropna(subset=[c for c in features.columns if c.startswith("lag_")])
    y = fully["y"]
    X = fully.drop(columns=["y"])
    return X, y


def future_frame(history: pd.DataFrame, horizon: int,
                 future_regressors: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a feature frame for the next `horizon` weeks, given history.

    Lag features use either known history values or recursive predictions filled in by the
    caller during stepwise prediction. We build the rows here; we don't predict.
    """
    last_date = pd.Timestamp(history["date"].max())
    future_dates = [last_date + pd.Timedelta(weeks=h) for h in range(1, horizon + 1)]
    iso = [d.isocalendar() for d in future_dates]
    future_df = pd.DataFrame({
        "date": future_dates,
        "year": [int(c.year) for c in iso],
        "week": [int(c.week) for c in iso],
        "price": [np.nan] * horizon,
    })
    combined = pd.concat([history, future_df], ignore_index=True)
    feats = build_features(combined, regressors=future_regressors)
    return feats.tail(horizon).copy()
