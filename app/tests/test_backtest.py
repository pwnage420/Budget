"""Tests for app.core.backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.backtest import aggregate_metrics, pooled_residual_sigma, walk_forward
from app.core.data_loader import load_series_from_dataframe
from app.core.models import HoltWintersModel, NaiveModel, SeasonalAvgTrendModel


def test_walk_forward_runs(synthetic_history):
    history = load_series_from_dataframe(synthetic_history).df
    folds = walk_forward(history, [NaiveModel(), SeasonalAvgTrendModel(), HoltWintersModel()],
                          holdout_years=1)
    assert len(folds) >= 3  # at least one fold per model
    for f in folds:
        if not f.failed:
            assert "rmse" in f.metrics
            assert len(f.residuals) == 52


def test_aggregate_metrics_sorts_by_rmse(synthetic_history):
    history = load_series_from_dataframe(synthetic_history).df
    folds = walk_forward(history, [NaiveModel(), SeasonalAvgTrendModel()], holdout_years=1)
    metrics = aggregate_metrics(folds)
    rmses = metrics["rmse"].dropna().tolist()
    assert rmses == sorted(rmses)


def test_pooled_sigma_per_horizon(synthetic_history):
    history = load_series_from_dataframe(synthetic_history).df
    folds = walk_forward(history, [SeasonalAvgTrendModel()], holdout_years=1)
    sigma = pooled_residual_sigma(folds, "Seasonal Avg + Trend")
    assert len(sigma) == 52
    assert (sigma > 0).all()


def test_single_year_data_returns_empty_folds():
    # Only one year — walk-forward has nothing to hold out
    dates = pd.date_range("2024-01-01", periods=52, freq="W-MON")
    df = pd.DataFrame({
        "date": dates,
        "year": 2024,
        "week": list(range(1, 53)),
        "price": np.linspace(3.0, 4.0, 52),
    })
    folds = walk_forward(df, [NaiveModel()], holdout_years=1)
    # Need at least 2 years to backtest; should return empty
    assert folds == []
