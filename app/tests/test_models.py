"""Tests for app.core.models — happy path + graceful failures."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.data_loader import load_series_from_dataframe
from app.core.models import (
    HoltWintersModel,
    LightGBMModel,
    NaiveModel,
    RandomForestModel,
    SarimaModel,
    SeasonalAvgTrendModel,
    XGBoostModel,
    fit_and_forecast,
)


def _short_history(n: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=n, freq="W-MON")
    return pd.DataFrame({
        "date": dates,
        "year": [d.isocalendar().year for d in dates],
        "week": [d.isocalendar().week for d in dates],
        "price": 3.0 + 0.5 * np.sin(np.linspace(0, 4 * np.pi, n)),
    })


def test_naive_predicts_52_weeks(synthetic_history):
    history = load_series_from_dataframe(synthetic_history).df
    m = NaiveModel()
    res = fit_and_forecast(m, history, horizon=52)
    assert not res.failed
    assert len(res.forecast) == 52
    assert set(res.forecast.columns) == {"date", "mean", "p20", "p30", "p70", "p80"}


def test_seasonal_avg_trend(synthetic_history):
    history = load_series_from_dataframe(synthetic_history).df
    res = fit_and_forecast(SeasonalAvgTrendModel(), history, horizon=52)
    assert not res.failed
    fc = res.forecast
    assert (fc["p20"] <= fc["p30"]).all()
    assert (fc["p30"] <= fc["mean"]).all()
    assert (fc["mean"] <= fc["p70"]).all()
    assert (fc["p70"] <= fc["p80"]).all()


def test_holt_winters(synthetic_history):
    history = load_series_from_dataframe(synthetic_history).df
    res = fit_and_forecast(HoltWintersModel(), history, horizon=52)
    assert not res.failed
    assert len(res.forecast) == 52


def test_sarima(synthetic_history):
    history = load_series_from_dataframe(synthetic_history).df
    res = fit_and_forecast(SarimaModel(), history, horizon=52)
    assert not res.failed
    assert len(res.forecast) == 52


def test_random_forest(synthetic_history):
    history = load_series_from_dataframe(synthetic_history).df
    res = fit_and_forecast(RandomForestModel(), history, horizon=52)
    assert not res.failed
    assert len(res.forecast) == 52
    assert res.forecast["mean"].between(0, 20).all()


def test_xgboost(synthetic_history):
    history = load_series_from_dataframe(synthetic_history).df
    res = fit_and_forecast(XGBoostModel(), history, horizon=52)
    assert not res.failed
    assert len(res.forecast) == 52


def test_lightgbm(synthetic_history):
    history = load_series_from_dataframe(synthetic_history).df
    res = fit_and_forecast(LightGBMModel(), history, horizon=52)
    assert not res.failed
    assert len(res.forecast) == 52


def test_model_fails_gracefully_on_too_short_history():
    """Models with strong seasonal assumptions should fail cleanly, not crash."""
    short = _short_history(20)
    res = fit_and_forecast(SarimaModel(), short, horizon=52)
    # Either succeeds with degraded model or returns ModelFailure — both acceptable
    assert isinstance(res.failed, bool)
    if res.failed:
        assert res.failure_reason  # plain-English reason present
