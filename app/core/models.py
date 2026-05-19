"""Forecasting models — common Protocol + concrete implementations.

Every model exposes:
    fit(history_df, regressors=None) -> None
    predict(horizon, residual_std=None, regressors_future=None) -> pd.DataFrame
        DataFrame columns: date, mean, p20, p30, p70, p80

Bands for non-probabilistic models are parametric: ŷ ± z × σ. The backtest later
overrides σ with the pooled residual std (see backtest.py / blending.py).

Optional libs (prophet, pmdarima, xgboost, lightgbm) are imported lazily and the
model declares itself unavailable if the import fails — never crashes the panel.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from .features import build_features, train_split
from .seeding import seed_all

WEEKS_PER_YEAR = 52
DEFAULT_Z = {"p20": 0.84, "p30": 0.52, "p70": 0.52, "p80": 0.84}  # one-sided z values


@dataclass
class ForecastResult:
    name: str
    forecast: pd.DataFrame  # date, mean, p20, p30, p70, p80
    fit_seconds: float = 0.0
    in_sample_rmse: float = float("nan")
    failed: bool = False
    failure_reason: str = ""
    extras: dict = field(default_factory=dict)


def _band(mean: np.ndarray, sigma: float | np.ndarray) -> dict[str, np.ndarray]:
    """Parametric bands from mean + scalar/array sigma. Sigma can vary by horizon."""
    sig = np.asarray(sigma, dtype=float)
    return {
        "p20": mean - DEFAULT_Z["p20"] * sig,
        "p30": mean - DEFAULT_Z["p30"] * sig,
        "p70": mean + DEFAULT_Z["p70"] * sig,
        "p80": mean + DEFAULT_Z["p80"] * sig,
    }


def _make_forecast_df(future_dates: list[pd.Timestamp], mean: np.ndarray,
                      sigma: float | np.ndarray) -> pd.DataFrame:
    bands = _band(mean, sigma)
    return pd.DataFrame({
        "date": future_dates,
        "mean": mean,
        "p20": bands["p20"],
        "p30": bands["p30"],
        "p70": bands["p70"],
        "p80": bands["p80"],
    })


def _future_dates(history: pd.DataFrame, horizon: int) -> list[pd.Timestamp]:
    last = pd.Timestamp(history["date"].max())
    return [last + pd.Timedelta(weeks=h) for h in range(1, horizon + 1)]


def _in_sample_rmse(history: pd.DataFrame, fitted: np.ndarray) -> float:
    actual = history["price"].to_numpy(dtype=float)
    n = min(len(actual), len(fitted))
    if n == 0:
        return float("nan")
    return float(np.sqrt(np.mean((fitted[-n:] - actual[-n:]) ** 2)))


class Model(Protocol):
    name: str
    available: bool

    def fit(self, history: pd.DataFrame, regressors: pd.DataFrame | None = None) -> None: ...

    def predict(self, horizon: int, residual_sigma: np.ndarray | float | None = None,
                future_regressors: pd.DataFrame | None = None) -> pd.DataFrame: ...


# ----- Baselines -----------------------------------------------------------------------

class NaiveModel:
    name = "Naive (last-year same week)"
    available = True

    def __init__(self) -> None:
        self.history: pd.DataFrame | None = None
        self.sigma: float = 0.3

    def fit(self, history: pd.DataFrame, regressors: pd.DataFrame | None = None) -> None:
        self.history = history.copy()
        # residual std from year-over-year diffs as a default sigma
        prices = history["price"].to_numpy(dtype=float)
        if len(prices) > WEEKS_PER_YEAR:
            diffs = prices[WEEKS_PER_YEAR:] - prices[:-WEEKS_PER_YEAR]
            self.sigma = float(np.std(diffs)) or 0.3
        else:
            self.sigma = float(np.std(prices)) or 0.3

    def predict(self, horizon: int, residual_sigma: np.ndarray | float | None = None,
                future_regressors: pd.DataFrame | None = None) -> pd.DataFrame:
        assert self.history is not None
        hist = self.history
        last_year_prices = hist.tail(WEEKS_PER_YEAR)["price"].to_numpy(dtype=float)
        # tile / extend to cover horizon
        rep = int(np.ceil(horizon / WEEKS_PER_YEAR))
        means = np.tile(last_year_prices, rep)[:horizon]
        sigma = residual_sigma if residual_sigma is not None else self.sigma
        return _make_forecast_df(_future_dates(hist, horizon), means, sigma)


class SeasonalAvgTrendModel:
    name = "Seasonal Avg + Trend"
    available = True

    def __init__(self) -> None:
        self.weekly_mean: dict[int, float] = {}
        self.trend_per_week: float = 0.0
        self.history: pd.DataFrame | None = None
        self.sigma: float = 0.3
        self.last_t: int = 0

    def fit(self, history: pd.DataFrame, regressors: pd.DataFrame | None = None) -> None:
        h = history.copy().sort_values("date").reset_index(drop=True)
        self.history = h
        self.weekly_mean = {int(w): float(g["price"].mean())
                            for w, g in h.groupby("week")}
        # Fit a simple OLS trend on de-seasonalised series
        t = np.arange(len(h), dtype=float)
        deseason = h["price"].to_numpy(dtype=float) - h["week"].map(self.weekly_mean).to_numpy()
        if len(t) > 1:
            self.trend_per_week = float(np.polyfit(t, deseason, 1)[0])
        self.last_t = len(h)
        # In-sample residual std
        fitted = h["week"].map(self.weekly_mean).to_numpy() + self.trend_per_week * t
        self.sigma = float(np.std(h["price"].to_numpy() - fitted)) or 0.3

    def predict(self, horizon: int, residual_sigma: np.ndarray | float | None = None,
                future_regressors: pd.DataFrame | None = None) -> pd.DataFrame:
        assert self.history is not None
        future_dates = _future_dates(self.history, horizon)
        weeks = [d.isocalendar().week for d in future_dates]
        base = np.array([self.weekly_mean.get(int(w), float(np.mean(list(self.weekly_mean.values()))))
                         for w in weeks])
        trend = self.trend_per_week * (self.last_t + np.arange(1, horizon + 1))
        means = base + trend
        sigma = residual_sigma if residual_sigma is not None else self.sigma
        return _make_forecast_df(future_dates, means, sigma)


class HoltWintersModel:
    name = "Holt-Winters ETS"
    available = True

    def __init__(self) -> None:
        self.fitted_model = None
        self.history: pd.DataFrame | None = None
        self.sigma: float = 0.3

    def fit(self, history: pd.DataFrame, regressors: pd.DataFrame | None = None) -> None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        self.history = history.copy()
        y = history["price"].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.fitted_model = ExponentialSmoothing(
                    y, trend="add", seasonal="add", seasonal_periods=WEEKS_PER_YEAR,
                    initialization_method="estimated",
                ).fit(optimized=True, use_brute=False)
            except Exception:
                # Fall back to additive without seasonal if seasonal init fails
                self.fitted_model = ExponentialSmoothing(
                    y, trend="add", seasonal=None,
                    initialization_method="estimated",
                ).fit(optimized=True)
        resid = y - np.asarray(self.fitted_model.fittedvalues, dtype=float)
        self.sigma = float(np.std(resid)) or 0.3

    def predict(self, horizon: int, residual_sigma: np.ndarray | float | None = None,
                future_regressors: pd.DataFrame | None = None) -> pd.DataFrame:
        assert self.fitted_model is not None and self.history is not None
        means = np.asarray(self.fitted_model.forecast(horizon), dtype=float)
        sigma = residual_sigma if residual_sigma is not None else self.sigma
        return _make_forecast_df(_future_dates(self.history, horizon), means, sigma)


class SarimaModel:
    name = "SARIMA(1,0,1)(1,1,1)[52]"
    available = True

    def __init__(self, order=(1, 0, 1), seasonal_order=(1, 1, 1, WEEKS_PER_YEAR)) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted = None
        self.history: pd.DataFrame | None = None
        self.sigma: float = 0.3

    def fit(self, history: pd.DataFrame, regressors: pd.DataFrame | None = None) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        self.history = history.copy()
        y = history["price"].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fitted = SARIMAX(
                y, order=self.order, seasonal_order=self.seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)
        resid = np.asarray(self.fitted.resid, dtype=float)
        self.sigma = float(np.std(resid[~np.isnan(resid)])) or 0.3

    def predict(self, horizon: int, residual_sigma: np.ndarray | float | None = None,
                future_regressors: pd.DataFrame | None = None) -> pd.DataFrame:
        assert self.fitted is not None and self.history is not None
        means = np.asarray(self.fitted.forecast(horizon), dtype=float)
        sigma = residual_sigma if residual_sigma is not None else self.sigma
        return _make_forecast_df(_future_dates(self.history, horizon), means, sigma)


class AutoSarimaModel:
    name = "Auto-SARIMA"

    def __init__(self) -> None:
        try:
            import pmdarima  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False
        self.fitted = None
        self.history: pd.DataFrame | None = None
        self.sigma: float = 0.3

    def fit(self, history: pd.DataFrame, regressors: pd.DataFrame | None = None) -> None:
        if not self.available:
            raise RuntimeError("pmdarima not installed — install requirements-full.txt")
        from pmdarima import auto_arima
        self.history = history.copy()
        y = history["price"].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fitted = auto_arima(
                y, seasonal=True, m=WEEKS_PER_YEAR,
                max_p=2, max_q=2, max_P=1, max_Q=1, max_d=1, max_D=1,
                stepwise=True, suppress_warnings=True, error_action="ignore",
            )
        resid = np.asarray(self.fitted.resid(), dtype=float)
        self.sigma = float(np.std(resid[~np.isnan(resid)])) or 0.3

    def predict(self, horizon: int, residual_sigma: np.ndarray | float | None = None,
                future_regressors: pd.DataFrame | None = None) -> pd.DataFrame:
        assert self.fitted is not None and self.history is not None
        means = np.asarray(self.fitted.predict(horizon), dtype=float)
        sigma = residual_sigma if residual_sigma is not None else self.sigma
        return _make_forecast_df(_future_dates(self.history, horizon), means, sigma)


# ----- ML / tree-based models ----------------------------------------------------------

class _TreeRegressorBase:
    """Direct multi-output forecasting via a single horizon-aware regressor.

    We append a `horizon_step` feature and stack one training row per (origin_t, h)
    pair using fully observed lag features. This is the direct strategy
    (Bontempi et al. 2013) avoiding recursive compounding.
    """
    name = "tree-base"
    available = True

    def __init__(self) -> None:
        self.model = None
        self.history: pd.DataFrame | None = None
        self.regressors: pd.DataFrame | None = None
        self.feature_cols: list[str] = []
        self.sigma: float = 0.3

    def _build_estimator(self):  # to be overridden
        raise NotImplementedError

    def fit(self, history: pd.DataFrame, regressors: pd.DataFrame | None = None) -> None:
        seed_all(42)
        self.history = history.copy()
        self.regressors = regressors.copy() if regressors is not None else None
        feats = build_features(history, regressors)
        X, y = train_split(feats)
        if len(X) == 0:
            raise ValueError("Not enough history to build lag features.")
        self.feature_cols = list(X.columns)
        self.model = self._build_estimator()
        self.model.fit(X, y.values)
        pred = self.model.predict(X)
        self.sigma = float(np.std(y.values - pred)) or 0.3

    def predict(self, horizon: int, residual_sigma: np.ndarray | float | None = None,
                future_regressors: pd.DataFrame | None = None) -> pd.DataFrame:
        assert self.model is not None and self.history is not None
        # Recursive prediction for lags: at each step, fill predicted price back into history
        rolling_hist = self.history.copy().reset_index(drop=True)
        means: list[float] = []
        last_date = pd.Timestamp(rolling_hist["date"].max())
        future_dates = []
        for h in range(1, horizon + 1):
            next_date = last_date + pd.Timedelta(weeks=h)
            iso = next_date.isocalendar()
            future_dates.append(next_date)
            extended = pd.concat([rolling_hist, pd.DataFrame({
                "date": [next_date], "year": [int(iso.year)],
                "week": [int(iso.week)], "price": [np.nan],
            })], ignore_index=True)
            regs = self._combine_regressors(future_regressors)
            feats = build_features(extended, regs)
            row = feats.iloc[[-1]][self.feature_cols].copy()
            row = row.ffill().fillna(0.0)
            # Pass a DataFrame so LightGBM matches the feature names from fit().
            # (XGB / sklearn don't care; LightGBM warns otherwise.)
            pred = float(self.model.predict(row)[0])
            means.append(pred)
            rolling_hist = pd.concat([rolling_hist, pd.DataFrame({
                "date": [next_date], "year": [int(iso.year)],
                "week": [int(iso.week)], "price": [pred],
            })], ignore_index=True)
        sigma = residual_sigma if residual_sigma is not None else self.sigma
        return _make_forecast_df(future_dates, np.array(means), sigma)

    def _combine_regressors(self, future_regressors: pd.DataFrame | None) -> pd.DataFrame | None:
        if self.regressors is None and future_regressors is None:
            return None
        parts = []
        if self.regressors is not None:
            parts.append(self.regressors)
        if future_regressors is not None:
            parts.append(future_regressors)
        return pd.concat(parts, ignore_index=True) if parts else None


class RandomForestModel(_TreeRegressorBase):
    name = "Random Forest"

    def _build_estimator(self):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=2,
            n_jobs=1, random_state=42, oob_score=False,
        )


class XGBoostModel(_TreeRegressorBase):
    name = "XGBoost"

    def __init__(self) -> None:
        super().__init__()
        try:
            import xgboost  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def _build_estimator(self):
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=1, random_state=42, verbosity=0, tree_method="hist",
        )


class LightGBMModel(_TreeRegressorBase):
    name = "LightGBM"

    def __init__(self) -> None:
        super().__init__()
        try:
            import lightgbm  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def _build_estimator(self):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, num_leaves=15,
            n_jobs=1, random_state=42, deterministic=True,
            force_col_wise=True, verbosity=-1,
        )


# ----- Prophet (optional, lazy import) -------------------------------------------------

class ProphetModel:
    name = "Prophet"

    def __init__(self) -> None:
        try:
            import prophet  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False
        self.model = None
        self.history: pd.DataFrame | None = None
        self.regressor_cols: list[str] = []
        self.sigma: float = 0.3

    def fit(self, history: pd.DataFrame, regressors: pd.DataFrame | None = None) -> None:
        if not self.available:
            raise RuntimeError("prophet not installed — install requirements-full.txt")
        from prophet import Prophet
        seed_all(42)
        self.history = history.copy()
        df = history.rename(columns={"date": "ds", "price": "y"})[["ds", "y"]]
        m = Prophet(
            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
            uncertainty_samples=200,
        )
        self.regressor_cols = []
        if regressors is not None and not regressors.empty:
            regs = regressors.copy()
            if "date" not in regs.columns:
                regs["date"] = regs.apply(
                    lambda r: pd.Timestamp.fromisocalendar(int(r["year"]), int(r["week"]), 1),
                    axis=1,
                )
            regs = regs.drop(columns=[c for c in ("year", "week") if c in regs.columns])
            regs = regs.rename(columns={"date": "ds"})
            df = df.merge(regs, on="ds", how="left")
            for col in regs.columns:
                if col == "ds":
                    continue
                m.add_regressor(col)
                self.regressor_cols.append(col)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df)
        self.model = m
        # In-sample residual std
        in_sample = m.predict(df.drop(columns=["y"]))
        resid = df["y"].to_numpy() - in_sample["yhat"].to_numpy()
        self.sigma = float(np.std(resid)) or 0.3

    def predict(self, horizon: int, residual_sigma: np.ndarray | float | None = None,
                future_regressors: pd.DataFrame | None = None) -> pd.DataFrame:
        assert self.model is not None and self.history is not None
        future_dates = _future_dates(self.history, horizon)
        future = pd.DataFrame({"ds": future_dates})
        if self.regressor_cols and future_regressors is not None:
            regs = future_regressors.copy()
            if "date" not in regs.columns:
                regs["date"] = regs.apply(
                    lambda r: pd.Timestamp.fromisocalendar(int(r["year"]), int(r["week"]), 1),
                    axis=1,
                )
            regs = regs.rename(columns={"date": "ds"})
            future = future.merge(regs, on="ds", how="left")
        forecast = self.model.predict(future)
        means = forecast["yhat"].to_numpy(dtype=float)
        sigma = residual_sigma if residual_sigma is not None else self.sigma
        return _make_forecast_df(future_dates, means, sigma)


# ----- Wrapper used by the panel -------------------------------------------------------

def fit_and_forecast(model: Model, history: pd.DataFrame, horizon: int,
                     regressors: pd.DataFrame | None = None,
                     future_regressors: pd.DataFrame | None = None,
                     residual_sigma: np.ndarray | float | None = None,
                     timeout_seconds: float | None = None) -> ForecastResult:
    """Defensive fit+predict that returns a structured result on failure (never raises)."""
    t0 = time.perf_counter()
    if not getattr(model, "available", True):
        return ForecastResult(name=model.name, forecast=pd.DataFrame(),
                              failed=True, failure_reason="Library not installed")
    try:
        model.fit(history, regressors=regressors)
        fc = model.predict(horizon, residual_sigma=residual_sigma,
                           future_regressors=future_regressors)
        elapsed = time.perf_counter() - t0
        # in-sample rmse if accessible
        in_sample_rmse = float("nan")
        if hasattr(model, "history") and getattr(model, "sigma", None) is not None:
            in_sample_rmse = float(model.sigma)
        return ForecastResult(
            name=model.name, forecast=fc,
            fit_seconds=elapsed, in_sample_rmse=in_sample_rmse,
        )
    except Exception as exc:  # noqa: BLE001 — surface plain-English reason
        return ForecastResult(
            name=model.name, forecast=pd.DataFrame(),
            fit_seconds=time.perf_counter() - t0, failed=True,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )


def default_panel(advanced: bool = False) -> list[Model]:
    """Default panel ordered most-to-least likely to succeed cheaply.

    Advanced gates Prophet / Auto-SARIMA / (future) BSTS / N-BEATS / TFT.
    """
    panel: list[Model] = [
        NaiveModel(),
        SeasonalAvgTrendModel(),
        HoltWintersModel(),
        SarimaModel(),
        RandomForestModel(),
        XGBoostModel(),
        LightGBMModel(),
    ]
    if advanced:
        panel.extend([AutoSarimaModel(), ProphetModel()])
    return panel
