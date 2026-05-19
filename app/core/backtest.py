"""Walk-forward expanding-window backtest.

For each holdout year N in the most recent K years, train on years <= N-1 and predict
year N (52 weeks). Returns per-fold metric frames and per-(model, horizon-step) residuals
used for band calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as M
from .models import WEEKS_PER_YEAR, Model, fit_and_forecast


@dataclass
class FoldResult:
    model: str
    holdout_year: int
    metrics: dict[str, float]
    forecast: pd.DataFrame
    residuals: np.ndarray  # length 52
    fit_seconds: float
    failed: bool
    failure_reason: str


def _years_in(df: pd.DataFrame) -> list[int]:
    return sorted(df["year"].unique().tolist())


def _fold_split(df: pd.DataFrame, holdout_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] < holdout_year].copy()
    test = df[df["year"] == holdout_year].copy().sort_values("week").reset_index(drop=True)
    return train, test


def _make_model_instance(template: Model) -> Model:
    """Return a *new* instance of the model — backtest needs a clean fitter per fold."""
    return template.__class__()


def walk_forward(series: pd.DataFrame, models: list[Model],
                 holdout_years: int = 1,
                 regressors: pd.DataFrame | None = None) -> list[FoldResult]:
    """Run the panel across the last `holdout_years` years of data."""
    years = _years_in(series)
    if len(years) < 2:
        return []
    folds = years[-holdout_years:]
    results: list[FoldResult] = []
    for hy in folds:
        train, test = _fold_split(series, hy)
        if train.empty or test.empty:
            continue
        actual = test["price"].to_numpy(dtype=float)
        horizon = len(test)
        train_regs = None
        future_regs = None
        if regressors is not None and not regressors.empty:
            train_regs = regressors[regressors["year"] < hy].copy()
            future_regs = regressors[regressors["year"] == hy].copy()
        for tpl in models:
            model = _make_model_instance(tpl)
            fc = fit_and_forecast(
                model, train, horizon=horizon,
                regressors=train_regs, future_regressors=future_regs,
            )
            if fc.failed or fc.forecast.empty:
                results.append(FoldResult(
                    model=model.name, holdout_year=hy,
                    metrics={}, forecast=pd.DataFrame(),
                    residuals=np.array([]), fit_seconds=fc.fit_seconds,
                    failed=True, failure_reason=fc.failure_reason,
                ))
                continue
            mean = fc.forecast["mean"].to_numpy(dtype=float)[:horizon]
            p20 = fc.forecast["p20"].to_numpy(dtype=float)[:horizon]
            p80 = fc.forecast["p80"].to_numpy(dtype=float)[:horizon]
            history_prices = train["price"]
            metrics = M.metrics_frame(
                actual=pd.Series(actual), mean=pd.Series(mean),
                p20=pd.Series(p20), p80=pd.Series(p80),
                history=history_prices, seasonality=WEEKS_PER_YEAR,
            )
            residuals = actual - mean
            results.append(FoldResult(
                model=model.name, holdout_year=hy, metrics=metrics,
                forecast=fc.forecast, residuals=residuals,
                fit_seconds=fc.fit_seconds, failed=False, failure_reason="",
            ))
    return results


def aggregate_metrics(folds: list[FoldResult]) -> pd.DataFrame:
    """Per-model mean metrics across folds, sorted by backtest RMSE ascending."""
    if not folds:
        return pd.DataFrame(columns=["model", "rmse", "mae", "mape", "smape", "mase",
                                     "bias", "coverage_p20_p80", "fit_seconds", "n_folds",
                                     "failed"])
    rows = []
    for name, group in pd.DataFrame([
        {"model": f.model, **f.metrics, "fit_seconds": f.fit_seconds, "failed": f.failed}
        for f in folds
    ]).groupby("model"):
        if group["failed"].all():
            rows.append({"model": name, "rmse": float("nan"), "mae": float("nan"),
                         "mape": float("nan"), "smape": float("nan"),
                         "mase": float("nan"), "bias": float("nan"),
                         "coverage_p20_p80": float("nan"),
                         "fit_seconds": float(group["fit_seconds"].mean()),
                         "n_folds": int(len(group)), "failed": True})
            continue
        rows.append({
            "model": name,
            "rmse": float(group["rmse"].mean()),
            "mae": float(group["mae"].mean()),
            "mape": float(group["mape"].mean()),
            "smape": float(group["smape"].mean()),
            "mase": float(group["mase"].mean()),
            "bias": float(group["bias"].mean()),
            "coverage_p20_p80": float(group["coverage_p20_p80"].mean()),
            "fit_seconds": float(group["fit_seconds"].mean()),
            "n_folds": int(len(group)),
            "failed": False,
        })
    out = pd.DataFrame(rows).sort_values("rmse", na_position="last").reset_index(drop=True)
    return out


def pooled_residual_sigma(folds: list[FoldResult], model_name: str,
                          window: int = 8) -> np.ndarray:
    """Per-horizon residual std for a model, pooled across folds + nearby horizon steps.

    See plan: with at most 4 folds we don't have enough per-horizon residuals to take
    a clean per-step quantile, so we pool across a sliding 8-week horizon window.
    """
    model_folds = [f for f in folds if f.model == model_name and not f.failed]
    if not model_folds:
        return np.array([])
    arr = np.stack([f.residuals[:WEEKS_PER_YEAR] for f in model_folds])  # (folds, 52)
    sigma = np.zeros(arr.shape[1])
    for h in range(arr.shape[1]):
        lo = max(0, h - window // 2)
        hi = min(arr.shape[1], h + window // 2 + 1)
        sigma[h] = float(np.std(arr[:, lo:hi]))
    return np.where(sigma > 0, sigma, np.std(arr))
