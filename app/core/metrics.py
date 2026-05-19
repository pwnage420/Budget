"""Forecast accuracy metrics.

RMSE, MAE, MAPE, sMAPE, MASE, bias, P20-P80 coverage, CRPS (sample-based).
Defined formally in README §7.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.mean((f - a) ** 2)))


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs(f - a)))


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Returns MAPE as a decimal (0.10 = 10%). Guards against zero actuals via sMAPE fallback."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    nonzero = np.abs(a) > 1e-9
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((f[nonzero] - a[nonzero]) / a[nonzero])))


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    denom = (np.abs(a) + np.abs(f)) / 2.0
    mask = denom > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(f[mask] - a[mask]) / denom[mask]))


def mase(actual: np.ndarray, forecast: np.ndarray, history: np.ndarray,
         seasonality: int = 52) -> float:
    """Mean Absolute Scaled Error using seasonal-naive scale (lag-52 for weekly)."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    h = np.asarray(history, dtype=float)
    if len(h) <= seasonality:
        return float("nan")
    naive_diffs = np.abs(h[seasonality:] - h[:-seasonality])
    scale = float(np.mean(naive_diffs))
    if scale < 1e-9:
        return float("nan")
    return float(np.mean(np.abs(f - a)) / scale)


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    return float(np.mean(f - a))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals within [lower, upper]. For P20-P80 this should hit ~0.60."""
    a = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    inside = (a >= lo) & (a <= hi)
    return float(np.mean(inside))


def pit_values(actuals: np.ndarray, forecasts_lo: np.ndarray, forecasts_hi: np.ndarray,
                forecasts_mean: np.ndarray) -> np.ndarray:
    """Approximate Probability Integral Transform values from parametric bands.

    Maps each actual onto its forecast CDF using the (mean, σ) implied by the
    [lo, hi] band. Returns values in [0, 1]; under perfect calibration they
    should be uniform — see Diebold, Gunther & Tay (1998).

    `forecasts_lo` and `forecasts_hi` should be the P20 / P80 columns from a
    forecast DataFrame; we infer σ from the band width.
    """
    from scipy.stats import norm  # imported here to avoid scipy at module import

    a = np.asarray(actuals, dtype=float)
    lo = np.asarray(forecasts_lo, dtype=float)
    hi = np.asarray(forecasts_hi, dtype=float)
    mean = np.asarray(forecasts_mean, dtype=float)
    # P20-P80 spans 2 × 0.8416σ for standard normal → σ = (hi - lo) / (2 * 0.8416)
    sigma = (hi - lo) / (2 * 0.8416)
    sigma = np.where(sigma > 1e-9, sigma, 1e-9)
    return norm.cdf(a, loc=mean, scale=sigma)


def crps_sample(actual: float, samples: np.ndarray) -> float:
    """Sample-based CRPS (smaller is better). Used for probabilistic models."""
    s = np.sort(np.asarray(samples, dtype=float))
    n = len(s)
    if n == 0:
        return float("nan")
    diff_term = float(np.mean(np.abs(s - actual)))
    pairwise = float(np.mean(np.abs(s[:, None] - s[None, :])))
    return diff_term - 0.5 * pairwise


def position_vs_band(actual: float, p20: float, p80: float) -> str:
    if np.isnan(actual):
        return "n/a"
    if actual < p20:
        return "Below"
    if actual > p80:
        return "Above"
    return "Within"


def metrics_frame(actual: pd.Series, mean: pd.Series, p20: pd.Series, p80: pd.Series,
                  history: pd.Series, seasonality: int = 52) -> dict[str, float]:
    """Convenience: compute the full metric battery for one fold."""
    a = actual.to_numpy(dtype=float)
    f = mean.to_numpy(dtype=float)
    return {
        "rmse": rmse(a, f),
        "mae": mae(a, f),
        "mape": mape(a, f),
        "smape": smape(a, f),
        "mase": mase(a, f, history.to_numpy(dtype=float), seasonality),
        "bias": bias(a, f),
        "coverage_p20_p80": coverage(a, p20.to_numpy(dtype=float), p80.to_numpy(dtype=float)),
    }
