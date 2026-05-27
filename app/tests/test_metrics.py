"""Tests for app.core.metrics."""
from __future__ import annotations

import numpy as np

from app.core.metrics import bias, coverage, crps_sample, mape, mase, rmse, smape


def test_rmse_zero_when_perfect():
    a = np.array([1.0, 2.0, 3.0])
    assert rmse(a, a) == 0.0


def test_rmse_pythagorean():
    a = np.array([0.0, 0.0])
    f = np.array([3.0, 4.0])
    # RMSE = sqrt(mean(9, 16)) = sqrt(12.5) ≈ 3.5355
    assert abs(rmse(a, f) - np.sqrt(12.5)) < 1e-9


def test_mape_handles_zero_actuals():
    a = np.array([0.0, 0.0])
    f = np.array([1.0, 1.0])
    assert np.isnan(mape(a, f))


def test_smape_finite_with_zero_actual():
    a = np.array([0.0, 2.0])
    f = np.array([1.0, 2.0])
    val = smape(a, f)
    assert np.isfinite(val)


def test_mase_with_history():
    history = np.linspace(1.0, 2.0, 100)
    a = np.array([2.0])
    f = np.array([2.05])
    val = mase(a, f, history, seasonality=52)
    assert np.isfinite(val)
    assert val > 0


def test_bias_signed():
    a = np.array([1.0, 1.0])
    f = np.array([2.0, 2.0])
    assert bias(a, f) == 1.0
    assert bias(f, a) == -1.0


def test_coverage_full_inside():
    a = np.array([1.5, 2.0, 2.5])
    lo = np.array([1.0, 1.0, 1.0])
    hi = np.array([3.0, 3.0, 3.0])
    assert coverage(a, lo, hi) == 1.0


def test_coverage_none_inside():
    a = np.array([10.0])
    lo = np.array([0.0])
    hi = np.array([1.0])
    assert coverage(a, lo, hi) == 0.0


def test_crps_sample_finite():
    samples = np.random.default_rng(0).normal(0, 1, 100)
    val = crps_sample(0.5, samples)
    assert np.isfinite(val)
