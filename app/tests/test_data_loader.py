"""Tests for app.core.data_loader."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.data_loader import (
    DataLoadError,
    load_series,
    load_series_from_dataframe,
)


def test_wide_layout_loads(tmp_path):
    df = pd.DataFrame({
        "Week": list(range(1, 53)),
        "2023": [3.0 + 0.1 * (i % 7) for i in range(52)],
        "2024": [3.2 + 0.1 * (i % 7) for i in range(52)],
    })
    p = tmp_path / "wide.csv"
    df.to_csv(p, index=False)
    series = load_series(p)
    assert series.report.ok
    assert series.report.years == [2023, 2024]
    assert len(series.df) == 104


def test_long_layout_loads(tmp_path):
    dates = pd.date_range("2023-01-02", periods=104, freq="W-MON")
    df = pd.DataFrame({"Date": dates, "Price": np.linspace(3.0, 4.0, 104)})
    p = tmp_path / "long.csv"
    df.to_csv(p, index=False)
    series = load_series(p)
    assert series.report.ok


def test_negative_prices_rejected():
    df = pd.DataFrame({
        "Week": [1, 2, 3],
        "2023": [-1.0, 2.0, 3.0],
    })
    with pytest.raises(DataLoadError, match="strictly positive"):
        load_series_from_dataframe(df)


def test_outlier_flagged():
    df = pd.DataFrame({
        "Week": list(range(1, 53)),
        "2021": [3.0] * 52,
        "2022": [3.0] * 52,
        "2023": [3.0] * 52,
        "2024": [3.0] * 52,
        "2025": [3.0] * 52,
    })
    df.loc[5, "2024"] = 20.0  # huge outlier in week 6
    series = load_series_from_dataframe(df)
    assert series.report.ok
    flagged_weeks = {(y, w) for y, w, _ in series.report.outliers}
    assert (2024, 6) in flagged_weeks


def test_macro_file_rejected(tmp_path):
    p = tmp_path / "evil.xlsm"
    p.write_bytes(b"fake")
    with pytest.raises(DataLoadError, match="Macro-enabled"):
        load_series(p)


def test_oversize_file_rejected(tmp_path):
    p = tmp_path / "big.csv"
    # Write a 51 MB file
    with open(p, "wb") as f:
        f.truncate(51 * 1024 * 1024)
    with pytest.raises(DataLoadError, match="too large"):
        load_series(p)


def test_unknown_layout_rejected():
    df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    with pytest.raises(DataLoadError):
        load_series_from_dataframe(df)
