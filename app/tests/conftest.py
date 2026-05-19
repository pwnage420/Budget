"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from app.core.sample_data import generate_price_series, generate_regressors


@pytest.fixture
def synthetic_history():
    return generate_price_series(seed=42)


@pytest.fixture
def synthetic_regressors():
    return generate_regressors(seed=42)
