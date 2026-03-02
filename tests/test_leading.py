"""
Unit tests for the Leading Index (LI) diffusion index.
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators.leading import (
    build_leading_index,
    compute_li_momentum,
    validate_li_against_btc,
    COMPONENT_SPEC,
    MA_WINDOW,
)


def make_mock_data(n_months: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic monthly data for all 7 LI components."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-31", periods=n_months, freq="ME")

    data = {}
    base_trend = np.cumsum(rng.normal(0, 1, n_months))

    for sid, invert, _ in COMPONENT_SPEC:
        series = base_trend + rng.normal(0, 0.5, n_months)
        if invert:
            series = -series
        data[sid] = series

    return pd.DataFrame(data, index=dates)


class TestBuildLeadingIndex:
    def test_output_range(self):
        data = make_mock_data()
        result = build_leading_index(data)
        # LI should be 0–100
        assert result.index.min() >= 0.0
        assert result.index.max() <= 100.0

    def test_correct_columns(self):
        data = make_mock_data()
        result = build_leading_index(data)
        expected_labels = [label for _, _, label in COMPONENT_SPEC]
        assert set(expected_labels).issubset(set(result.components.columns))

    def test_binary_components(self):
        data = make_mock_data()
        result = build_leading_index(data)
        # Components should only be 0 or 1 (integer binary)
        for col in result.components.columns:
            vals = result.components[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"Column {col} has non-binary values: {vals}"

    def test_missing_column_raises(self):
        data = make_mock_data()
        data = data.drop(columns=["WM2NS"])
        with pytest.raises(ValueError, match="Missing series"):
            build_leading_index(data)

    def test_named_li(self):
        data = make_mock_data()
        result = build_leading_index(data)
        assert result.index.name == "LI"

    def test_monotone_when_all_expanding(self):
        """When all components are above their MA, LI should be 100."""
        data = make_mock_data()
        # Make all series monotonically increasing (always above MA)
        dates = data.index
        for col in data.columns:
            data[col] = np.arange(len(data), dtype=float) * 100
        result = build_leading_index(data, ma_window=3)
        # After the initial MA warm-up, LI should be near 100
        tail = result.index.iloc[MA_WINDOW:]
        assert (tail >= 90).all(), f"Expected LI ~100 when all expanding, got: {tail.tail()}"

    def test_short_data(self):
        """Should still work (with warnings) for short series."""
        data = make_mock_data(n_months=15)
        result = build_leading_index(data, ma_window=3)
        assert len(result.index) > 0


class TestComputeLIMomentum:
    def test_output_length(self):
        data = make_mock_data()
        result = build_leading_index(data)
        mom = compute_li_momentum(result.index, window=3)
        assert len(mom) == len(result.index)

    def test_named_momentum(self):
        data = make_mock_data()
        result = build_leading_index(data)
        mom = compute_li_momentum(result.index)
        assert mom.name == "LI_momentum"


class TestValidateLI:
    def test_returns_dict(self):
        rng = np.random.default_rng(0)
        n = 60
        dates = pd.date_range("2019-01-31", periods=n, freq="ME")
        li = pd.Series(rng.uniform(20, 80, n), index=dates, name="LI")
        btc = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.1, n))), index=dates)
        result = validate_li_against_btc(li, btc, max_lag=6)
        assert "peak_correlation" in result
        assert "peak_lag_months" in result
        assert "correlation_at_0" in result
        assert 0 <= result["peak_lag_months"] <= 6
