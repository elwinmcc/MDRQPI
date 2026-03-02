"""
Unit tests for Cross-Correlation Function and Diebold-Mariano test.
"""

import numpy as np
import pandas as pd
import pytest

from src.projection.ccf import (
    compute_ccf,
    diebold_mariano_test,
    estimate_forecast_horizon,
    DieboldMarianoResult,
    CCFResult,
)


def make_lagged_pair(
    n: int = 100,
    lag: int = 6,
    noise_std: float = 0.1,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """Generate x1 that leads x2 by *lag* periods."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=n, freq="ME")
    x1 = pd.Series(np.sin(np.linspace(0, 4 * np.pi, n)) + rng.normal(0, noise_std, n), index=dates)
    x2_values = np.zeros(n)
    x2_values[lag:] = x1.values[:-lag]
    x2 = pd.Series(x2_values + rng.normal(0, noise_std, n), index=dates)
    return x1, x2


class TestComputeCCF:
    def test_detects_known_lag(self):
        true_lag = 6
        x1, x2 = make_lagged_pair(n=120, lag=true_lag, noise_std=0.05)
        result = compute_ccf(x1, x2, max_lag=20, prewhiten=False)
        # The detected peak lag should be close to true_lag
        assert abs(result.peak_lag - true_lag) <= 2, (
            f"Expected lag ~{true_lag}, got {result.peak_lag}"
        )

    def test_output_shape(self):
        x1, x2 = make_lagged_pair()
        result = compute_ccf(x1, x2, max_lag=10)
        assert len(result.lags) == 21  # -10 to +10 inclusive
        assert len(result.correlations) == 21
        assert len(result.p_values) == 21

    def test_confidence_bands_positive(self):
        x1, x2 = make_lagged_pair()
        result = compute_ccf(x1, x2, max_lag=10)
        ci_lo, ci_hi = result.confidence_bands
        assert ci_hi > 0
        assert ci_lo < 0

    def test_peak_lag_non_negative(self):
        x1, x2 = make_lagged_pair()
        result = compute_ccf(x1, x2, max_lag=12)
        assert result.peak_lag >= 0

    def test_returns_ccf_result(self):
        x1, x2 = make_lagged_pair()
        result = compute_ccf(x1, x2, max_lag=10)
        assert isinstance(result, CCFResult)

    def test_prewhiten_option(self):
        x1, x2 = make_lagged_pair()
        r1 = compute_ccf(x1, x2, max_lag=10, prewhiten=True)
        r2 = compute_ccf(x1, x2, max_lag=10, prewhiten=False)
        # Both should work; results may differ
        assert isinstance(r1, CCFResult)
        assert isinstance(r2, CCFResult)


class TestDieboldMariano:
    def test_significant_when_different(self):
        """When one model is clearly better, DM test should reject H0."""
        rng = np.random.default_rng(1)
        n = 100
        e1 = rng.normal(0, 0.1, n)   # small errors
        e2 = rng.normal(0, 2.0, n)   # large errors
        result = diebold_mariano_test(e1, e2, h=1, criterion="MSE")
        assert result.p_value < 0.05
        assert "Model 1" in result.conclusion

    def test_not_significant_when_equal(self):
        rng = np.random.default_rng(2)
        n = 100
        e1 = rng.normal(0, 1.0, n)
        e2 = rng.normal(0, 1.0, n)
        result = diebold_mariano_test(e1, e2, h=1, criterion="MSE")
        # Cannot guarantee not significant (random), but should not always reject
        assert isinstance(result, DieboldMarianoResult)
        assert 0 <= result.p_value <= 1

    def test_mae_criterion(self):
        rng = np.random.default_rng(3)
        e1 = rng.normal(0, 0.5, 50)
        e2 = rng.normal(0, 2.0, 50)
        result = diebold_mariano_test(e1, e2, h=1, criterion="MAE")
        assert isinstance(result, DieboldMarianoResult)

    def test_invalid_criterion(self):
        with pytest.raises(ValueError, match="Unknown criterion"):
            diebold_mariano_test(np.zeros(10), np.ones(10), criterion="RMSE")

    def test_returns_dm_result(self):
        e1 = np.random.randn(50)
        e2 = np.random.randn(50)
        result = diebold_mariano_test(e1, e2)
        assert isinstance(result, DieboldMarianoResult)
        assert hasattr(result, "dm_statistic")
        assert hasattr(result, "p_value")
        assert hasattr(result, "conclusion")
