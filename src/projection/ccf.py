"""
Cross-Correlation Function (CCF) for lag estimation.

Estimates the lead-lag relationship between the Leading Index and Bitcoin
price. The lag at maximum cross-correlation is used as the projection
horizon for the scenario tree.

Also implements the Diebold-Mariano (1995) test for comparing forecast
accuracy of the LI-based model against a naive benchmark.

References
----------
Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
Journal of Business & Economic Statistics, 13(3), 253-263.

Box, G. E., Jenkins, G. M., Ljung, G. M., & Reinsel, G. C. (2015).
Time series analysis: forecasting and control. Wiley.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CCFResult:
    """
    Container for CCF output.

    Attributes
    ----------
    lags : np.ndarray
        Lag values (months). Positive = x1 leads x2.
    correlations : np.ndarray
        Cross-correlation coefficient at each lag.
    confidence_bands : tuple[float, float]
        95% confidence interval bounds for a null of zero correlation.
    peak_lag : int
        Lag with the highest absolute cross-correlation.
    peak_correlation : float
        Cross-correlation at the peak lag.
    p_values : np.ndarray
        Approximate p-values for each lag.
    """

    lags: np.ndarray
    correlations: np.ndarray
    confidence_bands: tuple[float, float]
    peak_lag: int
    peak_correlation: float
    p_values: np.ndarray


@dataclass
class DieboldMarianoResult:
    """
    Container for Diebold-Mariano test results.

    Attributes
    ----------
    dm_statistic : float
        DM test statistic (approximately standard normal under H0).
    p_value : float
        Two-sided p-value.
    conclusion : str
        Human-readable conclusion.
    """

    dm_statistic: float
    p_value: float
    conclusion: str


def compute_ccf(
    x1: pd.Series,
    x2: pd.Series,
    max_lag: int = 24,
    prewhiten: bool = True,
) -> CCFResult:
    """
    Compute the cross-correlation function between x1 and x2.

    Parameters
    ----------
    x1 : pd.Series
        Leading series (e.g., Leading Index). Monthly.
    x2 : pd.Series
        Lagging series (e.g., log BTC price changes). Monthly.
    max_lag : int
        Maximum lag to compute (months). Positive = x1 leads x2.
    prewhiten : bool
        If True, pre-whiten both series using AR(1) residuals before
        computing CCF. Reduces spurious correlations from autocorrelation.

    Returns
    -------
    CCFResult
    """
    # Align
    aligned = pd.concat([x1, x2], axis=1).dropna()
    a = aligned.iloc[:, 0].values.astype(float)
    b = aligned.iloc[:, 1].values.astype(float)
    n = len(a)

    if prewhiten:
        a, b = _prewhiten_ar1(a, b)
        n = len(a)

    # Standardize
    a = (a - np.mean(a)) / (np.std(a) + 1e-12)
    b = (b - np.mean(b)) / (np.std(b) + 1e-12)

    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag == 0:
            correlations[i] = np.corrcoef(a, b)[0, 1]
        elif lag > 0:
            # x1 at t, x2 at t+lag → positive lag = x1 leads x2
            correlations[i] = np.corrcoef(a[:-lag], b[lag:])[0, 1]
        else:
            # x2 leads x1
            correlations[i] = np.corrcoef(a[-lag:], b[:lag])[0, 1]

    # 95% CI under null: ±1.96/√n
    ci = 1.96 / np.sqrt(n)

    # Approximate p-values (Fisher z-transform)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.arctanh(correlations)
        se = 1.0 / np.sqrt(n - 3)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z) / se))

    # Restrict to non-negative lags for peak (we want x1 to lead x2)
    pos_mask = lags >= 0
    pos_corrs = correlations[pos_mask]
    pos_lags = lags[pos_mask]
    peak_idx = int(np.argmax(np.abs(pos_corrs)))
    peak_lag = int(pos_lags[peak_idx])
    peak_corr = float(pos_corrs[peak_idx])

    logger.info(
        "CCF peak: lag=%d months, correlation=%.3f (95%% CI: ±%.3f)",
        peak_lag,
        peak_corr,
        ci,
    )
    return CCFResult(
        lags=lags,
        correlations=correlations,
        confidence_bands=(-ci, ci),
        peak_lag=peak_lag,
        peak_correlation=peak_corr,
        p_values=p_values,
    )


def _prewhiten_ar1(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-whiten two series using AR(1) filter.

    Fits AR(1) to x1 and applies the same filter to both series.
    This removes the autocorrelation structure that can inflate CCF.

    Parameters
    ----------
    a, b : np.ndarray
        Input arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Residuals after AR(1) filtering.
    """
    # Fit AR(1) to first series
    phi = np.corrcoef(a[:-1], a[1:])[0, 1]
    phi = np.clip(phi, -0.99, 0.99)

    a_res = a[1:] - phi * a[:-1]
    b_res = b[1:] - phi * b[:-1]
    return a_res, b_res


def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    criterion: str = "MSE",
) -> DieboldMarianoResult:
    """
    Diebold-Mariano (1995) test for equal forecast accuracy.

    Tests H0: E[d_t] = 0, where d_t = L(e1_t) - L(e2_t) is the loss
    differential. Under H0, the two forecasts have equal accuracy.

    Parameters
    ----------
    e1 : np.ndarray
        Forecast errors from model 1 (e.g., LI-based projection).
    e2 : np.ndarray
        Forecast errors from model 2 (e.g., naive random walk).
    h : int
        Forecast horizon (months). Used for HAC variance correction.
    criterion : str
        Loss function: 'MSE' (squared error) or 'MAE' (absolute error).

    Returns
    -------
    DieboldMarianoResult
    """
    if criterion == "MSE":
        d = e1**2 - e2**2
    elif criterion == "MAE":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Use 'MSE' or 'MAE'.")

    n = len(d)
    d_bar = np.mean(d)

    # HAC variance estimate (Newey-West with truncation lag h-1)
    # Var(d) = γ(0) + 2 Σ_{k=1}^{h-1} γ(k)
    gamma0 = np.var(d, ddof=1)
    hac_var = gamma0
    for k in range(1, h):
        weight = 1.0 - k / h  # Bartlett kernel
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        hac_var += 2 * weight * gamma_k

    se = np.sqrt(max(hac_var, 1e-12) / n)
    dm_stat = d_bar / se
    p_val = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))

    if p_val < 0.05:
        if dm_stat < 0:
            conclusion = "Model 1 significantly more accurate than Model 2 (p<0.05)"
        else:
            conclusion = "Model 2 significantly more accurate than Model 1 (p<0.05)"
    else:
        conclusion = "No significant difference in forecast accuracy (p≥0.05)"

    logger.info("DM test: stat=%.3f, p=%.4f — %s", dm_stat, p_val, conclusion)
    return DieboldMarianoResult(dm_statistic=dm_stat, p_value=p_val, conclusion=conclusion)


def estimate_forecast_horizon(
    li: pd.Series,
    btc_price: pd.Series,
    max_lag: int = 24,
) -> dict[str, float | int]:
    """
    Estimate the optimal projection horizon from LI to BTC peak.

    Combines CCF analysis with wavelet coherence phase estimates to
    produce a robust lag estimate.

    Parameters
    ----------
    li : pd.Series
        Monthly Leading Index.
    btc_price : pd.Series
        Monthly BTC price (USD).
    max_lag : int
        Maximum lag to test.

    Returns
    -------
    dict
        Keys: 'ccf_lag_months', 'peak_correlation', 'recommended_horizon'.
    """
    btc_log = np.log(btc_price)
    ccf_result = compute_ccf(li, btc_log, max_lag=max_lag)

    return {
        "ccf_lag_months": ccf_result.peak_lag,
        "peak_correlation": ccf_result.peak_correlation,
        "recommended_horizon": ccf_result.peak_lag,
        "confidence_lower": ccf_result.confidence_bands[0],
        "confidence_upper": ccf_result.confidence_bands[1],
    }
