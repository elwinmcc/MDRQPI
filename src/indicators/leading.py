"""
Leading Index (LI) — 7-component equal-weight diffusion index.

Construction
------------
Each of the 7 components is binarized: 1 if the raw value exceeds its
6-month trailing moving average (expansion signal), else 0.
The index is the mean of all binary signals × 100, so it ranges from
0 (all components contracting) to 100 (all expanding).

LI > 50  → net expansionary monetary conditions → bullish signal
LI < 50  → net contractionary conditions        → bearish signal
LI crossing 50 → cycle turn signal

Components
----------
1. WM2NS  — M2 Money Supply (YoY change momentum)
2. WALCL  — Fed Balance Sheet (QE/QT signal)
3. WTREGEN — TGA Balance (inverted; drain → injection)
4. RRPONTSYD — Overnight RRP (inverted; drawdown → liquidity release)
5. BAMLH0A0HYM2 — HY Credit Spread (inverted; tightening = bearish)
6. T10Y2Y — Yield Curve Slope (steepening = bullish)
7. DTWEXBGS — Trade-Weighted USD (inverted; weakening dollar = bullish)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Component definitions: (series_id, invert)
# invert=True means higher values are bearish (we flip before binarizing)
COMPONENT_SPEC: list[tuple[str, bool, str]] = [
    ("WM2NS",       False, "M2 Money Supply"),
    ("WALCL",       False, "Fed Balance Sheet"),
    ("WTREGEN",     True,  "TGA Balance (inv)"),
    ("RRPONTSYD",   True,  "Overnight RRP (inv)"),
    ("BAMLH0A0HYM2", True, "HY Credit Spread (inv)"),
    ("T10Y2Y",      False, "Yield Curve Slope"),
    ("DTWEXBGS",    True,  "Trade-Weighted USD (inv)"),
]

MA_WINDOW = 6  # 6-month trailing MA for binarization


@dataclass
class LeadingIndexResult:
    """
    Output container for the Leading Index.

    Attributes
    ----------
    index : pd.Series
        Monthly LI values (0–100).
    components : pd.DataFrame
        Binary signals for each of the 7 components.
    raw_components : pd.DataFrame
        Raw (possibly inverted) component values before binarization.
    """

    index: pd.Series
    components: pd.DataFrame
    raw_components: pd.DataFrame


def _compute_binary_signal(series: pd.Series, invert: bool, ma_window: int) -> pd.Series:
    """
    Compute the binary expansion/contraction signal for one component.

    Parameters
    ----------
    series : pd.Series
        Monthly time series.
    invert : bool
        If True, negate the series before comparison (bearish = high values).
    ma_window : int
        Trailing MA window in months.

    Returns
    -------
    pd.Series
        Binary series: 1 (expanding) or 0 (contracting).
    """
    s = series.copy()
    if invert:
        s = -s
    ma = s.rolling(ma_window, min_periods=ma_window // 2).mean()
    binary = (s > ma).astype(int)
    return binary


def build_leading_index(
    data: pd.DataFrame,
    ma_window: int = MA_WINDOW,
    component_spec: list[tuple[str, bool, str]] | None = None,
) -> LeadingIndexResult:
    """
    Construct the 7-component equal-weight Leading Index diffusion index.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing all required series as columns.
        Must include: WM2NS, WALCL, WTREGEN, RRPONTSYD,
        BAMLH0A0HYM2, T10Y2Y, DTWEXBGS.
    ma_window : int
        Window for the trailing MA used in binarization.
    component_spec : list of (series_id, invert, label), optional
        Override the default component list.

    Returns
    -------
    LeadingIndexResult
        Computed LI and component signals.

    Raises
    ------
    ValueError
        If any required series is missing from *data*.
    """
    spec = component_spec or COMPONENT_SPEC
    missing = [sid for sid, _, _ in spec if sid not in data.columns]
    if missing:
        raise ValueError(f"Missing series in data: {missing}")

    binary_signals: dict[str, pd.Series] = {}
    raw_components: dict[str, pd.Series] = {}

    for series_id, invert, label in spec:
        raw = data[series_id].dropna()
        raw_components[label] = raw
        binary = _compute_binary_signal(raw, invert=invert, ma_window=ma_window)
        binary_signals[label] = binary

    components_df = pd.DataFrame(binary_signals)

    # Equal-weight diffusion: mean of binary signals × 100
    # Only compute index where at least 4 of 7 components have data
    n_components = len(spec)
    min_valid = n_components - 3  # tolerate up to 3 missing at edges

    li = components_df.mean(axis=1, skipna=True) * 100.0

    # Mask months where fewer than min_valid components have data
    valid_count = components_df.notna().sum(axis=1)
    li = li.where(valid_count >= min_valid)

    li.name = "LI"
    li = li.dropna()

    logger.info(
        "Built Leading Index: %d monthly observations (%s to %s), "
        "current value=%.1f",
        len(li),
        li.index[0].date() if len(li) else "N/A",
        li.index[-1].date() if len(li) else "N/A",
        li.iloc[-1] if len(li) else float("nan"),
    )

    raw_df = pd.DataFrame(raw_components)
    return LeadingIndexResult(
        index=li,
        components=components_df,
        raw_components=raw_df,
    )


def compute_li_momentum(li: pd.Series, window: int = 3) -> pd.Series:
    """
    Compute the 3-month rate of change of the Leading Index.

    Useful for detecting acceleration/deceleration in liquidity conditions
    before the level signal crosses 50.

    Parameters
    ----------
    li : pd.Series
        Monthly Leading Index series.
    window : int
        Number of months for the rate-of-change calculation.

    Returns
    -------
    pd.Series
        LI momentum (percentage points change over *window* months).
    """
    return li.diff(window).rename("LI_momentum")


def validate_li_against_btc(
    li: pd.Series,
    btc_price: pd.Series,
    max_lag: int = 18,
) -> dict[str, float]:
    """
    Validate Leading Index predictive power against BTC price via CCF.

    Computes the cross-correlation between LI and future BTC returns
    at lags 0 to max_lag months. Reports the peak correlation and its lag.

    Parameters
    ----------
    li : pd.Series
        Monthly Leading Index (0–100).
    btc_price : pd.Series
        Monthly BTC price in USD.
    max_lag : int
        Maximum lead (LI leads BTC) to test.

    Returns
    -------
    dict[str, float]
        Keys: 'peak_correlation', 'peak_lag_months', 'correlation_at_0'.
    """
    btc_ret = np.log(btc_price).diff().dropna()
    aligned = pd.concat([li, btc_ret], axis=1).dropna()
    li_aligned = aligned.iloc[:, 0]
    btc_aligned = aligned.iloc[:, 1]

    correlations: dict[int, float] = {}
    for lag in range(0, max_lag + 1):
        if lag == 0:
            corr = li_aligned.corr(btc_aligned)
        else:
            # LI at t, BTC return at t+lag → positive lag = LI leads
            corr = li_aligned.iloc[:-lag].corr(btc_aligned.iloc[lag:])
        correlations[lag] = corr

    peak_lag = max(correlations, key=lambda k: abs(correlations[k]))
    return {
        "peak_correlation": correlations[peak_lag],
        "peak_lag_months": float(peak_lag),
        "correlation_at_0": correlations[0],
        "all_correlations": correlations,
    }
