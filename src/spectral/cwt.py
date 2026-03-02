"""
Morlet Continuous Wavelet Transform (CWT) for cycle analysis.

Uses PyWavelets (pywt.cwt) with the Morlet wavelet (ω₀=6).
Analyzes periods from 12 to 120 months to capture:
  - Short crypto cycles (~12-24 months)
  - Bitcoin halving cycles (~48 months)
  - Long macro cycles (~72-120 months)

References
----------
Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet analysis.
Bulletin of the American Meteorological Society, 79(1), 61-78.

Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004). Application of the
cross wavelet transform and wavelet coherence to geophysical time series.
Nonlinear Processes in Geophysics, 11(5/6), 561-566.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pywt

logger = logging.getLogger(__name__)

OMEGA_0 = 6.0          # Standard Morlet ω₀
MIN_PERIOD = 12        # months
MAX_PERIOD = 120       # months
N_VOICES = 64          # number of scales (voices per octave × octaves)


@dataclass
class CWTResult:
    """
    Container for CWT output.

    Attributes
    ----------
    power : np.ndarray
        Wavelet power spectrum, shape (n_scales, n_time).
    scales : np.ndarray
        Wavelet scales used.
    periods : np.ndarray
        Corresponding periods in months.
    freqs : np.ndarray
        Corresponding frequencies (cycles/month).
    time_index : pd.DatetimeIndex
        Time axis of the input series.
    cone_of_influence : np.ndarray
        COI in months at each time step. Periods above the COI are
        edge-affected and should be treated with caution.
    dominant_period : pd.Series
        Time series of the dominant (peak power) period at each month.
    """

    power: np.ndarray
    scales: np.ndarray
    periods: np.ndarray
    freqs: np.ndarray
    time_index: pd.DatetimeIndex
    cone_of_influence: np.ndarray
    dominant_period: pd.Series


def _morlet_scales(
    min_period: float,
    max_period: float,
    n_voices: int,
    sampling_period: float = 1.0,
    omega0: float = OMEGA_0,
) -> np.ndarray:
    """
    Compute log-spaced scales for the Morlet wavelet.

    The relationship between scale s and period T for the Morlet wavelet is:
        T = (4π / (ω₀ + √(2 + ω₀²))) × s × sampling_period
    which simplifies to approximately T ≈ s for ω₀=6.

    Parameters
    ----------
    min_period, max_period : float
        Period range in months.
    n_voices : int
        Number of scales.
    sampling_period : float
        Sampling interval (1.0 for monthly).
    omega0 : float
        Morlet wavelet center frequency.

    Returns
    -------
    np.ndarray
        Array of wavelet scales.
    """
    # Exact period-to-scale conversion for Morlet
    # Period T = scale × (4π) / (ω₀ + √(2 + ω₀²))
    factor = (4 * np.pi) / (omega0 + np.sqrt(2 + omega0**2))
    s_min = min_period / factor / sampling_period
    s_max = max_period / factor / sampling_period
    scales = np.geomspace(s_min, s_max, n_voices)
    return scales


def _scale_to_period(scales: np.ndarray, omega0: float = OMEGA_0) -> np.ndarray:
    """Convert pywt Morlet scales to periods in months."""
    factor = (4 * np.pi) / (omega0 + np.sqrt(2 + omega0**2))
    return scales * factor


def _cone_of_influence(n: int, scales: np.ndarray) -> np.ndarray:
    """
    Compute the Cone of Influence (COI) for the Morlet wavelet.

    The COI is the region affected by edge effects. At time t from the
    edge, periods longer than the COI period are unreliable.

    Parameters
    ----------
    n : int
        Number of time steps.
    scales : np.ndarray
        Wavelet scales.

    Returns
    -------
    np.ndarray
        COI period (in months) at each time step, shape (n,).
    """
    # For Morlet, e-folding time τ = √2 × scale
    t = np.arange(n)
    edge_dist = np.minimum(t, n - 1 - t)
    # COI period is proportional to distance from edge
    coi_period = np.sqrt(2) * 2 * edge_dist
    return coi_period


def compute_cwt(
    series: pd.Series,
    min_period: int = MIN_PERIOD,
    max_period: int = MAX_PERIOD,
    n_voices: int = N_VOICES,
    omega0: float = OMEGA_0,
) -> CWTResult:
    """
    Compute the Morlet CWT power spectrum of a monthly time series.

    Parameters
    ----------
    series : pd.Series
        Monthly time series with DatetimeIndex. Will be standardized
        (zero mean, unit variance) before transform.
    min_period : int
        Minimum period to analyze (months).
    max_period : int
        Maximum period to analyze (months).
    n_voices : int
        Number of log-spaced scales (higher = finer frequency resolution).
    omega0 : float
        Morlet wavelet center frequency (standard = 6).

    Returns
    -------
    CWTResult
        Wavelet power, scales, periods, COI, and dominant period series.
    """
    s = series.dropna().copy()
    if len(s) < min_period * 2:
        raise ValueError(
            f"Series too short ({len(s)} obs) for wavelet analysis "
            f"(min_period={min_period})"
        )

    # Standardize
    x = (s.values - s.mean()) / (s.std() + 1e-12)
    n = len(x)

    # Build scales
    scales = _morlet_scales(
        min_period=float(min_period),
        max_period=float(max_period),
        n_voices=n_voices,
        omega0=omega0,
    )

    # pywt Morlet wavelet name
    wavelet_name = f"cmor{1.5:.1f}-{omega0:.1f}"
    # Use the standard 'morl' (real) or 'cmor' (complex) wavelet
    # pywt.cwt with 'cmor1.5-6.0' gives complex coefficients
    try:
        coeffs, freqs = pywt.cwt(x, scales, "cmor1.5-6.0", sampling_period=1.0)
    except Exception:
        # Fallback to real Morlet
        coeffs, freqs = pywt.cwt(x, scales, "morl", sampling_period=1.0)

    power = np.abs(coeffs) ** 2  # shape: (n_scales, n_time)
    periods = _scale_to_period(scales, omega0=omega0)
    coi = _cone_of_influence(n, scales)

    # Dominant period at each time step (argmax over scale axis)
    dom_scale_idx = np.argmax(power, axis=0)
    dom_periods = periods[dom_scale_idx]
    dominant_period = pd.Series(dom_periods, index=s.index, name="dominant_period_months")

    logger.info(
        "CWT complete: %d scales (%.0f–%.0f months), %d time steps",
        len(scales),
        periods[0],
        periods[-1],
        n,
    )
    return CWTResult(
        power=power,
        scales=scales,
        periods=periods,
        freqs=freqs,
        time_index=s.index,
        cone_of_influence=coi,
        dominant_period=dominant_period,
    )


def extract_cycle_band(
    series: pd.Series,
    low_period: float,
    high_period: float,
    omega0: float = OMEGA_0,
) -> pd.Series:
    """
    Bandpass-filter a monthly series to a specific cycle band using CWT.

    Reconstructs the signal by summing wavelet coefficients within the
    period band [low_period, high_period] months.

    Parameters
    ----------
    series : pd.Series
        Monthly time series.
    low_period, high_period : float
        Period band in months (e.g., 36, 60 for 3-5 year cycles).
    omega0 : float
        Morlet ω₀.

    Returns
    -------
    pd.Series
        Bandpass-filtered series.
    """
    s = series.dropna()
    x = s.values
    n = len(x)

    scales = _morlet_scales(
        min_period=max(low_period - 1, MIN_PERIOD),
        max_period=min(high_period + 1, MAX_PERIOD),
        n_voices=32,
        omega0=omega0,
    )
    periods = _scale_to_period(scales, omega0=omega0)
    mask = (periods >= low_period) & (periods <= high_period)

    try:
        coeffs, freqs = pywt.cwt(x, scales, "cmor1.5-6.0", sampling_period=1.0)
    except Exception:
        coeffs, freqs = pywt.cwt(x, scales, "morl", sampling_period=1.0)

    # Zero out coefficients outside the band
    filtered_coeffs = coeffs.copy()
    filtered_coeffs[~mask, :] = 0.0

    # iCWT approximation via summation (Torrence & Compo eq. 11)
    delta_j = np.log2(periods[-1] / periods[0]) / (len(periods) - 1) if len(periods) > 1 else 1.0
    psi0 = np.pi ** (-0.25)  # Morlet normalization
    dj = delta_j
    dt = 1.0

    reconstruction = (
        dj * np.sqrt(dt) / (psi0 * len(scales))
        * np.sum(np.real(filtered_coeffs) / np.sqrt(scales[:, np.newaxis]), axis=0)
    )
    # Re-scale to original amplitude
    reconstruction = reconstruction * s.std() + s.mean()

    return pd.Series(
        reconstruction,
        index=s.index,
        name=f"cwt_band_{int(low_period)}_{int(high_period)}m",
    )


def dominant_period_stats(cwt_result: CWTResult) -> dict[str, float]:
    """
    Summarize the dominant cycle period from a CWT result.

    Parameters
    ----------
    cwt_result : CWTResult

    Returns
    -------
    dict[str, float]
        Keys: 'mean_dominant_period', 'recent_dominant_period',
        'min_dominant_period', 'max_dominant_period'.
    """
    dp = cwt_result.dominant_period
    return {
        "mean_dominant_period": float(dp.mean()),
        "recent_dominant_period": float(dp.iloc[-1]),
        "min_dominant_period": float(dp.min()),
        "max_dominant_period": float(dp.max()),
    }
