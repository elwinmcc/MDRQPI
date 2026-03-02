"""
Wavelet coherence between the Leading Index and Bitcoin price.

Wavelet coherence R²(t, s) measures the local correlation between two
series in time-frequency space. High coherence at a given period and time
indicates that the two series are co-moving at that frequency.

The cross-wavelet phase angle φ(t, s) indicates which series leads:
  - Positive φ (0 to π): BTC price leads LI
  - Negative φ (-π to 0): LI leads BTC price

References
----------
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

from src.spectral.cwt import _morlet_scales, _scale_to_period, OMEGA_0, MIN_PERIOD, MAX_PERIOD

logger = logging.getLogger(__name__)


@dataclass
class CoherenceResult:
    """
    Container for wavelet coherence output.

    Attributes
    ----------
    coherence : np.ndarray
        Squared wavelet coherence R²(t,s), shape (n_scales, n_time). Values 0–1.
    phase : np.ndarray
        Cross-wavelet phase angle in radians, shape (n_scales, n_time).
        Positive = x2 leads x1.
    periods : np.ndarray
        Period axis (months).
    time_index : pd.DatetimeIndex
        Time axis.
    cone_of_influence : np.ndarray
        COI periods at each time step.
    avg_coherence_by_period : pd.Series
        Time-averaged coherence for each period (useful for summary).
    dominant_lead_months : pd.Series
        Estimated lead of LI over BTC at the most coherent period, per month.
    """

    coherence: np.ndarray
    phase: np.ndarray
    periods: np.ndarray
    time_index: pd.DatetimeIndex
    cone_of_influence: np.ndarray
    avg_coherence_by_period: pd.Series
    dominant_lead_months: pd.Series


def _smooth_wavelet_spectrum(
    spectrum: np.ndarray, scales: np.ndarray, smoothing_radius: int = 3
) -> np.ndarray:
    """
    Apply scale-adaptive smoothing to a wavelet spectrum.

    Uses a Gaussian kernel in the time direction and a boxcar in scale.

    Parameters
    ----------
    spectrum : np.ndarray
        Complex or real spectrum, shape (n_scales, n_time).
    scales : np.ndarray
        Wavelet scales.
    smoothing_radius : int
        Half-width of smoothing window (time direction).

    Returns
    -------
    np.ndarray
        Smoothed spectrum.
    """
    smoothed = np.zeros_like(spectrum, dtype=complex)
    kernel_size = 2 * smoothing_radius + 1

    for i in range(spectrum.shape[0]):
        # Time smoothing with scale-adaptive window
        s = scales[i]
        w = max(1, int(np.round(s / 2)))
        from scipy.ndimage import uniform_filter1d
        if np.iscomplexobj(spectrum):
            smoothed[i, :] = (
                uniform_filter1d(np.real(spectrum[i, :]), w)
                + 1j * uniform_filter1d(np.imag(spectrum[i, :]), w)
            )
        else:
            smoothed[i, :] = uniform_filter1d(np.real(spectrum[i, :]), w)

    # Scale smoothing (boxcar over 3 scales)
    from scipy.ndimage import uniform_filter1d as uf
    if np.iscomplexobj(smoothed):
        out = (
            uf(np.real(smoothed), 3, axis=0)
            + 1j * uf(np.imag(smoothed), 3, axis=0)
        )
    else:
        out = uf(np.real(smoothed), 3, axis=0)
    return out


def compute_coherence(
    x1: pd.Series,
    x2: pd.Series,
    min_period: int = MIN_PERIOD,
    max_period: int = MAX_PERIOD,
    n_voices: int = 32,
    omega0: float = OMEGA_0,
) -> CoherenceResult:
    """
    Compute wavelet coherence between two monthly series.

    Typically called with x1=LI and x2=BTC price (log).

    Parameters
    ----------
    x1 : pd.Series
        First monthly series (e.g., Leading Index).
    x2 : pd.Series
        Second monthly series (e.g., log BTC price).
    min_period, max_period : int
        Period range in months.
    n_voices : int
        Number of scales.
    omega0 : float
        Morlet ω₀.

    Returns
    -------
    CoherenceResult
    """
    # Align to common index
    aligned = pd.concat([x1, x2], axis=1).dropna()
    s1 = (aligned.iloc[:, 0] - aligned.iloc[:, 0].mean()) / (aligned.iloc[:, 0].std() + 1e-12)
    s2 = (aligned.iloc[:, 1] - aligned.iloc[:, 1].mean()) / (aligned.iloc[:, 1].std() + 1e-12)
    n = len(s1)

    scales = _morlet_scales(min_period, max_period, n_voices, omega0=omega0)
    periods = _scale_to_period(scales, omega0=omega0)

    wavelet = "cmor1.5-6.0"
    try:
        W1, _ = pywt.cwt(s1.values, scales, wavelet, sampling_period=1.0)
        W2, _ = pywt.cwt(s2.values, scales, wavelet, sampling_period=1.0)
    except Exception:
        wavelet = "morl"
        W1, _ = pywt.cwt(s1.values, scales, wavelet, sampling_period=1.0)
        W2, _ = pywt.cwt(s2.values, scales, wavelet, sampling_period=1.0)

    # Cross-wavelet spectrum
    Wxy = W1 * np.conj(W2)

    # Smooth spectra for coherence
    S11 = _smooth_wavelet_spectrum(np.abs(W1) ** 2, scales)
    S22 = _smooth_wavelet_spectrum(np.abs(W2) ** 2, scales)
    Sxy = _smooth_wavelet_spectrum(Wxy, scales)

    # Squared wavelet coherence
    denom = (np.abs(S11) * np.abs(S22)) + 1e-12
    coherence = np.abs(Sxy) ** 2 / denom
    coherence = np.clip(np.real(coherence), 0.0, 1.0)

    # Phase angle
    phase = np.angle(Sxy)  # radians: positive = x2 leads x1

    # COI
    t = np.arange(n)
    edge_dist = np.minimum(t, n - 1 - t)
    coi = np.sqrt(2) * 2 * edge_dist

    # Time-averaged coherence by period (outside COI)
    avg_coh = np.nanmean(coherence, axis=1)
    avg_coherence_by_period = pd.Series(avg_coh, index=periods, name="avg_coherence")

    # Dominant lead estimate: at the most coherent period, what is the phase lag?
    # Phase lag in months = φ × T / (2π), where φ is the cross-wavelet phase
    # Positive phase → x2 leads x1 → LI lags BTC (if x1=LI, x2=BTC)
    best_scale_idx = np.argmax(avg_coh)
    phase_at_best = phase[best_scale_idx, :]
    T_best = periods[best_scale_idx]
    lead_months = -phase_at_best * T_best / (2 * np.pi)  # negate: LI leads BTC if positive
    dominant_lead = pd.Series(lead_months, index=aligned.index, name="li_lead_months")

    logger.info(
        "Wavelet coherence: dominant period=%.1f months, "
        "avg coherence at dominant=%.3f, "
        "mean LI lead=%.1f months",
        T_best,
        avg_coh[best_scale_idx],
        float(dominant_lead.mean()),
    )
    return CoherenceResult(
        coherence=coherence,
        phase=phase,
        periods=periods,
        time_index=aligned.index,
        cone_of_influence=coi,
        avg_coherence_by_period=avg_coherence_by_period,
        dominant_lead_months=dominant_lead,
    )


def summarize_coherence(result: CoherenceResult) -> dict[str, float]:
    """
    Summarize wavelet coherence results.

    Parameters
    ----------
    result : CoherenceResult

    Returns
    -------
    dict[str, float]
        Summary statistics.
    """
    avg_coh = result.avg_coherence_by_period
    peak_period = float(avg_coh.idxmax())
    peak_coherence = float(avg_coh.max())
    recent_lead = float(result.dominant_lead_months.iloc[-1])
    mean_lead = float(result.dominant_lead_months.mean())

    # Count time periods with high coherence (R² > 0.5)
    frac_high_coh = float((result.coherence > 0.5).mean())

    return {
        "peak_coherence_period_months": peak_period,
        "peak_coherence_r2": peak_coherence,
        "recent_li_lead_months": recent_lead,
        "mean_li_lead_months": mean_lead,
        "fraction_high_coherence": frac_high_coh,
    }
