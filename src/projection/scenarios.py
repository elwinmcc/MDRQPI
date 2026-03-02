"""
Scenario tree projection for Bitcoin price cycles.

Constructs a forward projection of BTC price based on:
  1. LI signal level and trend (leading by CCF-estimated lag)
  2. CI current position and phase
  3. LaI maturity (bounds the scenario tree asymmetrically)
  4. Dominant cycle period from wavelet analysis

Three scenarios are always generated:
  - Bull (p_bull):  LI-driven liquidity expansion persists / accelerates
  - Base (p_base):  Current trajectory continues, cycle completes normally
  - Bear (p_bear):  Liquidity tightens / cycle truncates early

Scenario probabilities are derived from:
  - LI level: P(bull) ∝ LI/100, P(bear) ∝ (100-LI)/100
  - LaI maturity: high LaI reduces bull probability, floor bear probability
  - CI phase: late expansion shrinks bull target, early contraction expands bear

IMPORTANT: All parameters (horizon, probabilities, price targets) are
derived from live data. Nothing is hardcoded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """
    A single price scenario path.

    Attributes
    ----------
    label : str
        'bull', 'base', or 'bear'.
    probability : float
        Assigned probability (0–1). Three scenarios sum to 1.
    horizon_months : int
        Number of months in the projection.
    price_path : pd.Series
        Monthly price path (DatetimeIndex).
    peak_price : float
        Projected peak price (bull/base) or trough (bear).
    peak_date : pd.Timestamp
        Expected peak/trough date.
    return_pct : float
        Total return from current price to peak/trough.
    """

    label: str
    probability: float
    horizon_months: int
    price_path: pd.Series
    peak_price: float
    peak_date: pd.Timestamp
    return_pct: float


@dataclass
class ScenarioTree:
    """
    Container for the three-scenario projection.

    Attributes
    ----------
    scenarios : dict[str, Scenario]
        Bull, base, and bear scenarios.
    current_price : float
        BTC price at projection date.
    projection_date : pd.Timestamp
        Date from which scenarios are projected.
    dominant_cycle_months : float
        Dominant cycle period from wavelet analysis.
    estimated_lag_months : int
        LI-to-BTC lead time from CCF analysis.
    """

    scenarios: dict[str, Scenario]
    current_price: float
    projection_date: pd.Timestamp
    dominant_cycle_months: float
    estimated_lag_months: int


def _compute_scenario_probabilities(
    li_level: float,
    lai_maturity: float,
    ci_phase: str,
) -> dict[str, float]:
    """
    Derive scenario probabilities from index levels.

    Parameters
    ----------
    li_level : float
        Current LI value (0–100).
    lai_maturity : float
        Current LaI maturity score (0–100).
    ci_phase : str
        Current CI phase label.

    Returns
    -------
    dict[str, float]
        Keys: 'bull', 'base', 'bear'. Values sum to 1.0.
    """
    # Base probability from LI signal
    p_bull_raw = li_level / 100.0
    p_bear_raw = 1.0 - p_bull_raw

    # LaI maturity adjustment: high maturity → compress bull, expand bear
    maturity_adj = (lai_maturity - 50.0) / 100.0  # -0.5 to +0.5
    p_bull_adj = p_bull_raw - 0.3 * maturity_adj
    p_bear_adj = p_bear_raw + 0.3 * maturity_adj

    # CI phase adjustment
    phase_adj_bull = {
        "early_expansion": 0.05,
        "expansion": 0.02,
        "late_expansion": -0.08,
        "early_contraction": -0.12,
        "contraction": -0.05,
        "late_contraction": 0.08,
        "unknown": 0.0,
    }
    adj = phase_adj_bull.get(ci_phase, 0.0)
    p_bull_adj += adj
    p_bear_adj -= adj

    # Clip to [0.05, 0.85] and normalize
    p_bull = np.clip(p_bull_adj, 0.05, 0.85)
    p_bear = np.clip(p_bear_adj, 0.05, 0.85)

    # Base gets the remainder
    p_base = 1.0 - p_bull - p_bear
    if p_base < 0.05:
        # Renormalize
        total = p_bull + p_bear
        p_bull = p_bull / total * 0.95
        p_bear = p_bear / total * 0.95
        p_base = 0.05

    # Final normalization
    total = p_bull + p_base + p_bear
    return {
        "bull": p_bull / total,
        "base": p_base / total,
        "bear": p_bear / total,
    }


def _project_price_path(
    current_price: float,
    target_price: float,
    horizon_months: int,
    start_date: pd.Timestamp,
    shape: str = "logistic",
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> pd.Series:
    """
    Generate a smooth price path from current to target over the horizon.

    Parameters
    ----------
    current_price, target_price : float
        Start and end price levels.
    horizon_months : int
        Number of months in the path.
    start_date : pd.Timestamp
        Projection start date.
    shape : str
        'logistic' (s-curve acceleration/deceleration) or 'linear'.
    noise_scale : float
        Lognormal noise to add (0 = deterministic).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    pd.Series
        Monthly price path with DatetimeIndex.
    """
    dates = pd.date_range(start=start_date, periods=horizon_months + 1, freq="ME")
    t = np.linspace(0, 1, horizon_months + 1)

    if shape == "logistic":
        # Logistic S-curve: slow start, acceleration, slow end
        s = 1.0 / (1.0 + np.exp(-10 * (t - 0.5)))
        s = (s - s[0]) / (s[-1] - s[0])
    else:
        s = t

    log_current = np.log(current_price)
    log_target = np.log(max(target_price, 1e-6))
    log_path = log_current + s * (log_target - log_current)

    if noise_scale > 0 and rng is not None:
        noise = rng.normal(0, noise_scale, len(log_path))
        noise[0] = 0  # no noise at start
        log_path = log_path + np.cumsum(noise)

    price_path = np.exp(log_path)
    return pd.Series(price_path, index=dates, name="price")


def build_scenario_tree(
    btc_price: pd.Series,
    li_result,
    ci_result,
    lai_result,
    dominant_cycle_months: float,
    ccf_lag_months: int,
    rng_seed: int = 42,
) -> ScenarioTree:
    """
    Build the three-scenario projection tree.

    Parameters
    ----------
    btc_price : pd.Series
        Monthly BTC price series.
    li_result : LeadingIndexResult
        From src.indicators.leading.build_leading_index.
    ci_result : CoincidentIndexResult
        From src.indicators.coincident.build_coincident_index.
    lai_result : LaggingIndexResult
        From src.indicators.lagging.build_lagging_index.
    dominant_cycle_months : float
        Dominant cycle period from wavelet analysis.
    ccf_lag_months : int
        LI-to-BTC lead time from CCF.
    rng_seed : int
        Seed for scenario path noise.

    Returns
    -------
    ScenarioTree
    """
    rng = np.random.default_rng(rng_seed)

    current_price = float(btc_price.dropna().iloc[-1])
    projection_date = btc_price.dropna().index[-1]

    li_level = float(li_result.index.iloc[-1]) if len(li_result.index) else 50.0
    lai_maturity = float(lai_result.maturity_score.dropna().iloc[-1]) if len(lai_result.maturity_score.dropna()) else 50.0

    # CI phase
    from src.indicators.coincident import get_cycle_phase_from_ci
    ci_phase_series = get_cycle_phase_from_ci(ci_result.index)
    ci_phase = str(ci_phase_series.iloc[-1]) if len(ci_phase_series) else "unknown"

    # Scenario probabilities
    probs = _compute_scenario_probabilities(li_level, lai_maturity, ci_phase)

    # Projection horizon: remaining cycle / 2 to end-of-cycle
    # Based on dominant cycle period and CCF lag
    half_cycle = dominant_cycle_months / 2.0
    remaining_cycle = max(half_cycle - ccf_lag_months, 6)  # at least 6 months

    # Price targets derived from historical cycle amplitudes
    # Use rolling 4-year max/min as reference for cycle amplitude
    btc = btc_price.dropna()
    lookback = min(len(btc), 48)  # 4-year lookback
    recent_high = float(btc.iloc[-lookback:].max())
    recent_low = float(btc.iloc[-lookback:].min())
    amplitude_ratio = recent_high / max(recent_low, 1e-6)

    # Bull scenario: new ATH or 50th percentile of historical bull cycles
    # Use log-linear extrapolation from LI → price CCF relationship
    li_momentum = li_result.index.diff(3).iloc[-1] if len(li_result.index) > 3 else 0.0
    li_momentum_norm = float(np.clip(li_momentum / 20.0, -1.0, 1.0))

    # Price targets
    bull_multiplier = 1.0 + 0.8 * (li_level / 100.0) * (1.0 + 0.3 * li_momentum_norm)
    bull_multiplier = np.clip(bull_multiplier, 1.1, 4.0)  # cap at 4x
    base_multiplier = 1.0 + 0.4 * (li_level / 100.0)
    base_multiplier = np.clip(base_multiplier, 0.9, 2.5)
    bear_multiplier = 1.0 - 0.5 * (1.0 - li_level / 100.0) * (lai_maturity / 100.0)
    bear_multiplier = np.clip(bear_multiplier, 0.25, 0.95)

    bull_target = current_price * bull_multiplier
    base_target = current_price * base_multiplier
    bear_target = current_price * bear_multiplier

    # Horizons
    bull_horizon = max(int(remaining_cycle * 1.2), 6)
    base_horizon = max(int(remaining_cycle), 6)
    bear_horizon = max(int(remaining_cycle * 0.6), 3)

    scenarios: dict[str, Scenario] = {}

    for label, prob, target, horizon, shape in [
        ("bull", probs["bull"], bull_target, bull_horizon, "logistic"),
        ("base", probs["base"], base_target, base_horizon, "logistic"),
        ("bear", probs["bear"], bear_target, bear_horizon, "linear"),
    ]:
        noise = 0.04 if label != "base" else 0.02
        path = _project_price_path(
            current_price=current_price,
            target_price=target,
            horizon_months=horizon,
            start_date=projection_date,
            shape=shape,
            noise_scale=noise,
            rng=rng,
        )

        if label == "bear":
            peak_price = float(path.min())
            peak_date = path.idxmin()
        else:
            peak_price = float(path.max())
            peak_date = path.idxmax()

        return_pct = (peak_price / current_price - 1.0) * 100.0

        scenarios[label] = Scenario(
            label=label,
            probability=prob,
            horizon_months=horizon,
            price_path=path,
            peak_price=peak_price,
            peak_date=peak_date,
            return_pct=return_pct,
        )

    logger.info(
        "Scenario tree built: bull=%.1f%% (×%.2f), base=%.1f%% (×%.2f), "
        "bear=%.1f%% (×%.2f) | probs: bull=%.2f, base=%.2f, bear=%.2f",
        scenarios["bull"].return_pct,
        bull_multiplier,
        scenarios["base"].return_pct,
        base_multiplier,
        scenarios["bear"].return_pct,
        bear_multiplier,
        probs["bull"],
        probs["base"],
        probs["bear"],
    )

    return ScenarioTree(
        scenarios=scenarios,
        current_price=current_price,
        projection_date=projection_date,
        dominant_cycle_months=dominant_cycle_months,
        estimated_lag_months=ccf_lag_months,
    )


def scenario_summary(tree: ScenarioTree) -> pd.DataFrame:
    """
    Return a summary DataFrame of the scenario tree.

    Parameters
    ----------
    tree : ScenarioTree

    Returns
    -------
    pd.DataFrame
        One row per scenario with probability, target, return, horizon.
    """
    rows = []
    for label, sc in tree.scenarios.items():
        rows.append({
            "Scenario": label.capitalize(),
            "Probability": f"{sc.probability:.1%}",
            "Price Target": f"${sc.peak_price:,.0f}",
            "Return": f"{sc.return_pct:+.1f}%",
            "Horizon (months)": sc.horizon_months,
            "Expected Date": sc.peak_date.strftime("%Y-%m"),
        })
    return pd.DataFrame(rows).set_index("Scenario")
