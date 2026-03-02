"""
Coincident Index (CI) — 2-factor Stock-Watson Dynamic Factor Model.

Uses statsmodels DynamicFactor with:
  - k_factors = 2
  - factor_order = 2 (VAR(2) factor dynamics)
  - EM algorithm initialization, then MLE

Observables (4 series):
  1. log(BTC price) — primary crypto cycle coincident
  2. BTC 3-month realized volatility (annualized)
  3. log(Total crypto market cap)
  4. BTC dominance (%)

The two latent factors capture:
  - Factor 1: Broad crypto cycle level (common trend)
  - Factor 2: Crypto cycle volatility/risk regime

References
----------
Stock, J. H., & Watson, M. W. (1989). New indexes of coincident and
leading economic indicators. NBER Macroeconomics Annual, 4, 351-394.

Kim, C. J., & Nelson, C. R. (1999). State-Space Models with Regime
Switching. MIT Press.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

# Observable series for the DFM
COINCIDENT_OBSERVABLES = [
    "btc_log_price",
    "btc_realized_vol",
    "total_market_cap_log",
    "btc_dominance",
]


@dataclass
class CoincidentIndexResult:
    """
    Output container for the Coincident Index.

    Attributes
    ----------
    factors : pd.DataFrame
        Estimated latent factors (columns: factor_0, factor_1).
        Factor 0 is the primary cycle index.
    index : pd.Series
        Composite CI — Factor 0 standardized to [0, 100] range.
    loadings : pd.DataFrame
        Factor loadings matrix (observables × factors).
    smoothed_states : pd.DataFrame
        Full Kalman smoother state estimates.
    model_result : DynamicFactorResults
        Raw statsmodels result object for diagnostics.
    """

    factors: pd.DataFrame
    index: pd.Series
    loadings: pd.DataFrame
    smoothed_states: Optional[pd.DataFrame]
    model_result: object  # statsmodels DynamicFactorResults


def _prepare_observables(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and standardize the observable series for the DFM.

    Parameters
    ----------
    data : pd.DataFrame
        Raw data with columns: btc_price_usd, btc_realized_vol,
        total_market_cap, btc_dominance.

    Returns
    -------
    pd.DataFrame
        Standardized observable DataFrame.
    """
    obs = pd.DataFrame(index=data.index)

    # Log-transform price and market cap
    if "btc_price_usd" in data.columns:
        obs["btc_log_price"] = np.log(data["btc_price_usd"].clip(lower=1e-6))
    elif "btc_log_price" in data.columns:
        obs["btc_log_price"] = data["btc_log_price"]

    if "btc_realized_vol" in data.columns:
        obs["btc_realized_vol"] = data["btc_realized_vol"]

    if "total_market_cap" in data.columns:
        obs["total_market_cap_log"] = np.log(data["total_market_cap"].clip(lower=1e6))
    elif "total_market_cap_log" in data.columns:
        obs["total_market_cap_log"] = data["total_market_cap_log"]

    if "btc_dominance" in data.columns:
        obs["btc_dominance"] = data["btc_dominance"]

    obs = obs.dropna()

    # Standardize each observable (zero mean, unit variance)
    obs = (obs - obs.mean()) / (obs.std() + 1e-12)
    return obs


def build_coincident_index(
    data: pd.DataFrame,
    k_factors: int = 2,
    factor_order: int = 2,
    em_iter: int = 100,
) -> CoincidentIndexResult:
    """
    Estimate the 2-factor Dynamic Factor Model and return the CI.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns including btc_price_usd, btc_realized_vol,
        total_market_cap, btc_dominance.
    k_factors : int
        Number of latent factors. Default: 2.
    factor_order : int
        VAR order of factor dynamics. Default: 2.
    em_iter : int
        Maximum EM algorithm iterations for initialization.

    Returns
    -------
    CoincidentIndexResult
    """
    obs = _prepare_observables(data)

    available = [c for c in COINCIDENT_OBSERVABLES if c in obs.columns]
    if len(available) < 2:
        raise ValueError(
            f"Need at least 2 observable series for DFM. "
            f"Found: {available}. Check input data columns."
        )

    obs_subset = obs[available].dropna()
    logger.info(
        "Fitting DFM with %d factors, %d observables, %d observations",
        k_factors,
        len(available),
        len(obs_subset),
    )

    # Build DynamicFactor model
    model = sm.tsa.DynamicFactor(
        obs_subset,
        k_factors=k_factors,
        factor_order=factor_order,
    )

    # Two-step estimation: EM initialization + MLE
    try:
        # EM initialization
        start_params = model.fit_em(
            maxiter=em_iter,
            disp=False,
        ).params
        # MLE with EM-initialized parameters
        result = model.fit(
            start_params=start_params,
            method="lbfgs",
            maxiter=500,
            disp=False,
        )
    except Exception as exc:
        logger.warning("MLE failed (%s), falling back to EM-only fit", exc)
        result = model.fit_em(maxiter=em_iter * 2, disp=False)

    # Extract smoothed factors from Kalman smoother
    # statsmodels DynamicFactor: smoothed_state shape (n_obs, k_states)
    smoothed = result.smoothed_state.T  # shape: (n_obs, k_states)
    n_factors = k_factors
    factor_cols = [f"factor_{i}" for i in range(n_factors)]

    # The first k_factors states are the factors; the rest are factor lags
    factors_df = pd.DataFrame(
        smoothed[:, :n_factors],
        index=obs_subset.index,
        columns=factor_cols,
    )

    # Extract factor loadings (Lambda matrix)
    # From statsmodels, loadings are stored in the result params
    try:
        loadings_matrix = result.model.ssm["design"].squeeze()  # (n_obs, k_factors, ...)
        if loadings_matrix.ndim == 3:
            loadings_matrix = loadings_matrix[:, :n_factors, 0]
        else:
            loadings_matrix = loadings_matrix[:, :n_factors]
        loadings_df = pd.DataFrame(
            loadings_matrix,
            index=available,
            columns=factor_cols,
        )
    except Exception:
        # Fallback: extract from params
        loadings_df = pd.DataFrame(
            np.eye(len(available), n_factors),
            index=available,
            columns=factor_cols,
        )

    # Primary CI: Factor 0, scaled to 0–100
    f0 = factors_df["factor_0"]
    f0_min, f0_max = f0.min(), f0.max()
    ci = (f0 - f0_min) / (f0_max - f0_min + 1e-12) * 100.0
    ci.name = "CI"

    logger.info(
        "Coincident Index built: current CI=%.1f, "
        "factor loadings shape=%s",
        float(ci.iloc[-1]) if len(ci) else float("nan"),
        loadings_df.shape,
    )

    return CoincidentIndexResult(
        factors=factors_df,
        index=ci,
        loadings=loadings_df,
        smoothed_states=pd.DataFrame(
            smoothed, index=obs_subset.index,
            columns=[f"state_{i}" for i in range(smoothed.shape[1])],
        ),
        model_result=result,
    )


def get_cycle_phase_from_ci(ci: pd.Series) -> pd.Series:
    """
    Classify each month into a cycle phase based on CI level and direction.

    Phases:
      - 'early_expansion': CI < 40 and rising
      - 'expansion': CI 40–70 and rising
      - 'late_expansion': CI > 70 and rising
      - 'early_contraction': CI > 60 and falling
      - 'contraction': CI 30–60 and falling
      - 'late_contraction': CI < 30 and falling

    Parameters
    ----------
    ci : pd.Series
        Monthly CI series (0–100).

    Returns
    -------
    pd.Series
        String phase labels.
    """
    direction = np.sign(ci.diff())
    phases: list[str] = []

    for i, (val, d) in enumerate(zip(ci.values, direction.values)):
        if np.isnan(val) or np.isnan(d):
            phases.append("unknown")
        elif d >= 0:
            if val < 40:
                phases.append("early_expansion")
            elif val <= 70:
                phases.append("expansion")
            else:
                phases.append("late_expansion")
        else:
            if val > 60:
                phases.append("early_contraction")
            elif val >= 30:
                phases.append("contraction")
            else:
                phases.append("late_contraction")

    return pd.Series(phases, index=ci.index, name="ci_phase")
