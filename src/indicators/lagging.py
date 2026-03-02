"""
Lagging Index (LaI) — 5-component equal-weight diffusion index.

The LaI confirms cycle turns after the fact and measures cycle maturity.
A high LaI (>60) signals a mature bull market — distribution is underway.
A low LaI (<40) signals a capitulated bear — accumulation phase.
LaI crossing 50 from above = late-cycle confirmation of peak.

Construction: same diffusion methodology as LI.
Each component: 1 if value > 6-month trailing MA, else 0.
Index = mean(binary signals) × 100.

Components
----------
1. BTC Exchange Inflows (proxy from realized vol, 30d MA)
   → Lagging: high inflows signal distribution
2. Stablecoin Dominance (% of total crypto market cap)
   → Inverted: rising stablecoin share = defensive, bearish
3. BTC MVRV Ratio proxy (price / realized price approximation)
   → High MVRV = paper profits → distribution incentive
4. Funding Rate proxy (from price momentum, as a lagging sentiment gauge)
   → High/persistent positive funding = late-cycle leverage excess
5. Long-term Holder Supply proxy (inverse of recent price volatility × trend)
   → LTH supply rising = accumulation (bullish lagging signal)

Note: True on-chain data (MVRV, exchange inflows, LTH supply, funding rates)
requires paid data providers (Glassnode, etc.). These components use
data-derived proxies from available free sources. When real on-chain data
is available, replace the proxy columns with the actual series.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MA_WINDOW = 6

# Component spec: (column_name, invert, label, description)
LAGGING_COMPONENT_SPEC: list[tuple[str, bool, str, str]] = [
    (
        "exchange_inflow_proxy",
        False,
        "Exchange Inflows",
        "BTC exchange inflow proxy (realized vol normalized). High = distribution.",
    ),
    (
        "stablecoin_dominance",
        True,
        "Stablecoin Dominance (inv)",
        "Stablecoin % of market cap (inverted). Rising = defensive, bearish.",
    ),
    (
        "mvrv_proxy",
        False,
        "MVRV Proxy",
        "Price / 12-month avg price proxy for MVRV. High = overvalued.",
    ),
    (
        "funding_proxy",
        False,
        "Funding Rate Proxy",
        "Momentum persistence proxy for funding rate excess.",
    ),
    (
        "lth_supply_proxy",
        False,
        "LTH Supply Proxy",
        "Long-term holder proxy (low vol persistence = HODLing). Rising = bullish.",
    ),
]


@dataclass
class LaggingIndexResult:
    """
    Output container for the Lagging Index.

    Attributes
    ----------
    index : pd.Series
        Monthly LaI values (0–100).
    components : pd.DataFrame
        Binary signals for each of the 5 components.
    raw_components : pd.DataFrame
        Raw component values before binarization.
    maturity_score : pd.Series
        Smoothed LaI level (0–100). >70 = late-cycle warning.
    """

    index: pd.Series
    components: pd.DataFrame
    raw_components: pd.DataFrame
    maturity_score: pd.Series


def _derive_lagging_components(data: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the 5 lagging indicator components from available data.

    Falls back to proxy computations when on-chain data is unavailable.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain: btc_price_usd, btc_realized_vol.
        Optional: btc_dominance, total_market_cap.

    Returns
    -------
    pd.DataFrame
        DataFrame with derived component columns.
    """
    derived = pd.DataFrame(index=data.index)

    btc_price = data.get("btc_price_usd", None)
    realized_vol = data.get("btc_realized_vol", None)

    # 1. Exchange Inflow Proxy
    # High realized vol × rising price = likely distribution
    if btc_price is not None and realized_vol is not None:
        aligned = pd.concat([btc_price, realized_vol], axis=1).dropna()
        price = aligned.iloc[:, 0]
        vol = aligned.iloc[:, 1]
        # Inflow proxy: vol × sign(price momentum)
        price_mom = price.pct_change(3)
        inflow_proxy = vol * (price_mom > 0).astype(float)
        derived["exchange_inflow_proxy"] = inflow_proxy
    elif realized_vol is not None:
        derived["exchange_inflow_proxy"] = realized_vol

    # 2. Stablecoin Dominance Proxy
    # If btc_dominance available: stablecoin ~ (1 - btc_dominance/100) × 15%
    # As a rough approximation (true stablecoin dominance requires broader data)
    if "btc_dominance" in data.columns:
        # When BTC dominance falls, alt/stablecoins gain share
        # High altcoin season often precedes stablecoin defensive move at top
        btc_dom = data["btc_dominance"].dropna()
        stablecoin_proxy = 100.0 - btc_dom
        derived["stablecoin_dominance"] = stablecoin_proxy
    elif btc_price is not None:
        # Rough proxy: when price momentum slows, stable share rises
        price_mom = btc_price.pct_change(6).rolling(3).mean()
        stablecoin_proxy = -price_mom * 100  # inverted momentum
        derived["stablecoin_dominance"] = stablecoin_proxy

    # 3. MVRV Proxy: current price / 12-month rolling average price
    if btc_price is not None:
        btc_p = btc_price.dropna()
        realized_price_proxy = btc_p.rolling(12, min_periods=6).mean()
        mvrv_proxy = btc_p / (realized_price_proxy + 1e-6)
        derived["mvrv_proxy"] = mvrv_proxy

    # 4. Funding Rate Proxy: 3-month return persistence
    # High persistent positive returns → leveraged longs accumulate
    if btc_price is not None:
        btc_p = btc_price.dropna()
        ret_1m = btc_p.pct_change(1)
        ret_3m = btc_p.pct_change(3)
        # Funding proxy: recent return × 3m return (both positive = late cycle)
        funding_proxy = (ret_1m * ret_3m).rolling(2).mean()
        derived["funding_proxy"] = funding_proxy

    # 5. LTH Supply Proxy: inverse of realized vol (HODLers = low vol environment)
    # When vol is low and trend is up → LTHs are accumulating
    if realized_vol is not None and btc_price is not None:
        vol = realized_vol.dropna()
        # Normalize vol to 0–1 range (rolling)
        vol_norm = (vol - vol.rolling(24, min_periods=12).min()) / (
            vol.rolling(24, min_periods=12).max() - vol.rolling(24, min_periods=12).min() + 1e-12
        )
        # LTH proxy: low normalized vol = more HODLing (bullish signal)
        lth_proxy = 1.0 - vol_norm
        derived["lth_supply_proxy"] = lth_proxy

    return derived


def build_lagging_index(
    data: pd.DataFrame,
    ma_window: int = MA_WINDOW,
    component_spec: list[tuple[str, bool, str, str]] | None = None,
) -> LaggingIndexResult:
    """
    Construct the 5-component equal-weight Lagging Index.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with required series. If on-chain columns are not
        present, proxy columns are derived automatically.
    ma_window : int
        Window for binarization MA.
    component_spec : list, optional
        Override component spec.

    Returns
    -------
    LaggingIndexResult
    """
    spec = component_spec or LAGGING_COMPONENT_SPEC

    # Check which components are directly available; derive the rest
    needed = [col for col, _, _, _ in spec]
    missing_cols = [c for c in needed if c not in data.columns]

    if missing_cols:
        logger.info("Deriving proxy components for: %s", missing_cols)
        derived = _derive_lagging_components(data)
        full_data = pd.concat([data, derived], axis=1)
    else:
        full_data = data

    binary_signals: dict[str, pd.Series] = {}
    raw_components: dict[str, pd.Series] = {}

    for col, invert, label, _ in spec:
        if col not in full_data.columns:
            logger.warning("LaI component '%s' (%s) not available, skipping", col, label)
            continue
        raw = full_data[col].dropna()
        raw_components[label] = raw

        if invert:
            s = -raw
        else:
            s = raw
        ma = s.rolling(ma_window, min_periods=ma_window // 2).mean()
        binary = (s > ma).astype(int)
        binary_signals[label] = binary

    if not binary_signals:
        raise ValueError("No LaI components could be built from available data.")

    components_df = pd.DataFrame(binary_signals)
    n_avail = components_df.notna().sum(axis=1)
    min_valid = max(1, len(binary_signals) - 2)

    lai = components_df.mean(axis=1, skipna=True) * 100.0
    lai = lai.where(n_avail >= min_valid)
    lai.name = "LaI"
    lai = lai.dropna()

    # Smoothed maturity score (3-month MA)
    maturity = lai.rolling(3, min_periods=1).mean()
    maturity.name = "maturity_score"

    logger.info(
        "Built Lagging Index: %d obs, current LaI=%.1f, maturity=%.1f",
        len(lai),
        float(lai.iloc[-1]) if len(lai) else float("nan"),
        float(maturity.iloc[-1]) if len(maturity) else float("nan"),
    )

    return LaggingIndexResult(
        index=lai,
        components=components_df,
        raw_components=pd.DataFrame(raw_components),
        maturity_score=maturity,
    )


def interpret_lagging_index(lai: pd.Series, ci_phase: str | None = None) -> str:
    """
    Provide a qualitative interpretation of the current LaI level.

    Parameters
    ----------
    lai : pd.Series
        Monthly LaI series.
    ci_phase : str, optional
        Current CI phase label (from coincident.get_cycle_phase_from_ci).

    Returns
    -------
    str
        Interpretation string.
    """
    if len(lai) == 0:
        return "Insufficient data"

    current = float(lai.iloc[-1])
    trend = float(lai.diff(3).iloc[-1]) if len(lai) > 3 else 0.0

    if current > 75:
        level_msg = "LATE CYCLE: LaI extremely high — distribution likely underway"
    elif current > 60:
        level_msg = "MATURING: LaI elevated — cycle maturity increasing"
    elif current > 40:
        level_msg = "MID CYCLE: LaI neutral zone"
    elif current > 25:
        level_msg = "EARLY RECOVERY: LaI depressed — capitulation phase"
    else:
        level_msg = "DEEP TROUGH: LaI very low — maximum bearish confirmation"

    trend_msg = "trending up" if trend > 5 else ("trending down" if trend < -5 else "flat")
    return f"{level_msg} | {trend_msg} ({current:.1f})"
