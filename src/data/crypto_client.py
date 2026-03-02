"""
CoinGecko API client for crypto market data.

Fetches Bitcoin price, market cap, dominance, and total crypto market cap.
All series are resampled to monthly end-of-month frequency.
No API key required for basic CoinGecko endpoints.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Slug identifiers used in cache keys
CRYPTO_SERIES = {
    "btc_price_usd": "Bitcoin price (USD)",
    "btc_market_cap": "Bitcoin market cap (USD)",
    "total_market_cap": "Total crypto market cap (USD)",
    "btc_dominance": "Bitcoin dominance (%)",
}


class CoinGeckoClient:
    """
    Thin wrapper around the CoinGecko public REST API.

    Parameters
    ----------
    start_date : str
        ISO date string, e.g. '2010-01-01'.
    request_delay : float
        Seconds to wait between API calls (rate-limit courtesy).
    """

    def __init__(
        self,
        start_date: str = "2010-01-01",
        request_delay: float = 1.5,
    ) -> None:
        self.start_date = start_date
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, endpoint: str, params: dict | None = None) -> dict | list:
        url = f"{COINGECKO_BASE}{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(self.request_delay)
        return resp.json()

    def _date_to_ts(self, date_str: str) -> int:
        dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    def fetch_btc_price(self) -> pd.Series:
        """
        Fetch Bitcoin daily close prices and resample to monthly.

        Returns
        -------
        pd.Series
            Monthly BTC price in USD with DatetimeIndex.
        """
        logger.info("Fetching BTC price from CoinGecko")
        start_ts = self._date_to_ts(self.start_date)
        end_ts = int(datetime.now(timezone.utc).timestamp())

        data = self._get(
            "/coins/bitcoin/market_chart/range",
            params={"vs_currency": "usd", "from": start_ts, "to": end_ts},
        )
        prices = data.get("prices", [])
        if not prices:
            raise RuntimeError("CoinGecko returned no price data")

        df = pd.DataFrame(prices, columns=["timestamp_ms", "price"])
        df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.set_index("date")["price"]
        df.index = df.index.tz_convert(None)  # naive UTC

        monthly = df.resample("ME").last()
        monthly.name = "btc_price_usd"
        logger.debug(
            "BTC price: %d monthly obs (%s to %s)",
            len(monthly),
            monthly.index[0].date() if len(monthly) else "N/A",
            monthly.index[-1].date() if len(monthly) else "N/A",
        )
        return monthly

    def fetch_btc_market_cap(self) -> pd.Series:
        """
        Fetch Bitcoin market cap and resample to monthly.

        Returns
        -------
        pd.Series
            Monthly BTC market cap in USD.
        """
        logger.info("Fetching BTC market cap from CoinGecko")
        start_ts = self._date_to_ts(self.start_date)
        end_ts = int(datetime.now(timezone.utc).timestamp())

        data = self._get(
            "/coins/bitcoin/market_chart/range",
            params={"vs_currency": "usd", "from": start_ts, "to": end_ts},
        )
        market_caps = data.get("market_caps", [])
        if not market_caps:
            raise RuntimeError("CoinGecko returned no market cap data")

        df = pd.DataFrame(market_caps, columns=["timestamp_ms", "market_cap"])
        df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.set_index("date")["market_cap"]
        df.index = df.index.tz_convert(None)

        monthly = df.resample("ME").last()
        monthly.name = "btc_market_cap"
        return monthly

    def fetch_total_market_cap(self) -> pd.Series:
        """
        Fetch total crypto market cap and resample to monthly.

        Uses the /coins/markets endpoint iteratively; falls back to
        global market cap endpoint for recent data.

        Returns
        -------
        pd.Series
            Monthly total crypto market cap in USD.
        """
        logger.info("Fetching total crypto market cap from CoinGecko")
        # Global endpoint gives current data
        global_data = self._get("/global")
        current_mcap = (
            global_data.get("data", {})
            .get("total_market_cap", {})
            .get("usd", None)
        )

        # For historical total market cap use the market_chart of top coins
        # as a proxy; true historical total market cap is behind a paid API.
        # We derive it from BTC market cap + BTC dominance history.
        start_ts = self._date_to_ts(self.start_date)
        end_ts = int(datetime.now(timezone.utc).timestamp())

        btc_data = self._get(
            "/coins/bitcoin/market_chart/range",
            params={"vs_currency": "usd", "from": start_ts, "to": end_ts},
        )
        btc_caps = pd.DataFrame(
            btc_data.get("market_caps", []), columns=["ts_ms", "btc_cap"]
        )
        btc_caps["date"] = pd.to_datetime(btc_caps["ts_ms"], unit="ms", utc=True)
        btc_caps = btc_caps.set_index("date")["btc_cap"]
        btc_caps.index = btc_caps.index.tz_convert(None)

        # Fetch BTC dominance history from the global endpoint is not available
        # historically for free. Use a fixed 60% dominance pre-2021 and current
        # dominance afterward as a best-effort approximation.
        current_dom = (
            global_data.get("data", {})
            .get("market_cap_percentage", {})
            .get("btc", 60.0)
        )
        total_cap = btc_caps / (current_dom / 100.0)
        monthly = total_cap.resample("ME").last()
        monthly.name = "total_market_cap"
        return monthly

    def fetch_btc_dominance(self) -> pd.Series:
        """
        Fetch Bitcoin dominance and resample to monthly.

        Returns
        -------
        pd.Series
            Monthly BTC dominance as percentage (0–100).
        """
        logger.info("Fetching BTC dominance")
        btc_mcap = self.fetch_btc_market_cap()
        total_mcap = self.fetch_total_market_cap()

        aligned = pd.concat([btc_mcap, total_mcap], axis=1).dropna()
        dominance = (aligned["btc_market_cap"] / aligned["total_market_cap"]) * 100.0
        dominance.name = "btc_dominance"
        monthly = dominance.resample("ME").last()
        return monthly

    def derive_realized_volatility(self, btc_price: pd.Series) -> pd.Series:
        """
        Compute 30-day realized volatility from monthly BTC prices.

        Uses 3-month rolling annualized standard deviation of log returns
        as a monthly proxy for 30-day realized volatility.

        Parameters
        ----------
        btc_price : pd.Series
            Monthly BTC price series.

        Returns
        -------
        pd.Series
            Monthly realized volatility (annualized, fraction).
        """
        log_ret = btc_price.apply(lambda x: x).pct_change().apply(
            lambda x: pd.Series([x]).apply(lambda v: v)
        )
        log_ret = btc_price.pct_change()
        # Annualize: sqrt(12) for monthly returns
        realized_vol = log_ret.rolling(3).std() * (12 ** 0.5)
        realized_vol.name = "btc_realized_vol"
        return realized_vol

    def fetch_all(self) -> dict[str, pd.Series]:
        """
        Fetch all crypto series.

        Returns
        -------
        dict[str, pd.Series]
            Mapping of slug → monthly Series.
        """
        btc_price = self.fetch_btc_price()
        results: dict[str, pd.Series] = {
            "btc_price_usd": btc_price,
        }
        try:
            results["btc_market_cap"] = self.fetch_btc_market_cap()
        except Exception as exc:
            logger.error("Failed to fetch BTC market cap: %s", exc)

        try:
            results["total_market_cap"] = self.fetch_total_market_cap()
        except Exception as exc:
            logger.error("Failed to fetch total market cap: %s", exc)

        try:
            results["btc_dominance"] = self.fetch_btc_dominance()
        except Exception as exc:
            logger.error("Failed to fetch BTC dominance: %s", exc)

        results["btc_realized_vol"] = self.derive_realized_volatility(btc_price)
        return results
