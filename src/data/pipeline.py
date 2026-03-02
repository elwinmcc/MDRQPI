"""
Data pipeline orchestrator.

Fetches all required series (FRED + CoinGecko), caches them locally as
parquet, and provides a single load_all() entry point that returns a
DataFrame with all series aligned to monthly frequency.

Usage
-----
From CLI::

    python -m src.data.pipeline

From code::

    from src.data.pipeline import DataPipeline
    pipeline = DataPipeline.from_config("config/settings.yaml")
    data = pipeline.load_all()
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from src.data.cache import DataCache
from src.data.crypto_client import CoinGeckoClient
from src.data.fred_client import FredClient, LEADING_SERIES, SUPPLEMENTAL_SERIES

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Orchestrates data fetching, caching, and alignment.

    Parameters
    ----------
    fred_client : FredClient
        Configured FRED API client.
    crypto_client : CoinGeckoClient
        CoinGecko API client.
    cache : DataCache
        Local parquet cache.
    """

    def __init__(
        self,
        fred_client: FredClient,
        crypto_client: CoinGeckoClient,
        cache: DataCache,
    ) -> None:
        self.fred = fred_client
        self.crypto = crypto_client
        self.cache = cache

    @classmethod
    def from_config(cls, config_path: str | Path = "config/settings.yaml") -> "DataPipeline":
        """
        Instantiate pipeline from YAML config file.

        Parameters
        ----------
        config_path : str | Path
            Path to settings.yaml.

        Returns
        -------
        DataPipeline
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                "Copy config/settings.yaml.example to config/settings.yaml "
                "and add your FRED API key."
            )
        with config_path.open() as f:
            cfg = yaml.safe_load(f)

        fred_key = cfg["fred"]["api_key"]
        start_date = cfg["data"].get("start_date", "2010-01-01")
        cache_dir = cfg["data"].get("cache_dir", "data/cache")

        return cls(
            fred_client=FredClient(api_key=fred_key, start_date=start_date),
            crypto_client=CoinGeckoClient(start_date=start_date),
            cache=DataCache(cache_dir=cache_dir),
        )

    def _fetch_or_load(
        self,
        key: str,
        fetch_fn,
        force_refresh: bool = False,
    ) -> pd.Series:
        """
        Return cached series if available, otherwise fetch and cache.

        Parameters
        ----------
        key : str
            Cache key / series identifier.
        fetch_fn : callable
            Zero-argument callable that fetches the series.
        force_refresh : bool
            If True, skip cache and re-fetch.

        Returns
        -------
        pd.Series
        """
        if not force_refresh and self.cache.exists(key):
            return self.cache.load(key)
        series = fetch_fn()
        self.cache.save(key, series)
        return series

    def fetch_fred_series(self, force_refresh: bool = False) -> dict[str, pd.Series]:
        """
        Fetch all FRED series (with caching).

        Parameters
        ----------
        force_refresh : bool
            If True, bypass cache and re-fetch from FRED.

        Returns
        -------
        dict[str, pd.Series]
            All FRED series keyed by series ID.
        """
        all_ids = list(LEADING_SERIES.keys()) + list(SUPPLEMENTAL_SERIES.keys())
        results: dict[str, pd.Series] = {}
        for sid in all_ids:
            results[sid] = self._fetch_or_load(
                sid,
                lambda s=sid: self.fred.fetch(s),
                force_refresh=force_refresh,
            )
        return results

    def fetch_crypto_series(self, force_refresh: bool = False) -> dict[str, pd.Series]:
        """
        Fetch all crypto series (with caching).

        Parameters
        ----------
        force_refresh : bool
            If True, bypass cache and re-fetch from CoinGecko.

        Returns
        -------
        dict[str, pd.Series]
        """
        slugs = ["btc_price_usd", "btc_market_cap", "total_market_cap",
                 "btc_dominance", "btc_realized_vol"]

        # Check if all already cached
        if not force_refresh and all(self.cache.exists(s) for s in slugs):
            return {s: self.cache.load(s) for s in slugs}

        # Fetch all at once (avoids redundant API calls)
        logger.info("Fetching all crypto series from CoinGecko")
        all_series = self.crypto.fetch_all()
        for key, series in all_series.items():
            self.cache.save(key, series)
        return all_series

    def load_all(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Return a single DataFrame with all series aligned to monthly frequency.

        Parameters
        ----------
        force_refresh : bool
            If True, bypass all caches.

        Returns
        -------
        pd.DataFrame
            Monthly DataFrame. Columns are series IDs / slugs.
            Index is DatetimeIndex (month-end).
        """
        fred_data = self.fetch_fred_series(force_refresh=force_refresh)
        crypto_data = self.fetch_crypto_series(force_refresh=force_refresh)

        all_series = {**fred_data, **crypto_data}
        df = pd.DataFrame(all_series)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        logger.info(
            "Loaded %d series, %d monthly observations (%s to %s)",
            len(df.columns),
            len(df),
            df.index[0].date() if len(df) else "N/A",
            df.index[-1].date() if len(df) else "N/A",
        )
        return df


def main() -> None:
    """CLI entry point: fetch and cache all data."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/settings.yaml"
    pipeline = DataPipeline.from_config(config_path)
    df = pipeline.load_all(force_refresh="--refresh" in sys.argv)
    print(f"\nLoaded data shape: {df.shape}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"\nColumns:\n{df.columns.tolist()}")
    print(f"\nTail:\n{df.tail()}")


if __name__ == "__main__":
    main()
