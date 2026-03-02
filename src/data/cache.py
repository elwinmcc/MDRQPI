"""
Parquet-based local data cache.

All fetched series are stored as parquet files keyed by series ID.
Avoids repeated API calls on subsequent runs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """
    Parquet-based file cache for time series data.

    Parameters
    ----------
    cache_dir : str | Path
        Directory where parquet files are stored.
    """

    def __init__(self, cache_dir: str | Path = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.parquet"

    def exists(self, key: str) -> bool:
        """Return True if a cached file exists for *key*."""
        return self._path(key).exists()

    def load(self, key: str) -> pd.Series:
        """
        Load a cached series.

        Parameters
        ----------
        key : str
            Cache key (typically the FRED series ID or a slug).

        Returns
        -------
        pd.Series
            Time-indexed series.

        Raises
        ------
        FileNotFoundError
            If no cache entry exists for *key*.
        """
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(f"No cache entry for '{key}' at {path}")
        df = pd.read_parquet(path)
        series = df.squeeze()
        series.index = pd.to_datetime(series.index)
        logger.debug("Loaded '%s' from cache (%d rows)", key, len(series))
        return series

    def save(self, key: str, series: pd.Series) -> None:
        """
        Persist a series to cache.

        Parameters
        ----------
        key : str
            Cache key.
        series : pd.Series
            Time-indexed series to cache.
        """
        path = self._path(key)
        df = series.to_frame(name="value")
        df.to_parquet(path, index=True)
        logger.debug("Saved '%s' to cache (%d rows)", key, len(series))

    def invalidate(self, key: str) -> None:
        """Remove a cached entry."""
        path = self._path(key)
        if path.exists():
            path.unlink()
            logger.info("Invalidated cache entry '%s'", key)

    def list_keys(self) -> list[str]:
        """Return all cached keys."""
        return [p.stem for p in self.cache_dir.glob("*.parquet")]
