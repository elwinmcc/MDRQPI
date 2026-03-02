"""
FRED API client wrapper.

Fetches monetary and financial time series from the St. Louis Fed FRED API.
All series are resampled to monthly frequency (end-of-month).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd
from fredapi import Fred

logger = logging.getLogger(__name__)

# Leading Index component series
LEADING_SERIES: dict[str, str] = {
    "WM2NS": "M2 Money Supply (weekly, billions USD)",
    "WALCL": "Fed Reserve Total Assets (weekly, millions USD)",
    "WTREGEN": "Treasury General Account Balance (weekly, millions USD)",
    "RRPONTSYD": "Overnight Reverse Repo (daily, billions USD)",
    "BAMLH0A0HYM2": "ICE BofA HY OAS (daily, %)",
    "T10Y2Y": "10Y-2Y Treasury Spread (daily, %)",
    "DTWEXBGS": "Trade-Weighted USD Broad Goods & Services (daily)",
}

# Additional series used across modules
SUPPLEMENTAL_SERIES: dict[str, str] = {
    "CPIAUCSL": "CPI All Items (monthly)",
    "UNRATE": "Unemployment Rate (monthly)",
    "INDPRO": "Industrial Production Index (monthly)",
    "FEDFUNDS": "Effective Federal Funds Rate (monthly)",
}

ALL_FRED_SERIES = {**LEADING_SERIES, **SUPPLEMENTAL_SERIES}


class FredClient:
    """
    Wrapper around fredapi.Fred that returns monthly pandas Series.

    Parameters
    ----------
    api_key : str
        FRED API key.
    start_date : str
        ISO date string, e.g. '2010-01-01'.
    """

    def __init__(self, api_key: str, start_date: str = "2010-01-01") -> None:
        self.fred = Fred(api_key=api_key)
        self.start_date = start_date

    def fetch(self, series_id: str) -> pd.Series:
        """
        Fetch a single FRED series and resample to monthly.

        Parameters
        ----------
        series_id : str
            FRED series identifier (e.g. 'WM2NS').

        Returns
        -------
        pd.Series
            Monthly frequency series with DatetimeIndex (period end).
        """
        logger.info("Fetching FRED series: %s", series_id)
        raw: pd.Series = self.fred.get_series(
            series_id,
            observation_start=self.start_date,
        )
        raw.index = pd.to_datetime(raw.index)
        raw = raw.dropna()

        # Resample to monthly end, taking last observation in each month
        monthly = raw.resample("ME").last()
        monthly.name = series_id
        logger.debug(
            "Fetched %s: %d monthly observations (%s to %s)",
            series_id,
            len(monthly),
            monthly.index[0].date() if len(monthly) else "N/A",
            monthly.index[-1].date() if len(monthly) else "N/A",
        )
        return monthly

    def fetch_all_leading(self) -> dict[str, pd.Series]:
        """
        Fetch all 7 Leading Index component series.

        Returns
        -------
        dict[str, pd.Series]
            Mapping of series ID → monthly Series.
        """
        results: dict[str, pd.Series] = {}
        for sid in LEADING_SERIES:
            try:
                results[sid] = self.fetch(sid)
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", sid, exc)
        return results

    def fetch_supplemental(self) -> dict[str, pd.Series]:
        """
        Fetch supplemental macro series.

        Returns
        -------
        dict[str, pd.Series]
            Mapping of series ID → monthly Series.
        """
        results: dict[str, pd.Series] = {}
        for sid in SUPPLEMENTAL_SERIES:
            try:
                results[sid] = self.fetch(sid)
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", sid, exc)
        return results
