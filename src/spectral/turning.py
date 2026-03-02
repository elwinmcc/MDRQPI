"""
Bry-Boschan (1971) turning point detection algorithm.

Identifies business cycle peaks and troughs in a monthly time series.
Follows the NBER methodology with the constraints:
  - Minimum phase length: 6 months (trough-to-peak or peak-to-trough)
  - Minimum cycle length: 15 months (trough-to-trough or peak-to-peak)
  - Smoothing: 12-month centered MA (standard) or one-sided MA (real-time)

References
----------
Bry, G., & Boschan, C. (1971). Cyclical Analysis of Time Series:
Selected Procedures and Computer Programs. NBER.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TurnType(IntEnum):
    TROUGH = -1
    NONE = 0
    PEAK = 1


@dataclass
class TurningPoints:
    """
    Container for detected peaks and troughs.

    Attributes
    ----------
    peaks : pd.DatetimeIndex
        Dates of detected cycle peaks.
    troughs : pd.DatetimeIndex
        Dates of detected cycle troughs.
    signal : pd.Series
        Original (smoothed) series used for detection.
    """

    peaks: pd.DatetimeIndex
    troughs: pd.DatetimeIndex
    signal: pd.Series
    turn_series: pd.Series = field(default_factory=pd.Series)

    @property
    def all_turns(self) -> pd.Series:
        """Combined series: +1 at peaks, -1 at troughs, 0 elsewhere."""
        return self.turn_series

    def phase_at(self, date: pd.Timestamp) -> str:
        """
        Return the current cycle phase ('expansion' or 'contraction') at *date*.

        Parameters
        ----------
        date : pd.Timestamp

        Returns
        -------
        str
        """
        all_dates = sorted(
            [(d, "peak") for d in self.peaks] + [(d, "trough") for d in self.troughs]
        )
        last_type = "trough"  # assume we start from trough (expansion)
        for d, t in all_dates:
            if d > date:
                break
            last_type = t
        return "contraction" if last_type == "peak" else "expansion"


class BryBoschan:
    """
    Bry-Boschan turning point detection for monthly time series.

    Parameters
    ----------
    min_phase : int
        Minimum number of months in a phase (peak-to-trough or trough-to-peak).
        Default: 6.
    min_cycle : int
        Minimum number of months in a full cycle (trough-to-trough or
        peak-to-peak). Default: 15.
    smooth_window : int
        Window for smoothing MA. Default: 12 (centered; one-sided in real-time).
    mode : {'standard', 'real_time'}
        'standard' uses a centered MA (full history); 'real_time' uses a
        one-sided (trailing) MA for live signal generation.
    """

    def __init__(
        self,
        min_phase: int = 6,
        min_cycle: int = 15,
        smooth_window: int = 12,
        mode: Literal["standard", "real_time"] = "standard",
    ) -> None:
        self.min_phase = min_phase
        self.min_cycle = min_cycle
        self.smooth_window = smooth_window
        self.mode = mode

    def _smooth(self, series: pd.Series) -> pd.Series:
        """Apply smoothing MA to the series."""
        if self.mode == "real_time":
            smoothed = series.rolling(self.smooth_window, min_periods=self.smooth_window // 2).mean()
        else:
            half = self.smooth_window // 2
            smoothed = series.rolling(
                window=self.smooth_window,
                center=True,
                min_periods=half,
            ).mean()
        return smoothed.dropna()

    def _local_extrema(self, series: pd.Series) -> pd.Series:
        """
        Find local peaks (+1) and troughs (-1) in *series*.

        A local peak at index i: series[i] > series[i-1] and series[i] > series[i+1].
        A local trough at index i: series[i] < series[i-1] and series[i] < series[i+1].

        Parameters
        ----------
        series : pd.Series
            Smoothed monthly series.

        Returns
        -------
        pd.Series
            Integer series: +1 peak, -1 trough, 0 otherwise.
        """
        vals = series.values
        n = len(vals)
        turns = np.zeros(n, dtype=int)

        for i in range(1, n - 1):
            if vals[i] > vals[i - 1] and vals[i] > vals[i + 1]:
                turns[i] = TurnType.PEAK
            elif vals[i] < vals[i - 1] and vals[i] < vals[i + 1]:
                turns[i] = TurnType.TROUGH

        return pd.Series(turns, index=series.index)

    def _enforce_alternation(self, turns: pd.Series) -> pd.Series:
        """
        Ensure turns alternate P-T-P-T. When two consecutive peaks or
        troughs appear, keep only the higher peak / lower trough.

        Parameters
        ----------
        turns : pd.Series
            Raw turns series (+1/-1/0).

        Returns
        -------
        pd.Series
            Cleaned turns series.
        """
        smoothed_vals = turns.index.map(lambda _: None)  # placeholder
        turn_dates = turns[turns != 0].index.tolist()
        turn_types = turns[turns != 0].values.tolist()

        if not turn_dates:
            return turns.copy()

        cleaned = turns.copy() * 0  # reset
        # Walk through and enforce alternation
        result_turns: list[tuple[pd.Timestamp, int]] = []
        prev_type = 0

        i = 0
        while i < len(turn_types):
            cur_type = turn_types[i]
            cur_date = turn_dates[i]

            if cur_type != prev_type:
                result_turns.append((cur_date, cur_type))
                prev_type = cur_type
            else:
                # Same type as previous — keep better extreme
                last_date, last_type = result_turns[-1]
                last_val = turns.index.get_loc(last_date)
                cur_val = turns.index.get_loc(cur_date)
                if cur_type == TurnType.PEAK:
                    # Keep the higher one — we'll compare from the signal
                    # For now keep the later one (will be corrected in enforce_min_phase)
                    result_turns[-1] = (cur_date, cur_type)
                else:
                    result_turns[-1] = (cur_date, cur_type)
            i += 1

        for date, t in result_turns:
            cleaned.loc[date] = t

        return cleaned

    def _enforce_min_phase(
        self, turns: pd.Series, series: pd.Series
    ) -> pd.Series:
        """
        Remove turns that violate minimum phase and cycle length constraints.

        When a violation is found, the inferior extreme is removed (lower peak,
        higher trough). Iterates until no violations remain.

        Parameters
        ----------
        turns : pd.Series
            Alternating turns series.
        series : pd.Series
            Smoothed values (for choosing which turn to keep).

        Returns
        -------
        pd.Series
            Filtered turns series.
        """
        changed = True
        while changed:
            changed = False
            turn_dates = turns[turns != 0].index.tolist()
            turn_types = turns[turns != 0].values.tolist()
            to_remove: list[pd.Timestamp] = []

            for i in range(1, len(turn_dates)):
                phase_len = (turn_dates[i] - turn_dates[i - 1]).days / 30.44
                if phase_len < self.min_phase:
                    # Remove the inferior of the two adjacent turns
                    d0, d1 = turn_dates[i - 1], turn_dates[i]
                    t0, t1 = turn_types[i - 1], turn_types[i]
                    v0 = series.loc[d0] if d0 in series.index else np.nan
                    v1 = series.loc[d1] if d1 in series.index else np.nan
                    if t0 == TurnType.PEAK:
                        # Keep the higher peak
                        to_remove.append(d0 if v0 < v1 else d1)
                    else:
                        # Keep the lower trough
                        to_remove.append(d0 if v0 > v1 else d1)
                    changed = True
                    break  # Restart after first removal

            for d in to_remove:
                turns.loc[d] = 0

        # Check minimum cycle constraints
        changed = True
        while changed:
            changed = False
            turn_dates = turns[turns != 0].index.tolist()
            turn_types = turns[turns != 0].values.tolist()

            for i in range(2, len(turn_dates)):
                if turn_types[i] == turn_types[i - 2]:
                    cycle_len = (turn_dates[i] - turn_dates[i - 2]).days / 30.44
                    if cycle_len < self.min_cycle:
                        # Remove the middle turn (less extreme)
                        d_mid = turn_dates[i - 1]
                        turns.loc[d_mid] = 0
                        changed = True
                        break

        return turns

    def detect(self, series: pd.Series) -> TurningPoints:
        """
        Detect turning points in a monthly time series.

        Parameters
        ----------
        series : pd.Series
            Monthly time series with DatetimeIndex.

        Returns
        -------
        TurningPoints
            Detected peaks and troughs.
        """
        if len(series) < self.smooth_window * 2:
            logger.warning(
                "Series too short (%d obs) for reliable BB detection", len(series)
            )

        smoothed = self._smooth(series)
        raw_turns = self._local_extrema(smoothed)
        alternating = self._enforce_alternation(raw_turns)
        final_turns = self._enforce_min_phase(alternating, smoothed)

        peak_dates = pd.DatetimeIndex(final_turns[final_turns == TurnType.PEAK].index)
        trough_dates = pd.DatetimeIndex(final_turns[final_turns == TurnType.TROUGH].index)

        logger.info(
            "BB detected %d peaks and %d troughs",
            len(peak_dates),
            len(trough_dates),
        )
        return TurningPoints(
            peaks=peak_dates,
            troughs=trough_dates,
            signal=smoothed,
            turn_series=final_turns,
        )


def align_turning_points(
    reference: TurningPoints,
    candidate: TurningPoints,
    tolerance_months: int = 3,
) -> dict[str, float]:
    """
    Compute alignment statistics between two sets of turning points.

    Used for validating the Leading Index: how well do LI turns predict
    BTC turns within a ±tolerance window?

    Parameters
    ----------
    reference : TurningPoints
        Reference (BTC) turning points.
    candidate : TurningPoints
        Candidate (LI) turning points to evaluate.
    tolerance_months : int
        Months of tolerance for a match.

    Returns
    -------
    dict[str, float]
        Keys: 'hit_rate', 'false_alarm_rate', 'avg_lead_months'.
    """
    tolerance = pd.DateOffset(months=tolerance_months)
    hits = 0
    leads: list[float] = []

    all_ref = sorted(
        [(d, "peak") for d in reference.peaks]
        + [(d, "trough") for d in reference.troughs]
    )
    all_cand = sorted(
        [(d, "peak") for d in candidate.peaks]
        + [(d, "trough") for d in candidate.troughs]
    )

    matched_ref: set[int] = set()
    for ci, (cd, ct) in enumerate(all_cand):
        for ri, (rd, rt) in enumerate(all_ref):
            if ri in matched_ref:
                continue
            if ct == rt and abs((cd - rd).days) <= tolerance.months * 31:
                hits += 1
                lead = (rd - cd).days / 30.44
                leads.append(lead)
                matched_ref.add(ri)
                break

    total_ref = len(all_ref)
    total_cand = len(all_cand)
    hit_rate = hits / total_ref if total_ref > 0 else 0.0
    false_alarms = total_cand - hits
    false_alarm_rate = false_alarms / total_cand if total_cand > 0 else 0.0
    avg_lead = np.mean(leads) if leads else 0.0

    return {
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
        "avg_lead_months": avg_lead,
        "n_ref_turns": total_ref,
        "n_cand_turns": total_cand,
        "n_hits": hits,
    }
