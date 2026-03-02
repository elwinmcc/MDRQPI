"""
Unit tests for Bry-Boschan turning point detection.
"""

import numpy as np
import pandas as pd
import pytest

from src.spectral.turning import BryBoschan, TurningPoints, align_turning_points


def make_synthetic_cycle(
    n_months: int = 120,
    period: float = 48.0,
    amplitude: float = 1.0,
    noise_std: float = 0.05,
    seed: int = 42,
) -> pd.Series:
    """Generate a synthetic monthly cycle with known peaks and troughs."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    signal = amplitude * np.sin(2 * np.pi * t / period)
    noise = rng.normal(0, noise_std, n_months)
    dates = pd.date_range("2014-01-31", periods=n_months, freq="ME")
    return pd.Series(signal + noise, index=dates)


class TestBryBoschan:
    def test_detects_peaks_and_troughs(self):
        series = make_synthetic_cycle(n_months=120, period=24)
        bb = BryBoschan(min_phase=5, min_cycle=12, smooth_window=3)
        tp = bb.detect(series)
        # With a 24-month cycle over 120 months, expect ~5 full cycles
        assert len(tp.peaks) >= 3
        assert len(tp.troughs) >= 3

    def test_alternation_enforced(self):
        series = make_synthetic_cycle(n_months=120, period=24)
        bb = BryBoschan(min_phase=5, min_cycle=12, smooth_window=3)
        tp = bb.detect(series)

        # Merge and sort all turns
        all_turns = sorted(
            [(d, "P") for d in tp.peaks] + [(d, "T") for d in tp.troughs]
        )
        for i in range(1, len(all_turns)):
            assert all_turns[i][1] != all_turns[i - 1][1], (
                f"Consecutive same-type turn at {all_turns[i][0]}"
            )

    def test_min_phase_respected(self):
        series = make_synthetic_cycle(n_months=120, period=24)
        min_phase = 6
        bb = BryBoschan(min_phase=min_phase, min_cycle=12, smooth_window=3)
        tp = bb.detect(series)

        all_turns = sorted(
            [(d, t) for d in tp.peaks for t in ["P"]]
            + [(d, t) for d in tp.troughs for t in ["T"]]
        )
        for i in range(1, len(all_turns)):
            phase_len = (all_turns[i][0] - all_turns[i - 1][0]).days / 30.44
            assert phase_len >= min_phase - 1, (
                f"Phase too short: {phase_len:.1f} < {min_phase}"
            )

    def test_short_series_warning(self, caplog):
        import logging
        short = make_synthetic_cycle(n_months=15)
        bb = BryBoschan()
        with caplog.at_level(logging.WARNING):
            tp = bb.detect(short)
        assert any("short" in rec.message.lower() for rec in caplog.records)

    def test_phase_at(self):
        series = make_synthetic_cycle(n_months=120, period=24)
        bb = BryBoschan(min_phase=5, min_cycle=12, smooth_window=3)
        tp = bb.detect(series)
        # Just ensure it returns a valid string
        phase = tp.phase_at(series.index[60])
        assert phase in ("expansion", "contraction")

    def test_returns_turning_points_type(self):
        series = make_synthetic_cycle()
        bb = BryBoschan()
        tp = bb.detect(series)
        assert isinstance(tp, TurningPoints)
        assert isinstance(tp.peaks, pd.DatetimeIndex)
        assert isinstance(tp.troughs, pd.DatetimeIndex)


class TestAlignTurningPoints:
    def test_perfect_alignment(self):
        series = make_synthetic_cycle(n_months=120, period=24)
        bb = BryBoschan(min_phase=5, min_cycle=12, smooth_window=3)
        tp = bb.detect(series)
        # Compare against itself — should have 100% hit rate
        stats = align_turning_points(tp, tp, tolerance_months=0)
        assert stats["hit_rate"] == pytest.approx(1.0)
        assert stats["false_alarm_rate"] == pytest.approx(0.0)

    def test_zero_turns(self):
        empty_tp = TurningPoints(
            peaks=pd.DatetimeIndex([]),
            troughs=pd.DatetimeIndex([]),
            signal=pd.Series(dtype=float),
        )
        stats = align_turning_points(empty_tp, empty_tp, tolerance_months=3)
        assert stats["hit_rate"] == pytest.approx(0.0)
