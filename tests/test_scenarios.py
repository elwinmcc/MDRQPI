"""
Unit tests for scenario tree construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.projection.scenarios import (
    _compute_scenario_probabilities,
    _project_price_path,
    scenario_summary,
    ScenarioTree,
    Scenario,
)


class TestScenarioProbabilities:
    def test_probabilities_sum_to_one(self):
        for li in [20, 50, 80]:
            for lai in [20, 50, 80]:
                for phase in ["expansion", "contraction", "unknown"]:
                    probs = _compute_scenario_probabilities(li, lai, phase)
                    total = sum(probs.values())
                    assert abs(total - 1.0) < 1e-9, (
                        f"Probs don't sum to 1 for li={li}, lai={lai}, phase={phase}: {probs}"
                    )

    def test_high_li_favors_bull(self):
        probs_high = _compute_scenario_probabilities(90, 50, "expansion")
        probs_low = _compute_scenario_probabilities(10, 50, "expansion")
        assert probs_high["bull"] > probs_low["bull"]
        assert probs_high["bear"] < probs_low["bear"]

    def test_high_lai_reduces_bull(self):
        probs_low_lai = _compute_scenario_probabilities(70, 20, "expansion")
        probs_high_lai = _compute_scenario_probabilities(70, 80, "expansion")
        assert probs_low_lai["bull"] > probs_high_lai["bull"]

    def test_all_probs_positive(self):
        probs = _compute_scenario_probabilities(50, 50, "unknown")
        for k, v in probs.items():
            assert v > 0, f"Probability for {k} is non-positive: {v}"

    def test_late_expansion_reduces_bull(self):
        probs_early = _compute_scenario_probabilities(70, 50, "early_expansion")
        probs_late = _compute_scenario_probabilities(70, 50, "late_expansion")
        assert probs_early["bull"] > probs_late["bull"]


class TestProjectPricePath:
    def test_starts_at_current(self):
        dates = pd.date_range("2025-01-31", periods=1, freq="ME")
        path = _project_price_path(
            current_price=50000,
            target_price=100000,
            horizon_months=12,
            start_date=dates[0],
        )
        assert abs(path.iloc[0] - 50000) < 1, f"Path doesn't start at current: {path.iloc[0]}"

    def test_ends_near_target(self):
        dates = pd.date_range("2025-01-31", periods=1, freq="ME")
        path = _project_price_path(
            current_price=50000,
            target_price=100000,
            horizon_months=12,
            start_date=dates[0],
            noise_scale=0.0,
        )
        assert abs(path.iloc[-1] - 100000) < 100

    def test_correct_length(self):
        dates = pd.date_range("2025-01-31", periods=1, freq="ME")
        horizon = 18
        path = _project_price_path(
            current_price=50000,
            target_price=80000,
            horizon_months=horizon,
            start_date=dates[0],
        )
        assert len(path) == horizon + 1  # inclusive of start

    def test_positive_prices(self):
        dates = pd.date_range("2025-01-31", periods=1, freq="ME")
        path = _project_price_path(
            current_price=50000,
            target_price=10000,  # bear scenario
            horizon_months=12,
            start_date=dates[0],
        )
        assert (path.values > 0).all()

    def test_both_shapes(self):
        dates = pd.date_range("2025-01-31", periods=1, freq="ME")
        for shape in ["logistic", "linear"]:
            path = _project_price_path(
                current_price=50000,
                target_price=80000,
                horizon_months=12,
                start_date=dates[0],
                shape=shape,
                noise_scale=0.0,
            )
            assert len(path) == 13


class TestScenarioSummary:
    def _make_mock_tree(self) -> ScenarioTree:
        rng = np.random.default_rng(0)
        dates = pd.date_range("2025-01-31", periods=1, freq="ME")
        start = dates[0]
        scenarios = {}
        for label, prob in [("bull", 0.4), ("base", 0.4), ("bear", 0.2)]:
            target = {"bull": 150000, "base": 100000, "bear": 40000}[label]
            path = _project_price_path(75000, target, 12, start, noise_scale=0.0)
            peak = path.max() if label != "bear" else path.min()
            peak_date = path.idxmax() if label != "bear" else path.idxmin()
            scenarios[label] = Scenario(
                label=label,
                probability=prob,
                horizon_months=12,
                price_path=path,
                peak_price=peak,
                peak_date=peak_date,
                return_pct=(peak / 75000 - 1) * 100,
            )
        return ScenarioTree(
            scenarios=scenarios,
            current_price=75000,
            projection_date=start,
            dominant_cycle_months=48.0,
            estimated_lag_months=8,
        )

    def test_summary_has_three_rows(self):
        tree = self._make_mock_tree()
        df = scenario_summary(tree)
        assert len(df) == 3

    def test_summary_has_correct_index(self):
        tree = self._make_mock_tree()
        df = scenario_summary(tree)
        assert set(df.index) == {"Bull", "Base", "Bear"}
