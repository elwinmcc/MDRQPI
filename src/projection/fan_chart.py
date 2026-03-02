"""
Fan chart visualization for scenario projections.

Generates a Plotly figure showing:
  - Historical BTC price
  - Three scenario paths (bull/base/bear)
  - Shaded probability fan (area between bull and bear)
  - Annotated turning points from Bry-Boschan
  - LI level overlay on secondary axis

Design: consistent with the Bloomberg/BoE fan chart style.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.projection.scenarios import ScenarioTree
from src.spectral.turning import TurningPoints

logger = logging.getLogger(__name__)

SCENARIO_COLORS = {
    "bull": "#26a69a",   # teal
    "base": "#42a5f5",   # blue
    "bear": "#ef5350",   # red
}

PHASE_COLORS = {
    "expansion": "rgba(38, 166, 154, 0.08)",
    "contraction": "rgba(239, 83, 80, 0.08)",
}


def build_fan_chart(
    btc_price: pd.Series,
    scenario_tree: ScenarioTree,
    li_index: Optional[pd.Series] = None,
    ci_index: Optional[pd.Series] = None,
    lai_index: Optional[pd.Series] = None,
    turning_points: Optional[TurningPoints] = None,
    title: str = "MDR v5 — Bitcoin Cyclical Projection",
    log_scale: bool = True,
) -> go.Figure:
    """
    Build the fan chart figure.

    Parameters
    ----------
    btc_price : pd.Series
        Monthly BTC price history.
    scenario_tree : ScenarioTree
        Output from build_scenario_tree.
    li_index : pd.Series, optional
        Leading Index (0–100) for overlay.
    ci_index : pd.Series, optional
        Coincident Index (0–100) for overlay.
    lai_index : pd.Series, optional
        Lagging Index (0–100) for overlay.
    turning_points : TurningPoints, optional
        BB turning points to annotate.
    title : str
        Chart title.
    log_scale : bool
        Use log scale for price axis.

    Returns
    -------
    go.Figure
        Plotly figure with subplots.
    """
    # Determine subplot layout
    n_index_rows = sum([li_index is not None, ci_index is not None, lai_index is not None])
    n_rows = 1 + (1 if n_index_rows > 0 else 0)

    row_heights = [0.65, 0.35] if n_rows == 2 else [1.0]
    subplot_titles = [title] + (["Indices (LI / CI / LaI)"] if n_rows == 2 else [])

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
    )

    # ── Historical BTC price ──────────────────────────────────────────────────
    btc_hist = btc_price.dropna()
    # Limit history to 5 years for readability
    cutoff = scenario_tree.projection_date - pd.DateOffset(years=5)
    btc_hist_plot = btc_hist[btc_hist.index >= cutoff]

    fig.add_trace(
        go.Scatter(
            x=btc_hist_plot.index,
            y=btc_hist_plot.values,
            mode="lines",
            name="BTC Price (historical)",
            line=dict(color="#ffffff", width=2),
            showlegend=True,
        ),
        row=1, col=1,
    )

    # ── Turning point markers ─────────────────────────────────────────────────
    if turning_points is not None:
        for peak_date in turning_points.peaks:
            if peak_date < cutoff:
                continue
            price_at = btc_hist.get(peak_date, None)
            if price_at is None:
                # Find nearest
                nearest = btc_hist.index.get_indexer([peak_date], method="nearest")[0]
                price_at = float(btc_hist.iloc[nearest])
            fig.add_trace(
                go.Scatter(
                    x=[peak_date],
                    y=[float(price_at)],
                    mode="markers",
                    name="Peak",
                    marker=dict(color="#ef5350", size=10, symbol="triangle-down"),
                    showlegend=False,
                ),
                row=1, col=1,
            )

        for trough_date in turning_points.troughs:
            if trough_date < cutoff:
                continue
            price_at = btc_hist.get(trough_date, None)
            if price_at is None:
                nearest = btc_hist.index.get_indexer([trough_date], method="nearest")[0]
                price_at = float(btc_hist.iloc[nearest])
            fig.add_trace(
                go.Scatter(
                    x=[trough_date],
                    y=[float(price_at)],
                    mode="markers",
                    name="Trough",
                    marker=dict(color="#26a69a", size=10, symbol="triangle-up"),
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # ── Scenario paths ────────────────────────────────────────────────────────
    proj_date = scenario_tree.projection_date

    # Fan fill: area between bull and bear
    bull_path = scenario_tree.scenarios["bull"].price_path
    bear_path = scenario_tree.scenarios["bear"].price_path

    # Align to common index (bear may be shorter)
    common_idx = bull_path.index.intersection(bear_path.index)
    if len(common_idx) > 0:
        bull_common = bull_path.loc[common_idx]
        bear_common = bear_path.loc[common_idx]
        fig.add_trace(
            go.Scatter(
                x=list(common_idx) + list(reversed(common_idx)),
                y=list(bull_common.values) + list(reversed(bear_common.values)),
                fill="toself",
                fillcolor="rgba(66, 165, 245, 0.15)",
                line=dict(width=0),
                name="Scenario fan",
                showlegend=True,
            ),
            row=1, col=1,
        )

    for label, sc in scenario_tree.scenarios.items():
        prob_label = f"{sc.label.capitalize()} ({sc.probability:.0%})"
        fig.add_trace(
            go.Scatter(
                x=sc.price_path.index,
                y=sc.price_path.values,
                mode="lines",
                name=prob_label,
                line=dict(
                    color=SCENARIO_COLORS[label],
                    width=2,
                    dash="dash" if label == "bear" else "solid",
                ),
            ),
            row=1, col=1,
        )
        # Peak/trough annotation
        fig.add_annotation(
            x=sc.peak_date,
            y=sc.peak_price,
            text=f"${sc.peak_price:,.0f}<br>({sc.return_pct:+.0f}%)",
            showarrow=True,
            arrowhead=2,
            arrowcolor=SCENARIO_COLORS[label],
            font=dict(color=SCENARIO_COLORS[label], size=10),
            row=1, col=1,
        )

    # Vertical line at projection date
    fig.add_vline(
        x=proj_date,
        line_dash="dot",
        line_color="rgba(255,255,255,0.4)",
        annotation_text=f"Now: ${scenario_tree.current_price:,.0f}",
        annotation_position="top right",
    )

    # ── Index panel ───────────────────────────────────────────────────────────
    if n_rows == 2:
        colors = {"LI": "#ffa726", "CI": "#42a5f5", "LaI": "#ab47bc"}
        for idx_series, label in [
            (li_index, "LI"),
            (ci_index, "CI"),
            (lai_index, "LaI"),
        ]:
            if idx_series is None:
                continue
            idx_plot = idx_series[idx_series.index >= cutoff].dropna()
            fig.add_trace(
                go.Scatter(
                    x=idx_plot.index,
                    y=idx_plot.values,
                    mode="lines",
                    name=label,
                    line=dict(color=colors[label], width=2),
                ),
                row=2, col=1,
            )

        # Reference line at 50
        fig.add_hline(
            y=50,
            line_dash="dot",
            line_color="rgba(255,255,255,0.3)",
            row=2, col=1,
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    yaxis_type = "log" if log_scale else "linear"
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="JetBrains Mono, Courier New, monospace", color="#e0e0e0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        height=700 if n_rows == 2 else 500,
    )

    fig.update_yaxes(
        title_text="BTC Price (USD)",
        type=yaxis_type,
        gridcolor="rgba(255,255,255,0.05)",
        row=1, col=1,
    )

    if n_rows == 2:
        fig.update_yaxes(
            title_text="Index (0–100)",
            range=[0, 100],
            gridcolor="rgba(255,255,255,0.05)",
            row=2, col=1,
        )

    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.05)",
        showgrid=True,
    )

    return fig


def build_waterfall_chart(
    scenario_tree: ScenarioTree,
    current_price: float,
) -> go.Figure:
    """
    Build a probability-weighted waterfall chart of scenario returns.

    Parameters
    ----------
    scenario_tree : ScenarioTree
    current_price : float
        Current BTC price.

    Returns
    -------
    go.Figure
    """
    scenarios = scenario_tree.scenarios
    expected_return = sum(
        sc.probability * sc.return_pct for sc in scenarios.values()
    )

    fig = go.Figure()

    labels = []
    values = []
    colors = []
    for label, sc in scenarios.items():
        labels.append(f"{sc.label.capitalize()}<br>({sc.probability:.0%})")
        values.append(sc.return_pct * sc.probability)
        colors.append(SCENARIO_COLORS[label])

    labels.append("Expected")
    values.append(expected_return)
    colors.append("#ffa726")

    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in values],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Probability-Weighted Scenario Returns",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        yaxis_title="Probability-Weighted Return (%)",
        showlegend=False,
        height=350,
    )

    return fig
