"""
MDR v5 — Dash Web Application.

Single-page dashboard with:
  - Fan chart (price projection + scenario tree)
  - LI / CI / LaI index panel
  - Wavelet scalogram (power spectrum)
  - CCF bar chart (lag structure)
  - Turning point timeline
  - Scenario summary table
  - LaI maturity gauge

Usage::

    python -m src.dashboard.app
    # or
    python -m src.dashboard.app --config config/settings.yaml --port 8050

Dash runs in debug mode locally. In production, use gunicorn:
    gunicorn -w 1 src.dashboard.app:server
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html, no_update
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── App initialization ────────────────────────────────────────────────────────

app = Dash(
    __name__,
    title="MDR v5 — Bitcoin Cyclical Projection",
    suppress_callback_exceptions=True,
)
server = app.server  # Expose Flask server for gunicorn

DARK_STYLE = {
    "backgroundColor": "#1a1a2e",
    "color": "#e0e0e0",
    "fontFamily": "JetBrains Mono, Courier New, monospace",
}

CARD_STYLE = {
    **DARK_STYLE,
    "border": "1px solid rgba(255,255,255,0.1)",
    "borderRadius": "8px",
    "padding": "16px",
    "margin": "8px",
}


# ── Layout ─────────────────────────────────────────────────────────────────────

def make_header() -> html.Div:
    return html.Div(
        [
            html.H1(
                "MDR v5 — Bitcoin Cyclical Projection Model",
                style={"margin": 0, "fontSize": "1.4rem", "color": "#42a5f5"},
            ),
            html.P(
                "Monetary Dilution → Liquidity Cycle → Bitcoin Price Projection",
                style={"margin": 0, "color": "#9e9e9e", "fontSize": "0.85rem"},
            ),
        ],
        style={**DARK_STYLE, "padding": "16px 24px", "borderBottom": "1px solid rgba(255,255,255,0.1)"},
    )


def make_status_bar() -> html.Div:
    return html.Div(
        [
            html.Div(id="status-li", style={"flex": 1, **CARD_STYLE}),
            html.Div(id="status-ci", style={"flex": 1, **CARD_STYLE}),
            html.Div(id="status-lai", style={"flex": 1, **CARD_STYLE}),
            html.Div(id="status-btc", style={"flex": 1, **CARD_STYLE}),
        ],
        style={"display": "flex", "flexWrap": "wrap"},
    )


def make_controls() -> html.Div:
    return html.Div(
        [
            html.Label("Log Scale:", style={"color": "#9e9e9e", "marginRight": "8px"}),
            dcc.Checklist(
                id="log-scale-toggle",
                options=[{"label": " Price axis", "value": "log"}],
                value=["log"],
                style={"color": "#e0e0e0", "display": "inline-block"},
            ),
            html.Label("History:", style={"color": "#9e9e9e", "marginLeft": "24px", "marginRight": "8px"}),
            dcc.Slider(
                id="history-slider",
                min=1,
                max=10,
                step=1,
                value=5,
                marks={i: f"{i}y" for i in range(1, 11)},
                tooltip={"placement": "bottom"},
                className="dark-slider",
            ),
            html.Label("Refresh:", style={"color": "#9e9e9e", "marginLeft": "24px", "marginRight": "8px"}),
            html.Button(
                "↻ Refresh Data",
                id="refresh-btn",
                n_clicks=0,
                style={
                    "background": "rgba(66, 165, 245, 0.2)",
                    "border": "1px solid #42a5f5",
                    "color": "#42a5f5",
                    "borderRadius": "4px",
                    "padding": "4px 12px",
                    "cursor": "pointer",
                },
            ),
            html.Div(id="last-updated", style={"color": "#616161", "fontSize": "0.75rem", "marginLeft": "16px"}),
        ],
        style={**DARK_STYLE, "padding": "12px 24px", "display": "flex", "alignItems": "center",
               "borderBottom": "1px solid rgba(255,255,255,0.08)"},
    )


app.layout = html.Div(
    [
        make_header(),
        make_controls(),
        make_status_bar(),

        # Main chart row
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="fan-chart", style={"height": "600px"}),
                    style={**CARD_STYLE, "flex": "3"},
                ),
                html.Div(
                    [
                        html.H3("Scenario Summary", style={"color": "#42a5f5", "marginTop": 0, "fontSize": "0.95rem"}),
                        html.Div(id="scenario-table"),
                    ],
                    style={**CARD_STYLE, "flex": "1", "minWidth": "220px"},
                ),
            ],
            style={"display": "flex"},
        ),

        # Second row: Wavelet scalogram + CCF
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="wavelet-chart", style={"height": "320px"}),
                    style={**CARD_STYLE, "flex": "2"},
                ),
                html.Div(
                    dcc.Graph(id="ccf-chart", style={"height": "320px"}),
                    style={**CARD_STYLE, "flex": "1"},
                ),
            ],
            style={"display": "flex"},
        ),

        # Third row: LI component breakdown
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="li-components-chart", style={"height": "280px"}),
                    style={**CARD_STYLE, "flex": "2"},
                ),
                html.Div(
                    dcc.Graph(id="maturity-gauge", style={"height": "280px"}),
                    style={**CARD_STYLE, "flex": "1"},
                ),
            ],
            style={"display": "flex"},
        ),

        # Hidden store for model output
        dcc.Store(id="model-store"),
        dcc.Loading(
            id="loading",
            type="circle",
            children=html.Div(id="loading-output"),
        ),
    ],
    style={**DARK_STYLE, "minHeight": "100vh"},
)


# ── Data loading helpers ──────────────────────────────────────────────────────

def _load_model_outputs(config_path: str = "config/settings.yaml", force_refresh: bool = False):
    """
    Run the full model pipeline and return all outputs as a dict.

    This is called lazily on first render or on refresh.
    """
    from src.data.pipeline import DataPipeline
    from src.indicators.leading import build_leading_index
    from src.indicators.coincident import build_coincident_index
    from src.indicators.lagging import build_lagging_index
    from src.spectral.turning import BryBoschan
    from src.spectral.cwt import compute_cwt, dominant_period_stats
    from src.spectral.coherence import compute_coherence
    from src.projection.ccf import compute_ccf, estimate_forecast_horizon
    from src.projection.scenarios import build_scenario_tree

    logger.info("Loading data pipeline...")
    pipeline = DataPipeline.from_config(config_path)
    data = pipeline.load_all(force_refresh=force_refresh)

    logger.info("Building Leading Index...")
    li_result = build_leading_index(data)

    logger.info("Building Coincident Index...")
    ci_result = build_coincident_index(data)

    logger.info("Building Lagging Index...")
    lai_result = build_lagging_index(data)

    logger.info("Running Bry-Boschan...")
    bb = BryBoschan(mode="standard")
    btc_price = data["btc_price_usd"].dropna()
    btc_tp = bb.detect(np.log(btc_price))

    logger.info("Running wavelet CWT...")
    cwt_result = compute_cwt(np.log(btc_price))
    dom_stats = dominant_period_stats(cwt_result)

    logger.info("Running coherence analysis...")
    li_series = li_result.index
    btc_log = np.log(btc_price)
    coh_result = compute_coherence(li_series, btc_log)

    logger.info("Running CCF analysis...")
    btc_log_diff = btc_log.diff().dropna()
    ccf_result = compute_ccf(li_series, btc_log_diff)
    horizon = estimate_forecast_horizon(li_series, btc_price)

    logger.info("Building scenario tree...")
    tree = build_scenario_tree(
        btc_price=btc_price,
        li_result=li_result,
        ci_result=ci_result,
        lai_result=lai_result,
        dominant_cycle_months=dom_stats["recent_dominant_period"],
        ccf_lag_months=int(horizon["ccf_lag_months"]),
    )

    return {
        "data": data,
        "btc_price": btc_price,
        "li_result": li_result,
        "ci_result": ci_result,
        "lai_result": lai_result,
        "btc_tp": btc_tp,
        "cwt_result": cwt_result,
        "dom_stats": dom_stats,
        "coh_result": coh_result,
        "ccf_result": ccf_result,
        "horizon": horizon,
        "tree": tree,
    }


# ── Callbacks ─────────────────────────────────────────────────────────────────

_model_cache: dict = {}  # Simple in-process cache


@app.callback(
    Output("model-store", "data"),
    Output("last-updated", "children"),
    Output("loading-output", "children"),
    Input("refresh-btn", "n_clicks"),
    prevent_initial_call=False,
)
def load_model(n_clicks):
    """Load or refresh the model outputs."""
    global _model_cache
    from datetime import datetime

    force = n_clicks > 0 and bool(_model_cache)
    try:
        if not _model_cache or force:
            _model_cache = _load_model_outputs(force_refresh=force)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        return {"loaded": True}, f"Last updated: {now}", ""
    except Exception as exc:
        logger.error("Model load failed: %s", exc)
        return {"loaded": False, "error": str(exc)}, "Load failed", ""


@app.callback(
    Output("fan-chart", "figure"),
    Input("model-store", "data"),
    Input("log-scale-toggle", "value"),
    Input("history-slider", "value"),
)
def update_fan_chart(store_data, log_toggle, history_years):
    if not store_data or not store_data.get("loaded"):
        return go.Figure()

    from src.projection.fan_chart import build_fan_chart

    out = _model_cache
    fig = build_fan_chart(
        btc_price=out["btc_price"],
        scenario_tree=out["tree"],
        li_index=out["li_result"].index,
        ci_index=out["ci_result"].index,
        lai_index=out["lai_result"].index,
        turning_points=out["btc_tp"],
        log_scale="log" in (log_toggle or []),
    )
    return fig


@app.callback(
    Output("wavelet-chart", "figure"),
    Input("model-store", "data"),
    Input("history-slider", "value"),
)
def update_wavelet_chart(store_data, history_years):
    if not store_data or not store_data.get("loaded"):
        return go.Figure()

    cwt = _model_cache["cwt_result"]
    time_index = cwt.time_index
    periods = cwt.periods
    power = cwt.power

    # Log-scale power for display
    log_power = np.log10(power + 1e-12)

    cutoff = time_index[-1] - pd.DateOffset(years=history_years or 5)
    time_mask = time_index >= cutoff
    log_power_plot = log_power[:, time_mask]
    time_plot = time_index[time_mask]

    fig = go.Figure(
        go.Heatmap(
            x=time_plot,
            y=periods,
            z=log_power_plot,
            colorscale="RdBu_r",
            showscale=True,
            colorbar=dict(title="log₁₀ Power"),
        )
    )

    # Dominant period overlay
    dom_period = _model_cache["cwt_result"].dominant_period
    dom_plot = dom_period[dom_period.index >= cutoff]
    fig.add_trace(
        go.Scatter(
            x=dom_plot.index,
            y=dom_plot.values,
            mode="lines",
            name="Dominant period",
            line=dict(color="#ffa726", width=2),
        )
    )

    fig.update_layout(
        title=f"Wavelet Power Spectrum (Morlet, ω₀=6) — Dominant: {cwt.dominant_period.iloc[-1]:.0f} months",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        yaxis_title="Period (months)",
        yaxis_type="log",
        yaxis_tickvals=[12, 24, 48, 60, 96, 120],
        yaxis_ticktext=["12m", "24m", "48m", "60m", "96m", "120m"],
        height=320,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


@app.callback(
    Output("ccf-chart", "figure"),
    Input("model-store", "data"),
)
def update_ccf_chart(store_data):
    if not store_data or not store_data.get("loaded"):
        return go.Figure()

    ccf = _model_cache["ccf_result"]
    lags = ccf.lags
    corrs = ccf.correlations
    ci_lo, ci_hi = ccf.confidence_bands

    colors = [
        "#26a69a" if (l >= 0 and c > 0) else
        "#ef5350" if (l >= 0 and c < 0) else
        "rgba(255,255,255,0.2)"
        for l, c in zip(lags, corrs)
    ]

    fig = go.Figure(
        go.Bar(
            x=lags,
            y=corrs,
            marker_color=colors,
            name="CCF",
        )
    )
    # Confidence bands
    fig.add_hline(y=ci_hi, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.add_hline(y=ci_lo, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    # Peak lag marker
    fig.add_vline(
        x=ccf.peak_lag,
        line_color="#ffa726",
        line_dash="dash",
        annotation_text=f"Peak: {ccf.peak_lag}m",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"CCF: LI → BTC (peak lag = {ccf.peak_lag} months, r={ccf.peak_correlation:.2f})",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        xaxis_title="Lag (months, positive = LI leads BTC)",
        yaxis_title="Correlation",
        height=320,
        margin=dict(l=60, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


@app.callback(
    Output("li-components-chart", "figure"),
    Input("model-store", "data"),
    Input("history-slider", "value"),
)
def update_li_components(store_data, history_years):
    if not store_data or not store_data.get("loaded"):
        return go.Figure()

    li_result = _model_cache["li_result"]
    cutoff = _model_cache["btc_price"].index[-1] - pd.DateOffset(years=history_years or 5)
    comps = li_result.components[li_result.components.index >= cutoff]

    fig = go.Figure()

    # Heatmap of binary signals
    fig.add_trace(
        go.Heatmap(
            x=comps.index,
            y=comps.columns.tolist(),
            z=comps.values.T,
            colorscale=[[0, "#ef5350"], [1, "#26a69a"]],
            showscale=False,
            zmin=0,
            zmax=1,
        )
    )

    # LI overlay
    li_plot = li_result.index[li_result.index.index >= cutoff]
    fig.add_trace(
        go.Scatter(
            x=li_plot.index,
            y=li_plot.values / 100.0 * 6 - 0.5,  # scale to heatmap row space
            mode="lines",
            name="LI (scaled)",
            line=dict(color="#ffa726", width=2),
            yaxis="y",
        )
    )

    fig.update_layout(
        title=f"LI Component Signals (green=expanding, red=contracting) | Current LI: {li_result.index.iloc[-1]:.1f}",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        height=280,
        margin=dict(l=120, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


@app.callback(
    Output("maturity-gauge", "figure"),
    Input("model-store", "data"),
)
def update_maturity_gauge(store_data):
    if not store_data or not store_data.get("loaded"):
        return go.Figure()

    lai = _model_cache["lai_result"]
    maturity = float(lai.maturity_score.dropna().iloc[-1]) if len(lai.maturity_score.dropna()) else 50.0
    li_val = float(_model_cache["li_result"].index.iloc[-1]) if len(_model_cache["li_result"].index) else 50.0
    ci_val = float(_model_cache["ci_result"].index.iloc[-1]) if len(_model_cache["ci_result"].index) else 50.0

    fig = go.Figure()

    for val, label, row_y in [
        (li_val, "LI", 0.85),
        (ci_val, "CI", 0.5),
        (maturity, "LaI", 0.15),
    ]:
        color = "#26a69a" if val > 50 else "#ef5350"
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=val,
                title={"text": label, "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 33], "color": "rgba(239,83,80,0.2)"},
                        {"range": [33, 67], "color": "rgba(255,167,38,0.1)"},
                        {"range": [67, 100], "color": "rgba(38,166,154,0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
                domain={"x": [0, 1], "y": [row_y - 0.12, row_y + 0.12]},
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Index Gauges",
    )
    return fig


@app.callback(
    Output("status-li", "children"),
    Output("status-ci", "children"),
    Output("status-lai", "children"),
    Output("status-btc", "children"),
    Input("model-store", "data"),
)
def update_status_bar(store_data):
    if not store_data or not store_data.get("loaded"):
        empty = html.P("Loading...", style={"color": "#9e9e9e"})
        return empty, empty, empty, empty

    out = _model_cache

    def make_card(label, value, interpretation, color):
        return html.Div([
            html.P(label, style={"color": "#9e9e9e", "margin": "0 0 4px 0", "fontSize": "0.75rem"}),
            html.P(f"{value:.1f}" if isinstance(value, float) else str(value),
                   style={"color": color, "margin": 0, "fontSize": "1.5rem", "fontWeight": "bold"}),
            html.P(interpretation, style={"color": "#9e9e9e", "margin": 0, "fontSize": "0.7rem"}),
        ])

    li_val = float(out["li_result"].index.iloc[-1])
    ci_val = float(out["ci_result"].index.iloc[-1])
    lai_val = float(out["lai_result"].maturity_score.dropna().iloc[-1])
    btc_val = float(out["btc_price"].iloc[-1])

    li_color = "#26a69a" if li_val > 50 else "#ef5350"
    ci_color = "#26a69a" if ci_val > 50 else "#ef5350"
    lai_color = "#ef5350" if lai_val > 65 else ("#ffa726" if lai_val > 40 else "#26a69a")
    btc_color = "#ffffff"

    li_interp = "Bullish (expanding)" if li_val > 50 else "Bearish (contracting)"
    ci_interp = "Cycle expanding" if ci_val > 50 else "Cycle contracting"
    lai_interp = "Late cycle" if lai_val > 65 else ("Mid cycle" if lai_val > 40 else "Early recovery")
    btc_interp = f"Lag est: {out['horizon']['ccf_lag_months']}m | Cycle: {out['dom_stats']['recent_dominant_period']:.0f}m"

    return (
        make_card("Leading Index", li_val, li_interp, li_color),
        make_card("Coincident Index", ci_val, ci_interp, ci_color),
        make_card("Lagging Index (Maturity)", lai_val, lai_interp, lai_color),
        make_card("BTC Price", f"${btc_val:,.0f}", btc_interp, btc_color),
    )


@app.callback(
    Output("scenario-table", "children"),
    Input("model-store", "data"),
)
def update_scenario_table(store_data):
    if not store_data or not store_data.get("loaded"):
        return html.P("Loading...", style={"color": "#9e9e9e"})

    from src.projection.scenarios import scenario_summary
    tree = _model_cache["tree"]
    df = scenario_summary(tree)

    rows = [
        html.Tr([html.Th(col, style={"color": "#9e9e9e", "padding": "4px 8px"}) for col in [""] + df.columns.tolist()]),
    ]
    scenario_colors_map = {"Bull": "#26a69a", "Base": "#42a5f5", "Bear": "#ef5350"}
    for idx, row in df.iterrows():
        color = scenario_colors_map.get(str(idx), "#e0e0e0")
        cells = [html.Td(str(idx), style={"color": color, "padding": "4px 8px", "fontWeight": "bold"})]
        for val in row.values:
            cells.append(html.Td(str(val), style={"color": "#e0e0e0", "padding": "4px 8px"}))
        rows.append(html.Tr(cells))

    return html.Table(
        rows,
        style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.8rem"},
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="MDR v5 Dashboard")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logger.info("Starting MDR v5 dashboard on port %d", args.port)
    app.run(debug=args.debug, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()
