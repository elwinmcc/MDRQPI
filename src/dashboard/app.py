"""
MDR v5 — Dash Web Application.

Single-page dashboard focused on data and scoring:
  - Cycle Scorecard (phase, LI, CI, LaI)
  - LI + LaI component signal tables
  - Cycle timing block (phase duration, dominant period, lead time)
  - Scenario projection table (bull / base / bear targets + timeframes)
  - Model stats panel
  - Fan chart (single price-projection chart)

Usage::

    python -m src.dashboard.app
    # or
    python -m src.dashboard.app --config config/settings.yaml --port 8050

In production::

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
from dash import Dash, Input, Output, dcc, html

logger = logging.getLogger(__name__)

# ── App initialization ────────────────────────────────────────────────────────

app = Dash(
    __name__,
    title="MDR v5 — Bitcoin Cyclical Projection",
    suppress_callback_exceptions=True,
)
server = app.server  # Expose Flask server for gunicorn / Vercel

# ── Theme tokens ─────────────────────────────────────────────────────────────

BG      = "#1a1a2e"
BG2     = "#16213e"
BORDER  = "rgba(255,255,255,0.1)"
TEXT    = "#e0e0e0"
GRAY    = "#9e9e9e"
GREEN   = "#26a69a"
RED     = "#ef5350"
AMBER   = "#ffa726"
BLUE    = "#42a5f5"
FONT    = "JetBrains Mono, Courier New, monospace"

DARK_STYLE = {"backgroundColor": BG, "color": TEXT, "fontFamily": FONT}
CARD_STYLE = {
    **DARK_STYLE,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "padding": "16px",
    "margin": "8px",
}

# ── Layout helpers ────────────────────────────────────────────────────────────

def _placeholder(msg: str = "Load model to see data.") -> html.P:
    return html.P(msg, style={"color": GRAY, "margin": 0, "fontSize": "0.8rem"})


def _section_title(text: str, color: str = BLUE) -> html.H3:
    return html.H3(
        text,
        style={"color": color, "marginTop": 0, "marginBottom": "12px", "fontSize": "0.85rem",
               "letterSpacing": "0.05em", "textTransform": "uppercase"},
    )


def _stat_row(label: str, value: str, value_color: str | None = None) -> html.Div:
    return html.Div(
        [
            html.Span(
                label + ":",
                style={"color": GRAY, "fontSize": "0.78rem", "display": "inline-block",
                       "minWidth": "140px"},
            ),
            html.Span(
                value,
                style={"color": value_color or TEXT, "fontSize": "0.82rem",
                       "fontWeight": "600" if value_color else "normal"},
            ),
        ],
        style={"marginBottom": "7px"},
    )


# ── Layout ────────────────────────────────────────────────────────────────────

def make_header() -> html.Div:
    return html.Div(
        [
            html.H1(
                "MDR v5 — Bitcoin Cyclical Projection Model",
                style={"margin": 0, "fontSize": "1.4rem", "color": BLUE},
            ),
            html.P(
                "Monetary Dilution → Liquidity Cycle → Bitcoin Price Projection",
                style={"margin": 0, "color": GRAY, "fontSize": "0.85rem"},
            ),
        ],
        style={**DARK_STYLE, "padding": "16px 24px",
               "borderBottom": f"1px solid {BORDER}"},
    )


def make_controls() -> html.Div:
    return html.Div(
        [
            html.Button(
                "↻ Refresh Data",
                id="refresh-btn",
                n_clicks=0,
                style={
                    "background": "rgba(66,165,245,0.15)",
                    "border": f"1px solid {BLUE}",
                    "color": BLUE,
                    "borderRadius": "4px",
                    "padding": "5px 14px",
                    "cursor": "pointer",
                    "fontSize": "0.83rem",
                },
            ),
            html.Label(
                "History:",
                style={"color": GRAY, "marginLeft": "24px", "marginRight": "8px",
                       "fontSize": "0.83rem"},
            ),
            dcc.Slider(
                id="history-slider",
                min=1, max=10, step=1, value=5,
                marks={i: f"{i}y" for i in range(1, 11)},
                tooltip={"placement": "bottom"},
                className="dark-slider",
            ),
            html.Label(
                "Log scale:",
                style={"color": GRAY, "marginLeft": "24px", "marginRight": "6px",
                       "fontSize": "0.83rem"},
            ),
            dcc.Checklist(
                id="log-scale-toggle",
                options=[{"label": " price", "value": "log"}],
                value=["log"],
                style={"color": TEXT, "display": "inline-block", "fontSize": "0.83rem"},
            ),
            html.Div(
                id="last-updated",
                children="Click ↻ Refresh Data to load the model.",
                style={"color": "#555", "fontSize": "0.73rem", "marginLeft": "24px"},
            ),
            dcc.Loading(
                id="loading",
                type="dot",
                children=html.Div(id="loading-output"),
                style={"marginLeft": "8px"},
            ),
        ],
        style={**DARK_STYLE, "padding": "10px 24px", "display": "flex",
               "alignItems": "center", "borderBottom": f"1px solid rgba(255,255,255,0.06)"},
    )


app.layout = html.Div(
    [
        make_header(),
        make_controls(),

        # ── Row 1: Scorecard ─────────────────────────────────────────────
        html.Div(
            [
                html.Div(id="status-phase", style={"flex": 1, **CARD_STYLE}),
                html.Div(id="status-li",    style={"flex": 1, **CARD_STYLE}),
                html.Div(id="status-ci",    style={"flex": 1, **CARD_STYLE}),
                html.Div(id="status-lai",   style={"flex": 1, **CARD_STYLE}),
                html.Div(id="status-btc",   style={"flex": 1, **CARD_STYLE}),
            ],
            style={"display": "flex", "flexWrap": "wrap"},
        ),

        # ── Row 2: Component tables + Cycle timing ───────────────────────
        html.Div(
            [
                html.Div(
                    [_section_title("Leading Index — Signals"), html.Div(id="li-component-table")],
                    style={**CARD_STYLE, "flex": "1", "minWidth": "240px"},
                ),
                html.Div(
                    [_section_title("Lagging Index — Signals", AMBER), html.Div(id="lai-component-table")],
                    style={**CARD_STYLE, "flex": "1", "minWidth": "240px"},
                ),
                html.Div(
                    [_section_title("Cycle Timing", GREEN), html.Div(id="cycle-timing-block")],
                    style={**CARD_STYLE, "flex": "1", "minWidth": "220px"},
                ),
            ],
            style={"display": "flex", "flexWrap": "wrap"},
        ),

        # ── Row 3: Projection table + Model stats ────────────────────────
        html.Div(
            [
                html.Div(
                    [_section_title("Price Projections"), html.Div(id="scenario-table")],
                    style={**CARD_STYLE, "flex": "3", "minWidth": "340px"},
                ),
                html.Div(
                    [_section_title("Model Stats", GRAY), html.Div(id="model-stats-block")],
                    style={**CARD_STYLE, "flex": "1", "minWidth": "200px"},
                ),
            ],
            style={"display": "flex", "flexWrap": "wrap"},
        ),

        # ── Row 4: Fan chart (single chart) ──────────────────────────────
        html.Div(
            [_section_title("BTC Price Projection — Fan Chart"),
             dcc.Graph(id="fan-chart", style={"height": "420px"})],
            style={**CARD_STYLE},
        ),

        dcc.Store(id="model-store"),
    ],
    style={**DARK_STYLE, "minHeight": "100vh"},
)

# ── Data loading helpers ──────────────────────────────────────────────────────

def _resolve_config_path() -> str:
    """
    Return a valid config path, preferring the file but falling back to
    an in-memory config built from environment variables.

    On Vercel (and similar platforms) the settings file is absent; set the
    FRED_API_KEY environment variable instead.
    """
    import os

    file_path = "config/settings.yaml"
    if os.path.exists(file_path):
        return file_path

    fred_key = os.environ.get("FRED_API_KEY", "")
    if not fred_key:
        raise EnvironmentError(
            "No config/settings.yaml found and FRED_API_KEY env var is not set. "
            "Add FRED_API_KEY to your Vercel environment variables."
        )

    import tempfile, yaml
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(
        {
            "fred": {"api_key": fred_key},
            "data": {"cache_dir": "/tmp/mdr_cache", "start_date": "2010-01-01"},
            "model": {},
        },
        tmp,
    )
    tmp.close()
    return tmp.name


def _load_model_outputs(config_path: str | None = None, force_refresh: bool = False):
    """Run the full model pipeline and return all outputs as a dict."""
    from src.data.pipeline import DataPipeline
    from src.indicators.leading import build_leading_index
    from src.indicators.coincident import build_coincident_index
    from src.indicators.lagging import build_lagging_index
    from src.spectral.turning import BryBoschan
    from src.spectral.cwt import compute_cwt, dominant_period_stats
    from src.spectral.coherence import compute_coherence
    from src.projection.ccf import compute_ccf, estimate_forecast_horizon
    from src.projection.scenarios import build_scenario_tree

    config_path = config_path or _resolve_config_path()

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

_model_cache: dict = {}


@app.callback(
    Output("model-store", "data"),
    Output("last-updated", "children"),
    Output("loading-output", "children"),
    Input("refresh-btn", "n_clicks"),
    prevent_initial_call=True,
)
def load_model(n_clicks):
    """Load or refresh the model outputs (triggered only by the Refresh button)."""
    global _model_cache
    from datetime import datetime

    force_data = bool(_model_cache)
    try:
        _model_cache = _load_model_outputs(config_path=None, force_refresh=force_data)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        return {"loaded": True}, f"Last updated: {now}", ""
    except Exception as exc:
        logger.error("Model load failed: %s", exc)
        return {"loaded": False, "error": str(exc)}, "Load failed", ""


# ── Scorecard ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("status-phase", "children"),
    Output("status-li",    "children"),
    Output("status-ci",    "children"),
    Output("status-lai",   "children"),
    Output("status-btc",   "children"),
    Input("model-store", "data"),
)
def update_scorecard(store_data):
    empty = _placeholder()
    if not store_data or not store_data.get("loaded"):
        return empty, empty, empty, empty, empty

    out = _model_cache

    def card(label, big_text, sub, color):
        return html.Div([
            html.P(label,    style={"color": GRAY, "margin": "0 0 4px 0", "fontSize": "0.72rem", "letterSpacing": "0.06em", "textTransform": "uppercase"}),
            html.P(big_text, style={"color": color, "margin": "0 0 4px 0", "fontSize": "1.6rem", "fontWeight": "700", "lineHeight": "1"}),
            html.P(sub,      style={"color": GRAY, "margin": 0, "fontSize": "0.72rem"}),
        ])

    btc_tp   = out["btc_tp"]
    last_d   = out["btc_price"].index[-1]
    phase    = btc_tp.phase_at(last_d)
    phase_color = GREEN if phase == "expansion" else RED

    li_val  = float(out["li_result"].index.iloc[-1])
    ci_val  = float(out["ci_result"].index.iloc[-1])
    lai_val = float(out["lai_result"].maturity_score.dropna().iloc[-1])
    btc_val = float(out["btc_price"].iloc[-1])

    # LI momentum (3-month)
    from src.indicators.leading import compute_li_momentum
    li_mom = compute_li_momentum(out["li_result"].index).iloc[-1]
    li_arrow = " ↑" if li_mom > 0 else " ↓"
    li_color = GREEN if li_val > 50 else RED

    ci_color  = GREEN if ci_val > 50 else RED
    lai_color = RED if lai_val > 65 else (AMBER if lai_val > 40 else GREEN)
    lai_stage = "Late Cycle" if lai_val > 65 else ("Mid Cycle" if lai_val > 40 else "Early Recovery")

    # CI phase label
    from src.indicators.coincident import get_cycle_phase_from_ci
    ci_phases = get_cycle_phase_from_ci(out["ci_result"].index)
    ci_phase_label = str(ci_phases.iloc[-1]).replace("_", " ").title()

    return (
        card("Cycle Phase",       phase.upper(),            f"BTC Bry-Boschan",          phase_color),
        card("Leading Index",     f"{li_val:.1f}/100{li_arrow}", "Monetary liquidity",    li_color),
        card("Coincident Index",  f"{ci_val:.1f}/100",      ci_phase_label,              ci_color),
        card("Lagging Index",     f"{lai_val:.1f}/100",     lai_stage,                   lai_color),
        card("BTC Price",         f"${btc_val:,.0f}",       last_d.strftime("%b %Y"),    TEXT),
    )


# ── Component signal tables ───────────────────────────────────────────────────

def _signal_table(components_df: pd.DataFrame, raw_df: pd.DataFrame | None = None) -> html.Table:
    """Render a compact signal table from the latest row of a binary components df."""
    latest_sig = components_df.iloc[-1]
    latest_raw = raw_df.iloc[-1] if raw_df is not None else None

    rows = []
    for col in latest_sig.index:
        sig = bool(latest_sig[col])
        dot_color = GREEN if sig else RED
        raw_txt = ""
        if latest_raw is not None and col in latest_raw.index:
            v = latest_raw[col]
            raw_txt = f"  {v:+.2f}" if pd.notna(v) else ""

        rows.append(html.Tr([
            html.Td(html.Span("●", style={"color": dot_color, "fontSize": "1em"}),
                    style={"padding": "3px 6px 3px 0", "width": "16px", "verticalAlign": "middle"}),
            html.Td(col, style={"color": TEXT, "padding": "3px 0", "fontSize": "0.77rem", "verticalAlign": "middle"}),
            html.Td(raw_txt, style={"color": GRAY, "padding": "3px 0 3px 8px", "fontSize": "0.72rem",
                                    "textAlign": "right", "verticalAlign": "middle", "fontVariantNumeric": "tabular-nums"}),
        ]))

    return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"})


@app.callback(
    Output("li-component-table",  "children"),
    Output("lai-component-table", "children"),
    Input("model-store", "data"),
)
def update_component_tables(store_data):
    empty = _placeholder()
    if not store_data or not store_data.get("loaded"):
        return empty, empty

    out = _model_cache
    li_val   = float(out["li_result"].index.iloc[-1])
    lai_val  = float(out["lai_result"].maturity_score.dropna().iloc[-1])
    li_color = GREEN if li_val > 50 else RED

    from src.indicators.leading import compute_li_momentum
    li_mom = float(compute_li_momentum(out["li_result"].index).iloc[-1])

    li_table = html.Div([
        _signal_table(out["li_result"].components, out["li_result"].raw_components),
        html.Div(
            [
                _stat_row("Score", f"{li_val:.1f} / 100", li_color),
                _stat_row("3M Momentum", f"{li_mom:+.1f}", GREEN if li_mom > 0 else RED),
            ],
            style={"marginTop": "12px", "borderTop": f"1px solid {BORDER}", "paddingTop": "10px"},
        ),
    ])

    lai_color = RED if lai_val > 65 else (AMBER if lai_val > 40 else GREEN)
    lai_stage = "Late Cycle" if lai_val > 65 else ("Mid Cycle" if lai_val > 40 else "Early Recovery")
    lai_table = html.Div([
        _signal_table(out["lai_result"].components, out["lai_result"].raw_components),
        html.Div(
            [
                _stat_row("Maturity Score", f"{lai_val:.1f} / 100", lai_color),
                _stat_row("Stage", lai_stage, lai_color),
            ],
            style={"marginTop": "12px", "borderTop": f"1px solid {BORDER}", "paddingTop": "10px"},
        ),
    ])

    return li_table, lai_table


# ── Cycle timing ──────────────────────────────────────────────────────────────

@app.callback(
    Output("cycle-timing-block", "children"),
    Input("model-store", "data"),
)
def update_cycle_timing(store_data):
    if not store_data or not store_data.get("loaded"):
        return _placeholder()

    out       = _model_cache
    btc_tp    = out["btc_tp"]
    dom_stats = out["dom_stats"]
    horizon   = out["horizon"]
    last_date = out["btc_price"].index[-1]

    phase = btc_tp.phase_at(last_date)
    phase_color = GREEN if phase == "expansion" else RED

    # Last turning point
    all_turns = sorted(
        [(d, "peak") for d in btc_tp.peaks] + [(d, "trough") for d in btc_tp.troughs]
    )
    if all_turns:
        last_tp_date, last_tp_type = all_turns[-1]
        phase_months = max(1, round((last_date - last_tp_date).days / 30.44))
        last_tp_str  = f"{last_tp_date.strftime('%b %Y')} ({last_tp_type})"
    else:
        phase_months = None
        last_tp_str  = "N/A"

    dom_period = dom_stats["recent_dominant_period"]
    lag        = int(horizon["ccf_lag_months"])

    # Cycle progress through current phase (half-period = one phase)
    half = dom_period / 2
    if phase_months:
        progress_pct = min(99, round(phase_months / half * 100))
        bar_filled   = round(progress_pct / 10)  # 0-10 blocks
        bar          = "█" * bar_filled + "░" * (10 - bar_filled) + f" {progress_pct}%"
    else:
        bar = "N/A"

    # Estimated next turning point
    if all_turns and phase_months:
        months_left = max(1, round(half - phase_months))
        est_next_date = last_date + pd.DateOffset(months=months_left)
        next_event = "peak" if phase == "expansion" else "trough"
        est_str = f"~{est_next_date.strftime('%b %Y')} ({next_event})"
    else:
        est_str = "N/A"

    rows: list[html.Div] = [
        _stat_row("Current Phase",   phase.upper(),        phase_color),
        _stat_row("Phase Duration",  f"{phase_months}m" if phase_months else "N/A"),
        _stat_row("Phase Progress",  bar),
        _stat_row("Last Event",      last_tp_str),
        _stat_row("Est. Next Event", est_str,              AMBER),
        html.Div(style={"height": "10px"}),  # spacer
        _stat_row("Dominant Cycle",  f"{dom_period:.0f} months"),
        _stat_row("LI Lead Time",    f"{lag} months"),
        _stat_row("Data Through",    last_date.strftime("%b %Y")),
    ]
    return rows


# ── Projection scenario table ─────────────────────────────────────────────────

@app.callback(
    Output("scenario-table", "children"),
    Input("model-store", "data"),
)
def update_scenario_table(store_data):
    if not store_data or not store_data.get("loaded"):
        return _placeholder()

    tree = _model_cache["tree"]
    current = tree.current_price

    col_style  = {"color": GRAY, "padding": "5px 12px", "fontSize": "0.72rem",
                  "letterSpacing": "0.06em", "textTransform": "uppercase",
                  "borderBottom": f"1px solid {BORDER}"}
    cell_style = {"color": TEXT, "padding": "7px 12px", "fontSize": "0.82rem",
                  "fontVariantNumeric": "tabular-nums"}

    header = html.Tr([
        html.Th("Scenario",    style=col_style),
        html.Th("Probability", style=col_style),
        html.Th("Target",      style=col_style),
        html.Th("Return",      style=col_style),
        html.Th("Horizon",     style=col_style),
        html.Th("Peak / Trough Date", style=col_style),
    ])

    scenario_colors = {"bull": GREEN, "base": BLUE, "bear": RED}
    rows = [header]
    for key in ("bull", "base", "bear"):
        sc    = tree.scenarios[key]
        color = scenario_colors[key]
        ret_color = GREEN if sc.return_pct >= 0 else RED
        rows.append(html.Tr([
            html.Td(key.upper(),                      style={**cell_style, "color": color, "fontWeight": "700"}),
            html.Td(f"{sc.probability * 100:.0f}%",   style=cell_style),
            html.Td(f"${sc.peak_price:,.0f}",         style=cell_style),
            html.Td(f"{sc.return_pct:+.1f}%",         style={**cell_style, "color": ret_color}),
            html.Td(f"{sc.horizon_months} months",    style=cell_style),
            html.Td(sc.peak_date.strftime("%b %Y"),   style=cell_style),
        ]))

    # Expected value row
    ev = sum(sc.peak_price * sc.probability for sc in tree.scenarios.values())
    ev_ret = (ev / current - 1) * 100
    ev_color = GREEN if ev_ret >= 0 else RED
    rows.append(html.Tr([
        html.Td("EV",                  style={**cell_style, "color": GRAY, "fontStyle": "italic",
                                              "borderTop": f"1px solid {BORDER}"}),
        html.Td("—",                   style={**cell_style, "borderTop": f"1px solid {BORDER}"}),
        html.Td(f"${ev:,.0f}",         style={**cell_style, "borderTop": f"1px solid {BORDER}"}),
        html.Td(f"{ev_ret:+.1f}%",     style={**cell_style, "color": ev_color, "borderTop": f"1px solid {BORDER}"}),
        html.Td("—",                   style={**cell_style, "borderTop": f"1px solid {BORDER}"}),
        html.Td("—",                   style={**cell_style, "borderTop": f"1px solid {BORDER}"}),
    ]))

    return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"})


# ── Model stats panel ─────────────────────────────────────────────────────────

@app.callback(
    Output("model-stats-block", "children"),
    Input("model-store", "data"),
)
def update_model_stats(store_data):
    if not store_data or not store_data.get("loaded"):
        return _placeholder()

    out      = _model_cache
    ccf      = out["ccf_result"]
    dom      = out["dom_stats"]
    tree     = out["tree"]
    ci_val   = float(out["ci_result"].index.iloc[-1])
    last_d   = out["btc_price"].index[-1]

    from src.indicators.coincident import get_cycle_phase_from_ci
    ci_phase = str(get_cycle_phase_from_ci(out["ci_result"].index).iloc[-1]).replace("_", " ").title()

    rows = [
        _stat_row("CI Value",         f"{ci_val:.1f} / 100", GREEN if ci_val > 50 else RED),
        _stat_row("CI Phase",         ci_phase),
        html.Div(style={"height": "8px"}),
        _stat_row("CCF Peak Lag",     f"{ccf.peak_lag} months"),
        _stat_row("CCF Correlation",  f"{ccf.peak_correlation:.3f}"),
        html.Div(style={"height": "8px"}),
        _stat_row("Dominant Cycle",   f"{dom['recent_dominant_period']:.0f} months"),
        _stat_row("Cycle Range",      f"{dom.get('min_dominant_period', 0):.0f}–{dom.get('max_dominant_period', 0):.0f}m"),
        html.Div(style={"height": "8px"}),
        _stat_row("Current Price",    f"${tree.current_price:,.0f}"),
        _stat_row("Projection From",  tree.projection_date.strftime("%b %Y")),
    ]
    return rows


# ── Fan chart ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("fan-chart", "figure"),
    Input("model-store", "data"),
    Input("log-scale-toggle", "value"),
    Input("history-slider", "value"),
)
def update_fan_chart(store_data, log_toggle, history_years):
    if not store_data or not store_data.get("loaded"):
        return go.Figure(layout=go.Layout(
            paper_bgcolor=BG, plot_bgcolor=BG2, template="plotly_dark",
            margin=dict(l=40, r=20, t=20, b=40),
        ))

    from src.projection.fan_chart import build_fan_chart

    out = _model_cache
    return build_fan_chart(
        btc_price=out["btc_price"],
        scenario_tree=out["tree"],
        li_index=out["li_result"].index,
        ci_index=out["ci_result"].index,
        lai_index=out["lai_result"].index,
        turning_points=out["btc_tp"],
        log_scale="log" in (log_toggle or []),
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
    app.run(debug=args.debug, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()
