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
from pathlib import Path

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

# ── Theme tokens ──────────────────────────────────────────────────────────────

BG     = "#1a1a2e"
BG2    = "#16213e"
BORDER = "rgba(255,255,255,0.1)"
TEXT   = "#e0e0e0"
GRAY   = "#9e9e9e"
GREEN  = "#26a69a"
RED    = "#ef5350"
AMBER  = "#ffa726"
BLUE   = "#42a5f5"
FONT   = "JetBrains Mono, Courier New, monospace"

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
        style={"color": color, "marginTop": 0, "marginBottom": "12px",
               "fontSize": "0.85rem", "letterSpacing": "0.05em",
               "textTransform": "uppercase"},
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
               "alignItems": "center",
               "borderBottom": f"1px solid rgba(255,255,255,0.06)"},
    )


app.layout = html.Div(
    [
        make_header(),
        make_controls(),

        # ── Error banner (hidden when no error) ──────────────────────────
        html.Div(id="error-banner", style={"margin": "8px"}),

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
                    [_section_title("Leading Index — Signals"),
                     html.Div(id="li-component-table")],
                    style={**CARD_STYLE, "flex": "1", "minWidth": "240px"},
                ),
                html.Div(
                    [_section_title("Lagging Index — Signals", AMBER),
                     html.Div(id="lai-component-table")],
                    style={**CARD_STYLE, "flex": "1", "minWidth": "240px"},
                ),
                html.Div(
                    [_section_title("Cycle Timing", GREEN),
                     html.Div(id="cycle-timing-block")],
                    style={**CARD_STYLE, "flex": "1", "minWidth": "220px"},
                ),
            ],
            style={"display": "flex", "flexWrap": "wrap"},
        ),

        # ── Row 3: Projection table + Model stats ────────────────────────
        html.Div(
            [
                html.Div(
                    [_section_title("Price Projections"),
                     html.Div(id="scenario-table")],
                    style={**CARD_STYLE, "flex": "3", "minWidth": "340px"},
                ),
                html.Div(
                    [_section_title("Model Stats", GRAY),
                     html.Div(id="model-stats-block")],
                    style={**CARD_STYLE, "flex": "1", "minWidth": "200px"},
                ),
            ],
            style={"display": "flex", "flexWrap": "wrap"},
        ),

        # ── Row 4: Fan chart ─────────────────────────────────────────────
        html.Div(
            [_section_title("BTC Price Projection — Fan Chart"),
             dcc.Graph(id="fan-chart", style={"height": "420px"})],
            style={**CARD_STYLE},
        ),

        dcc.Store(id="model-store"),
    ],
    style={**DARK_STYLE, "minHeight": "100vh"},
)


# ── Config resolution ─────────────────────────────────────────────────────────

def _resolve_config_path() -> str:
    """Return a valid config path; fall back to env-var-built temp YAML on Vercel."""
    import os

    if os.path.exists("config/settings.yaml"):
        return "config/settings.yaml"

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


# ── Store serialisation / deserialisation ─────────────────────────────────────

def _s(series: pd.Series) -> dict:
    """Serialize a pd.Series to a {ISO-date-str: float} dict."""
    return {
        str(k)[:10]: float(v)
        for k, v in series.items()
        if pd.notna(v)
    }


def _ds(d: dict) -> pd.Series:
    """Deserialize a {date-str: float} dict back to a pd.Series."""
    if not d:
        return pd.Series(dtype=float)
    return pd.Series(
        list(d.values()),
        index=pd.to_datetime(list(d.keys())),
        dtype=float,
    ).sort_index()


def _serialize_for_store(out: dict) -> dict:
    """Convert all model outputs to a JSON-safe dict for dcc.Store."""
    li  = out["li_result"]
    ci  = out["ci_result"]
    lai = out["lai_result"]
    tree = out["tree"]
    ccf  = out["ccf_result"]
    dom  = out["dom_stats"]
    btc_tp = out["btc_tp"]
    last_date = out["btc_price"].index[-1]

    from src.indicators.leading import compute_li_momentum
    from src.indicators.coincident import get_cycle_phase_from_ci

    li_mom    = compute_li_momentum(li.index)
    ci_phases = get_cycle_phase_from_ci(ci.index)

    def _df(df: pd.DataFrame) -> dict:
        return {col: _s(df[col]) for col in df.columns}

    scenarios_ser: dict = {}
    for key, sc in tree.scenarios.items():
        scenarios_ser[key] = {
            "label": sc.label,
            "probability": float(sc.probability),
            "peak_price": float(sc.peak_price),
            "peak_date": sc.peak_date.strftime("%Y-%m-%d"),
            "return_pct": float(sc.return_pct),
            "horizon_months": int(sc.horizon_months),
            "price_path": _s(sc.price_path),
        }

    return {
        "loaded": True,
        # Price
        "btc_price": _s(out["btc_price"]),
        "btc_phase": btc_tp.phase_at(last_date),
        "btc_tp_peaks":   [d.strftime("%Y-%m-%d") for d in btc_tp.peaks],
        "btc_tp_troughs": [d.strftime("%Y-%m-%d") for d in btc_tp.troughs],
        # LI
        "li_index":       _s(li.index),
        "li_momentum_3m": float(li_mom.dropna().iloc[-1]),
        "li_components":  _df(li.components),
        "li_raw":         _df(li.raw_components),
        # CI
        "ci_index":       _s(ci.index),
        "ci_phase":       str(ci_phases.iloc[-1]),
        # LaI
        "lai_maturity":   _s(lai.maturity_score.dropna()),
        "lai_components": _df(lai.components),
        "lai_raw":        _df(lai.raw_components),
        # Spectral / CCF
        "dom_stats":      {k: float(v) for k, v in dom.items()},
        "ccf_peak_lag":   int(ccf.peak_lag),
        "ccf_peak_corr":  float(ccf.peak_correlation),
        "horizon":        {k: float(v) for k, v in out["horizon"].items()},
        # Scenarios
        "scenarios":        scenarios_ser,
        "current_price":    float(tree.current_price),
        "projection_date":  tree.projection_date.strftime("%Y-%m-%d"),
    }


def _restore_tree(store: dict):
    """Reconstruct a ScenarioTree from the serialised store."""
    from src.projection.scenarios import Scenario, ScenarioTree

    scenarios: dict = {}
    for key, s in store["scenarios"].items():
        path = _ds(s["price_path"])
        scenarios[key] = Scenario(
            label=s["label"],
            probability=s["probability"],
            horizon_months=s["horizon_months"],
            price_path=path,
            peak_price=s["peak_price"],
            peak_date=pd.Timestamp(s["peak_date"]),
            return_pct=s["return_pct"],
        )
    return ScenarioTree(
        scenarios=scenarios,
        current_price=store["current_price"],
        projection_date=pd.Timestamp(store["projection_date"]),
    )


def _restore_tp(store: dict):
    """Reconstruct a TurningPoints object from the serialised store."""
    from src.spectral.turning import TurningPoints

    peaks   = pd.to_datetime(store["btc_tp_peaks"])
    troughs = pd.to_datetime(store["btc_tp_troughs"])
    btc     = _ds(store["btc_price"])
    return TurningPoints(peaks=peaks, troughs=troughs, signal=btc)


# ── Pipeline ──────────────────────────────────────────────────────────────────

# In-process cache — used only to skip FRED re-fetching on refresh.
# All display callbacks read from dcc.Store, NOT this dict.
_raw_cache: dict = {}


def _run_pipeline(config_path: str, force_refresh: bool) -> dict:
    from src.data.pipeline import DataPipeline
    from src.indicators.leading import build_leading_index
    from src.indicators.coincident import build_coincident_index
    from src.indicators.lagging import build_lagging_index
    from src.spectral.turning import BryBoschan
    from src.spectral.cwt import compute_cwt, dominant_period_stats
    from src.spectral.coherence import compute_coherence
    from src.projection.ccf import compute_ccf, estimate_forecast_horizon
    from src.projection.scenarios import build_scenario_tree

    pipeline = DataPipeline.from_config(config_path)
    data = pipeline.load_all(force_refresh=force_refresh)

    li_result  = build_leading_index(data)
    # Reduce EM iterations to avoid serverless timeout (default=100)
    ci_result  = build_coincident_index(data, em_iter=20)
    lai_result = build_lagging_index(data)

    bb        = BryBoschan(mode="standard")
    btc_price = data["btc_price_usd"].dropna()
    btc_tp    = bb.detect(np.log(btc_price))

    cwt_result = compute_cwt(np.log(btc_price))
    dom_stats  = dominant_period_stats(cwt_result)

    li_series    = li_result.index
    btc_log      = np.log(btc_price)
    coh_result   = compute_coherence(li_series, btc_log)
    btc_log_diff = btc_log.diff().dropna()
    ccf_result   = compute_ccf(li_series, btc_log_diff)
    horizon      = estimate_forecast_horizon(li_series, btc_price)

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

@app.callback(
    Output("model-store",  "data"),
    Output("last-updated", "children"),
    Output("loading-output", "children"),
    Input("refresh-btn", "n_clicks"),
    prevent_initial_call=True,
)
def load_model(n_clicks):
    """Run the pipeline and serialise all results into dcc.Store."""
    global _raw_cache
    from datetime import datetime

    config_path  = _resolve_config_path()
    force_refresh = bool(_raw_cache)   # bypass data cache on re-run
    try:
        _raw_cache = _run_pipeline(config_path, force_refresh)
        store      = _serialize_for_store(_raw_cache)
        now        = datetime.now().strftime("%Y-%m-%d %H:%M")
        return store, f"Last updated: {now}", ""
    except Exception as exc:
        logger.exception("Model load failed")
        return {"loaded": False, "error": str(exc)}, "Load failed", ""


# ── Error banner ──────────────────────────────────────────────────────────────

@app.callback(
    Output("error-banner", "children"),
    Input("model-store", "data"),
)
def update_error_banner(store_data):
    if not store_data:
        return None
    if store_data.get("loaded"):
        return None
    err = store_data.get("error", "Unknown error")
    return html.Div(
        [
            html.Strong("Load error: ", style={"marginRight": "6px"}),
            html.Span(err, style={"fontFamily": FONT, "fontSize": "0.8rem",
                                  "wordBreak": "break-all"}),
        ],
        style={
            "background": "rgba(239,83,80,0.15)",
            "border": f"1px solid {RED}",
            "borderRadius": "6px",
            "padding": "10px 16px",
            "color": RED,
            "fontSize": "0.83rem",
        },
    )


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

    def card(label, big_text, sub, color):
        return html.Div([
            html.P(label,    style={"color": GRAY, "margin": "0 0 4px 0",
                                    "fontSize": "0.72rem", "letterSpacing": "0.06em",
                                    "textTransform": "uppercase"}),
            html.P(big_text, style={"color": color, "margin": "0 0 4px 0",
                                    "fontSize": "1.6rem", "fontWeight": "700",
                                    "lineHeight": "1"}),
            html.P(sub,      style={"color": GRAY, "margin": 0, "fontSize": "0.72rem"}),
        ])

    btc_price  = _ds(store_data["btc_price"])
    last_d     = btc_price.index[-1]
    phase      = store_data["btc_phase"]
    phase_color = GREEN if phase == "expansion" else RED

    li_val  = _ds(store_data["li_index"]).iloc[-1]
    ci_val  = _ds(store_data["ci_index"]).iloc[-1]
    lai_val = _ds(store_data["lai_maturity"]).iloc[-1]
    btc_val = btc_price.iloc[-1]

    li_mom   = store_data["li_momentum_3m"]
    li_arrow = " ↑" if li_mom > 0 else " ↓"
    li_color = GREEN if li_val > 50 else RED
    ci_color = GREEN if ci_val > 50 else RED
    lai_color = RED if lai_val > 65 else (AMBER if lai_val > 40 else GREEN)
    lai_stage = "Late Cycle" if lai_val > 65 else ("Mid Cycle" if lai_val > 40 else "Early Recovery")

    ci_phase_label = store_data["ci_phase"].replace("_", " ").title()

    return (
        card("Cycle Phase",      phase.upper(),
             "BTC Bry-Boschan",               phase_color),
        card("Leading Index",    f"{li_val:.1f}/100{li_arrow}",
             "Monetary liquidity",             li_color),
        card("Coincident Index", f"{ci_val:.1f}/100",
             ci_phase_label,                   ci_color),
        card("Lagging Index",    f"{lai_val:.1f}/100",
             lai_stage,                        lai_color),
        card("BTC Price",        f"${btc_val:,.0f}",
             last_d.strftime("%b %Y"),          TEXT),
    )


# ── Component signal tables ───────────────────────────────────────────────────

def _signal_table(components: dict, raw: dict) -> html.Table:
    """Render a signal table from serialised component dicts."""
    rows = []
    for col in components:
        sig_series = _ds(components[col])
        sig = bool(sig_series.iloc[-1]) if len(sig_series) else False

        raw_txt = ""
        if col in raw:
            rv = _ds(raw[col])
            if len(rv):
                raw_txt = f"  {rv.iloc[-1]:+.2f}"

        rows.append(html.Tr([
            html.Td(
                html.Span("●", style={"color": GREEN if sig else RED, "fontSize": "1em"}),
                style={"padding": "3px 6px 3px 0", "width": "16px",
                       "verticalAlign": "middle"},
            ),
            html.Td(col,     style={"color": TEXT, "padding": "3px 0",
                                    "fontSize": "0.77rem", "verticalAlign": "middle"}),
            html.Td(raw_txt, style={"color": GRAY, "padding": "3px 0 3px 8px",
                                    "fontSize": "0.72rem", "textAlign": "right",
                                    "verticalAlign": "middle",
                                    "fontVariantNumeric": "tabular-nums"}),
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

    li_val  = _ds(store_data["li_index"]).iloc[-1]
    lai_val = _ds(store_data["lai_maturity"]).iloc[-1]
    li_color  = GREEN if li_val > 50 else RED
    lai_color = RED if lai_val > 65 else (AMBER if lai_val > 40 else GREEN)
    lai_stage = "Late Cycle" if lai_val > 65 else ("Mid Cycle" if lai_val > 40 else "Early Recovery")
    li_mom = store_data["li_momentum_3m"]

    li_table = html.Div([
        _signal_table(store_data["li_components"], store_data["li_raw"]),
        html.Div(
            [_stat_row("Score",       f"{li_val:.1f} / 100", li_color),
             _stat_row("3M Momentum", f"{li_mom:+.1f}",      GREEN if li_mom > 0 else RED)],
            style={"marginTop": "12px", "borderTop": f"1px solid {BORDER}",
                   "paddingTop": "10px"},
        ),
    ])

    lai_table = html.Div([
        _signal_table(store_data["lai_components"], store_data["lai_raw"]),
        html.Div(
            [_stat_row("Maturity Score", f"{lai_val:.1f} / 100", lai_color),
             _stat_row("Stage",          lai_stage,               lai_color)],
            style={"marginTop": "12px", "borderTop": f"1px solid {BORDER}",
                   "paddingTop": "10px"},
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

    phase     = store_data["btc_phase"]
    dom       = store_data["dom_stats"]
    horizon   = store_data["horizon"]
    btc_price = _ds(store_data["btc_price"])
    last_date = btc_price.index[-1]

    peaks   = pd.to_datetime(store_data["btc_tp_peaks"])
    troughs = pd.to_datetime(store_data["btc_tp_troughs"])

    all_turns = sorted(
        [(d, "peak") for d in peaks] + [(d, "trough") for d in troughs]
    )

    if all_turns:
        last_tp_date, last_tp_type = all_turns[-1]
        phase_months = max(1, round((last_date - last_tp_date).days / 30.44))
        last_tp_str  = f"{last_tp_date.strftime('%b %Y')} ({last_tp_type})"
    else:
        phase_months = None
        last_tp_str  = "N/A"

    dom_period = dom["recent_dominant_period"]
    lag        = int(horizon["ccf_lag_months"])
    half       = dom_period / 2

    if phase_months:
        pct       = min(99, round(phase_months / half * 100))
        bar_len   = round(pct / 10)
        bar       = "█" * bar_len + "░" * (10 - bar_len) + f" {pct}%"
        months_left = max(1, round(half - phase_months))
        est_date    = last_date + pd.DateOffset(months=months_left)
        next_event  = "peak" if phase == "expansion" else "trough"
        est_str     = f"~{est_date.strftime('%b %Y')} ({next_event})"
    else:
        bar     = "N/A"
        est_str = "N/A"

    phase_color = GREEN if phase == "expansion" else RED

    return [
        _stat_row("Current Phase",   phase.upper(),                                phase_color),
        _stat_row("Phase Duration",  f"{phase_months}m" if phase_months else "N/A"),
        _stat_row("Phase Progress",  bar),
        _stat_row("Last Event",      last_tp_str),
        _stat_row("Est. Next Event", est_str,                                      AMBER),
        html.Div(style={"height": "10px"}),
        _stat_row("Dominant Cycle",  f"{dom_period:.0f} months"),
        _stat_row("LI Lead Time",    f"{lag} months"),
        _stat_row("Data Through",    last_date.strftime("%b %Y")),
    ]


# ── Scenario projection table ─────────────────────────────────────────────────

@app.callback(
    Output("scenario-table", "children"),
    Input("model-store", "data"),
)
def update_scenario_table(store_data):
    if not store_data or not store_data.get("loaded"):
        return _placeholder()

    current = store_data["current_price"]
    scens   = store_data["scenarios"]

    cs = {"color": GRAY, "padding": "5px 12px", "fontSize": "0.72rem",
          "letterSpacing": "0.06em", "textTransform": "uppercase",
          "borderBottom": f"1px solid {BORDER}"}
    cd = {"color": TEXT, "padding": "7px 12px", "fontSize": "0.82rem",
          "fontVariantNumeric": "tabular-nums"}

    header = html.Tr([
        html.Th("Scenario",    style=cs),
        html.Th("Probability", style=cs),
        html.Th("Target",      style=cs),
        html.Th("Return",      style=cs),
        html.Th("Horizon",     style=cs),
        html.Th("Date",        style=cs),
    ])

    colors = {"bull": GREEN, "base": BLUE, "bear": RED}
    rows = [header]
    ev_num = 0.0
    for key in ("bull", "base", "bear"):
        s     = scens[key]
        color = colors[key]
        ret_c = GREEN if s["return_pct"] >= 0 else RED
        ev_num += s["peak_price"] * s["probability"]
        rows.append(html.Tr([
            html.Td(key.upper(),                        style={**cd, "color": color, "fontWeight": "700"}),
            html.Td(f"{s['probability'] * 100:.0f}%",  style=cd),
            html.Td(f"${s['peak_price']:,.0f}",        style=cd),
            html.Td(f"{s['return_pct']:+.1f}%",        style={**cd, "color": ret_c}),
            html.Td(f"{s['horizon_months']} months",   style=cd),
            html.Td(pd.Timestamp(s["peak_date"]).strftime("%b %Y"), style=cd),
        ]))

    ev_ret = (ev_num / current - 1) * 100
    rows.append(html.Tr([
        html.Td("EV", style={**cd, "color": GRAY, "fontStyle": "italic",
                              "borderTop": f"1px solid {BORDER}"}),
        html.Td("—",  style={**cd, "borderTop": f"1px solid {BORDER}"}),
        html.Td(f"${ev_num:,.0f}", style={**cd, "borderTop": f"1px solid {BORDER}"}),
        html.Td(f"{ev_ret:+.1f}%",
                style={**cd, "color": GREEN if ev_ret >= 0 else RED,
                       "borderTop": f"1px solid {BORDER}"}),
        html.Td("—",  style={**cd, "borderTop": f"1px solid {BORDER}"}),
        html.Td("—",  style={**cd, "borderTop": f"1px solid {BORDER}"}),
    ]))

    return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"})


# ── Model stats ───────────────────────────────────────────────────────────────

@app.callback(
    Output("model-stats-block", "children"),
    Input("model-store", "data"),
)
def update_model_stats(store_data):
    if not store_data or not store_data.get("loaded"):
        return _placeholder()

    ci_val  = _ds(store_data["ci_index"]).iloc[-1]
    dom     = store_data["dom_stats"]
    ci_phase = store_data["ci_phase"].replace("_", " ").title()

    return [
        _stat_row("CI Value",        f"{ci_val:.1f} / 100",
                  GREEN if ci_val > 50 else RED),
        _stat_row("CI Phase",        ci_phase),
        html.Div(style={"height": "8px"}),
        _stat_row("CCF Peak Lag",    f"{store_data['ccf_peak_lag']} months"),
        _stat_row("CCF Correlation", f"{store_data['ccf_peak_corr']:.3f}"),
        html.Div(style={"height": "8px"}),
        _stat_row("Dominant Cycle",  f"{dom['recent_dominant_period']:.0f} months"),
        _stat_row("Cycle Range",
                  f"{dom.get('min_dominant_period', 0):.0f}–"
                  f"{dom.get('max_dominant_period', 0):.0f}m"),
        html.Div(style={"height": "8px"}),
        _stat_row("Current Price",   f"${store_data['current_price']:,.0f}"),
        _stat_row("Projection From",
                  pd.Timestamp(store_data["projection_date"]).strftime("%b %Y")),
    ]


# ── Fan chart ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("fan-chart", "figure"),
    Input("model-store", "data"),
    Input("log-scale-toggle", "value"),
    Input("history-slider", "value"),
)
def update_fan_chart(store_data, log_toggle, history_years):
    empty_fig = go.Figure(layout=go.Layout(
        paper_bgcolor=BG, plot_bgcolor=BG2, template="plotly_dark",
        margin=dict(l=40, r=20, t=20, b=40),
    ))
    if not store_data or not store_data.get("loaded"):
        return empty_fig

    try:
        from src.projection.fan_chart import build_fan_chart

        return build_fan_chart(
            btc_price=_ds(store_data["btc_price"]),
            scenario_tree=_restore_tree(store_data),
            li_index=_ds(store_data["li_index"]),
            ci_index=_ds(store_data["ci_index"]),
            lai_index=_ds(store_data["lai_maturity"]),
            turning_points=_restore_tp(store_data),
            log_scale="log" in (log_toggle or []),
        )
    except Exception as exc:
        logger.exception("Fan chart build failed: %s", exc)
        return empty_fig


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="MDR v5 Dashboard")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--port",   type=int, default=8050)
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()
    app.run(debug=args.debug, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()
