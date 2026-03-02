# MDR v5 — Bitcoin Cyclical Projection Model

A quantitative macro model that projects Bitcoin (and alt) price cycles from monetary dilution dynamics. Core thesis: excess liquidity (money creation not absorbed by real economic activity) drives crypto cycles. The timing structure of leading → coincident → lagging indicators provides forecasting power.

## Architecture

The model has three layers:

1. **Leading Index (LI):** 7-component equal-weight diffusion index. Signals where the cycle is GOING.
2. **Coincident Index (CI):** 2-factor Stock-Watson dynamic factor model (Kalman filter). Signals where the cycle IS.
3. **Lagging Index (LaI):** 5-component diffusion index. Confirms turns and measures cycle maturity.

Plus: wavelet spectral analysis, Bry-Boschan turning point detection, cross-correlation lag estimation, scenario tree projections.

## Implementation Order

Build in this exact sequence. Each phase has a kill switch — if validation fails, stop and reassess.

1. `src/data/` — FRED API + crypto data pipeline + caching
2. `src/spectral/turning.py` — Bry-Boschan algorithm
3. `src/indicators/leading.py` — 7-component LI diffusion
4. `src/spectral/cwt.py` + `coherence.py` — Wavelet analysis
5. `src/projection/ccf.py` — Cross-correlation function
6. `src/indicators/coincident.py` — 2-factor DFM (Kalman)
7. `src/indicators/lagging.py` — 5-component LaI diffusion
8. `src/projection/scenarios.py` + `fan_chart.py`
9. `src/dashboard/app.py` — Dash web app

## Technical Constraints

- Python 3.11+
- Key dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`, `pywt`, `fredapi`, `plotly`, `dash`
- FRED API key required (store in `config/settings.yaml`, never commit)
- All data cached locally after first fetch
- Monthly frequency is the base; higher-freq data resampled to monthly

## Key Econometric Details

- **Diffusion index:** Binary (expanding/contracting) per component. Each component: 1 if value > 6-month MA, else 0. Sum / N × 100.
- **DFM estimation:** `statsmodels.tsa.statespace.DynamicFactor` with k_factors=2, factor_order=2. EM initialization, then MLE.
- **Wavelet:** Morlet CWT via `pywt.cwt`. ω₀=6. Scales corresponding to periods 12–120 months.
- **Bry-Boschan:** Min phase = 6 months, min cycle = 15 months, smoothing = 12-month centered MA. Real-time variant uses one-sided MA.
- **Validation:** Diebold-Mariano test (1995) for forecast comparison. BB turning point alignment for LI validation.

## Leading Index Components (7)

| # | Series | FRED ID | Economic rationale |
|---|--------|---------|-------------------|
| 1 | US M2 Money Supply (YoY %) | WM2NS | Primary monetary dilution signal |
| 2 | Fed Reserve Total Assets | WALCL | QE/QT cycle driver |
| 3 | TGA Balance (inverted) | WTREGEN | Treasury spending injection |
| 4 | Overnight Reverse Repo (inverted) | RRPONTSYD | Liquidity released to markets |
| 5 | US Credit Spread (HY-IG, inverted) | BAMLH0A0HYM2 | Risk appetite / financial conditions |
| 6 | Yield Curve Slope (10Y-2Y) | T10Y2Y | Forward economic expectations |
| 7 | Trade-Weighted USD (inverted) | DTWEXBGS | Global USD liquidity |

## Coincident Index Components (2 factors via DFM)

| # | Series | Source | Economic rationale |
|---|--------|--------|-------------------|
| 1 | Bitcoin Price (log) | CoinGecko | Primary cycle coincident |
| 2 | Bitcoin 30-day Realized Vol | Derived | Cycle volatility regime |
| 3 | Total Crypto Market Cap | CoinGecko | Broad crypto coincident |
| 4 | Bitcoin Dominance | CoinGecko | Risk-on/risk-off within crypto |

## Lagging Index Components (5)

| # | Series | Source | Economic rationale |
|---|--------|--------|-------------------|
| 1 | Bitcoin Exchange Inflows (30d MA) | Derived | Distribution pressure |
| 2 | Stablecoin Dominance | CoinGecko | Defensive positioning |
| 3 | Bitcoin MVRV Ratio (6m MA) | Derived | Valuation overshoot |
| 4 | Funding Rate (perpetual futures) | Derived | Leverage / sentiment excess |
| 5 | Long-term Holder Supply % | Derived | HODLer capitulation/accumulation |

## Setup

```bash
pip install -r requirements.txt
cp config/settings.yaml.example config/settings.yaml
# Edit config/settings.yaml and add your FRED API key
python -m src.data.pipeline   # fetch and cache data
python -m src.dashboard.app   # launch dashboard
```

## Project Structure

```
MDRQPI/
├── config/
│   ├── settings.yaml          # API keys, model parameters (gitignored)
│   └── settings.yaml.example  # Template
├── data/
│   └── cache/                 # Local parquet cache (gitignored)
├── src/
│   ├── data/
│   │   ├── fred_client.py     # FRED API wrapper
│   │   ├── crypto_client.py   # CoinGecko API wrapper
│   │   ├── cache.py           # Parquet-based local cache
│   │   └── pipeline.py        # Orchestrates fetch + cache
│   ├── spectral/
│   │   ├── turning.py         # Bry-Boschan turning point detection
│   │   ├── cwt.py             # Morlet CWT wavelet analysis
│   │   └── coherence.py       # Wavelet coherence LI↔BTC
│   ├── indicators/
│   │   ├── leading.py         # 7-component LI diffusion index
│   │   ├── coincident.py      # 2-factor DFM (Kalman filter)
│   │   └── lagging.py         # 5-component LaI diffusion index
│   ├── projection/
│   │   ├── ccf.py             # Cross-correlation lag estimation
│   │   ├── scenarios.py       # Scenario tree projections
│   │   └── fan_chart.py       # Fan chart visualization
│   └── dashboard/
│       └── app.py             # Dash web application
├── tests/
│   └── ...
└── requirements.txt
```

## What NOT To Do

- Do NOT combine LI/CI/LaI into a single composite score. The cascade structure IS the model.
- Do NOT use optimized weights. Equal-weight diffusion. Small sample = overfitting.
- Do NOT assume a fixed cycle period. Wavelet analysis exists because the period is time-varying.
- Do NOT hardcode cycle progress, scenario probabilities, or lag estimates. Derive from data.
