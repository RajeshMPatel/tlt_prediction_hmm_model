# TLT & VIX Predictive HMM Model

This project is a local-first Python web application that uses a **Hidden Markov Model (HMM)** to forecast the 5-day directional probabilities for Long-Term Treasury Bonds (**TLT**) and the Volatility Index (**VIX**).

By analyzing macroeconomic data, technical indicators, and historical volatility, the model identifies the current "Market Regime" (e.g., Risk-On, Defensive Bid, Broad Softness) and outputs actionable, data-backed trading signals.

## What It Does

1. **Automated Data Ingestion**: Downloads daily market data (TLT, VIX, SPY, DXY) directly from Yahoo Finance and macroeconomic data (Treasury Yields, Inflation Expectations, Credit Spreads) from the St. Louis Fed (FRED).
2. **Feature Engineering**: Calculates sophisticated technical indicators including Multi-timeframe Volatility, RSI, Trend Gaps, Synthetic Fear & Greed, and Real Yields.
3. **Ensemble HMM Regime Detection**: Trains an ensemble of 10 Gaussian Hidden Markov Models to classify the current market environment into distinct Regimes.
4. **Directional Forecasting**: Calculates the historical probability of TLT and VIX moving UP over the next 5 trading days based on the current regime.
5. **Actionable Signals**: Translates probabilities into plain-English trading signals (`STRONG`, `MODERATE`, `SKIP`) by backtesting against historical "confidence buckets" to ensure the edge is statistically significant.
6. **Web Dashboard**: Serves all predictions, backtest results, and regime charts via a sleek local FastAPI web interface.

## How It Works

The core of the system is the **Ensemble HMM**. Financial markets are noisy and non-stationary. A simple rule like "Buy when RSI > 70" might work in a low-volatility bull market but fail catastrophically in a high-volatility bear market. 

### What is a "Regime"?
A "Regime" is a distinct state or environment that the market is currently operating in. Just like the weather changes from sunny to stormy, the stock and bond markets transition between periods of calm, panic, inflation-fears, and deflation-fears. The HMM's job is to mathematically identify these invisible "weather patterns" by looking at the relationships between our data points (volatility, trend, yields, and credit spreads) at any given time.

### Why 3 Regimes?
We specifically configured the model to group the market into **3 Regimes**. In financial modeling for bonds and volatility, 3 states provide the optimal balance between capturing nuance and avoiding "overfitting":
1. **Calm / Risk-On**: Typically characterized by low volatility, steady or rising bond prices, and tight credit spreads.
2. **Stress / Risk-Off**: Characterized by high volatility, spiking VIX, and rapid yield changes.
3. **Transition / Choppy**: A middle-ground state where trends are unclear and correlations break down.

If we chose only 2 regimes, the model would be too simplistic (forcing everything into just "Good" or "Bad"). If we chose 5 or 6 regimes, the model would start finding random micro-patterns that don't actually mean anything (overfitting), making its future predictions unreliable. 3 regimes give us a clear, actionable macro context.

### The Ensemble Approach
The HMM acts as a context-engine:
- It looks at all the features and groups historical days into our 3 distinct Regimes.
- We use an **Ensemble** of 10 models (different random seeds) to reduce the "coin flip" variance that plagues single HMM runs. They vote on the current regime to provide a robust consensus.
- Once the current regime is diagnosed, the model looks at what happened historically when the market was in this exact state, generating the 5-day (H5) up/down probabilities.

## Environment setup (Conda)

```bash
conda env create -f environment.yml
conda activate tlt_vix_env
```

## Required environment variable

Set your FRED key before running:

```bash
export FRED_API_KEY="your_key_here"
```

If `FRED_API_KEY` is missing, the pipeline continues with graceful fallbacks for FRED-dependent series.

Yahoo downloads use direct HTTP requests to the Yahoo Chart API, with retry + exponential backoff by default. Tune these knobs in `config/config.yaml` under `data.yfinance` (`max_retries`, `retry_base_seconds`, `retry_backoff`, `request_pause_seconds`) if your network is frequently rate-limited.

If Yahoo remains rate-limited, the loader can fall back to Stooq symbols (`data.stooq`) to keep the pipeline running.

Additional FRED fallback behavior is enabled:
- `VIXCLS` is used as `vix_close` if market-source VIX download fails.
- A yield-based synthetic `tlt_close` proxy is built from `DGS20` (or `DGS30`) if TLT download fails.

## Run pipeline

To download today's data, train the model, and generate new predictions:

```bash
bash scripts/run_pipeline.sh
```

Force fresh re-download (ignores cache):

```bash
bash scripts/run_pipeline.sh --force-refresh
```

Pipeline output includes:

- `reports/daily/latest_probabilities.json` (The raw JSON output of the model)
- `reports/daily/status.json`
- Regime charts in `reports/charts/`
- Processed datasets in `data/processed/`
- Out-of-sample backtest metrics (time-series splits) under `latest_probabilities.json -> backtest`
- Action labels under `latest_probabilities.json -> actionable_signals`:
  - `STRONG_UP` / `STRONG_DOWN`
  - `MODERATE_UP` / `MODERATE_DOWN`
  - `SKIP`

### How to use daily signals (plain English)

When you look at the dashboard or the JSON output, you will see a specific action recommended for both TLT and VIX:

- `STRONG_*`: The highest-confidence setup. The model's probability is high, and historically, trades taken at this confidence level have a high win rate (e.g., >62%).
- `MODERATE_*`: Usable, but a weaker edge than strong signals. Good for scaling into positions or taking smaller sizing.
- `SKIP`: The model does not see a reliable edge for today. The probability is either a coin-flip (~50/50), or the historical backtest for this setup is unprofitable. It means "Do not initiate new risk".
- `confidence`: The model certainty (`max(P_up, P_down)`).
- `expected_hit_rate`: The recent backtested directional accuracy for this specific confidence bucket.
- `coverage`: How often that confidence bucket appears historically (lower means it's a rarer, more extreme setup).

## Run web app

To view the dashboard, regime charts, and predictions in your browser:

```bash
bash scripts/run_server.sh
```

Open:

- `http://127.0.0.1:4000/` (Main Dashboard)
- `http://127.0.0.1:4000/status` (Health Check)

## Cron automation (weekday mornings)

If you want the model to run automatically every morning, you can add it to your crontab.

Example cron entry (runs at 6:30 AM Mon-Fri):

```cron
30 6 * * 1-5 /path/to/run_pipeline.sh
```

Replace `/path/to/run_pipeline.sh` with your absolute path, for example:

`/Users/rajeshpatel/dev/stocks/tlt_vix_web/scripts/run_pipeline.sh`

## Project layout

```text
tlt_vix_web/
  environment.yml      # Conda dependencies
  PROJECT_SPEC.md      # Core project requirements
  README.md            
  requirements.txt     # Pip dependencies
  config/
    config.yaml        # Main configuration (features, horizons, ensemble size)
  src/                 # Core Python logic (data_loader, features, model, reporting)
  webapp/              # FastAPI server and HTML/CSS templates
  scripts/             # Bash scripts to run pipeline and server
  data/                
    raw/               # Cached parquet downloads
    processed/         # Feature-engineered matrices
  reports/
    charts/            # Auto-generated Matplotlib regime charts
    daily/             # JSON prediction outputs
```
