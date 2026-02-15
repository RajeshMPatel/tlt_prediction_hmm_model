# tlt_vix_web

`tlt_vix_web` is a local-first Python web project that computes H5 directional probabilities for TLT and VIX with an HMM regime model and serves results via FastAPI.

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

```bash
bash scripts/run_pipeline.sh
```

Force fresh re-download:

```bash
bash scripts/run_pipeline.sh --force-refresh
```

Pipeline output includes:

- `reports/daily/latest_probabilities.json`
- `reports/daily/status.json`
- Regime charts in `reports/charts/`
- Processed datasets in `data/processed/`
- Out-of-sample backtest metrics (time-series splits) under `latest_probabilities.json -> backtest`
- Confidence-bucket accuracy (`>=0.55`, `>=0.60`, `>=0.65`) under `backtest -> tlt/vix -> confidence_buckets`
- Action labels under `latest_probabilities.json -> actionable_signals`:
  - `STRONG_UP` / `STRONG_DOWN`
  - `MODERATE_UP` / `MODERATE_DOWN`
  - `SKIP`

### How to use daily signals (plain English)

- `STRONG_*`: highest-confidence setup based on recent out-of-sample buckets.
- `MODERATE_*`: usable but weaker edge than strong signals.
- `SKIP`: model does not see a reliable edge for today.
- `confidence` is the model certainty (`max(P_up, P_down)`).
- `expected_hit_rate` is recent backtested directional accuracy at that confidence bucket.
- `coverage` is how often that confidence bucket appears historically (lower means rarer signal).

## Run web app

```bash
bash scripts/run_server.sh
```

Open:

- `http://127.0.0.1:4000/`
- `http://127.0.0.1:4000/status`

## Cron automation (weekday mornings)

Example cron entry:

```cron
30 6 * * 1-5 /path/to/run_pipeline.sh
```

Replace `/path/to/run_pipeline.sh` with your absolute path, for example:

`/Users/rajeshpatel/dev/stocks/tlt_vix_web/scripts/run_pipeline.sh`

## Project layout

```text
tlt_vix_web/
  environment.yml
  PROJECT_SPEC.md
  README.md
  requirements.txt
  config/config.yaml
  src/
  webapp/
  scripts/
  data/raw
  data/processed
  reports/charts
  reports/daily
```
