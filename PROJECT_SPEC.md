# TLT VIX Web Project Spec

This project builds a local-first Python web app that:

- Downloads market and rates data each morning
- Engineers features for directional horizon-5 (H5) forecasting
- Fits an HMM regime model
- Produces directional probabilities for TLT and VIX
- Publishes outputs to a FastAPI website

Primary outputs:

- `P(TLT up H5)`, `P(TLT down H5)`
- `P(VIX up H5)`, `P(VIX down H5)`
- Regime probabilities
- Charts with regime shading

Data sources:

- Yahoo Finance: `TLT`, `^VIX`
- FRED: `DGS2`, `DGS10`, `DGS30`, `T10Y2Y`
- Global 10Y placeholders are included in config and handled gracefully when unavailable.

Operational constraints:

- Conda environment setup (`tlt_vix_env`)
- No Jupyter notebooks
- Local execution on Mac in VS Code/Cursor
- Deployable later to a server
