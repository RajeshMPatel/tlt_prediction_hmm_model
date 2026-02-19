"""Feature engineering for TLT/VIX H5 direction modeling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute classic RSI from price series."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _min_max_scale(series: pd.Series, window: int = 125) -> pd.Series:
    """Rolling min-max scaling to 0-100 range."""
    roll_min = series.rolling(window).min()
    roll_max = series.rolling(window).max()
    # Avoid division by zero
    denom = (roll_max - roll_min).replace(0.0, np.nan)
    scaled = (series - roll_min) / denom
    return scaled * 100.0


def build_feature_dataset(raw_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Create model features and H5 forward returns from merged raw data."""
    horizon = int(cfg["features"]["horizon_days"])
    vol_window = int(cfg["features"]["realized_vol_window"])

    df = raw_df.copy().sort_index()
    tlt_close_ffill = df["tlt_close"].ffill()
    vix_close_ffill = df["vix_close"].ffill()
    spy_close_ffill = df.get("spy_close", pd.Series(np.nan, index=df.index)).ffill()
    dxy_close_ffill = df.get("dxy_close", pd.Series(np.nan, index=df.index)).ffill()

    # Use forward-filled closes so weekend/holiday gaps do not invalidate
    # the next trading day's return and rolling volatility features.
    df["tlt_return"] = tlt_close_ffill.pct_change()
    df["vix_return"] = vix_close_ffill.pct_change()
    df[f"realized_vol_5d"] = df["tlt_return"].rolling(5).std() * np.sqrt(252)
    df[f"realized_vol_{vol_window}d"] = df["tlt_return"].rolling(vol_window).std() * np.sqrt(252)
    df[f"realized_vol_60d"] = df["tlt_return"].rolling(60).std() * np.sqrt(252)
    df["tlt_rsi_14"] = _rsi(tlt_close_ffill, window=14)
    df["vix_rsi_14"] = _rsi(vix_close_ffill, window=14)

    # New Features: DXY and Correlation
    df["dxy_return"] = dxy_close_ffill.pct_change()
    # 60-day rolling correlation between TLT and SPY returns
    spy_ret = spy_close_ffill.pct_change()
    df["tlt_spy_corr_60d"] = df["tlt_return"].rolling(60).corr(spy_ret)

    # Synthetic Fear & Greed Index (0-100)
    # -------------------------------------------------------------------------
    # User requested a "Sentiment" indicator similar to CNN Fear & Greed.
    # Since we cannot scrape CNN daily, we build a synthetic version using:
    # 1. VIX Component (Inverted: Low VIX = Greed/High Score)
    # 2. SPY Momentum (Price vs 125d MA)
    # 3. Credit Spread (Inverted: Low Spread = Greed/High Score)
    
    # VIX Score (Lower is better/greedier)
    vix_norm = _min_max_scale(vix_close_ffill, window=125)
    vix_score = 100.0 - vix_norm

    # SPY Momentum Score
    spy_ma_125 = spy_close_ffill.rolling(125).mean()
    spy_mom = (spy_close_ffill / spy_ma_125) - 1.0
    spy_score = _min_max_scale(spy_mom, window=125)

    # Credit Score (Lower spread is better/greedier)
    # Ensure we have credit spread data before using it
    credit_series = df.get("BAMLH0A0HYM2", pd.Series(np.nan, index=df.index)).ffill()
    credit_norm = _min_max_scale(credit_series, window=125)
    credit_score = 100.0 - credit_norm

    # Average the components available
    df["synthetic_fear_greed"] = (vix_score + spy_score + credit_score) / 3.0
    df["synthetic_fear_greed"] = df["synthetic_fear_greed"].ffill()

    # Trend and channel-style features.
    df["tlt_mom_5"] = tlt_close_ffill.pct_change(5)
    df["tlt_mom_20"] = tlt_close_ffill.pct_change(20)
    df["vix_mom_5"] = vix_close_ffill.pct_change(5)
    df["vix_mom_20"] = vix_close_ffill.pct_change(20)

    tlt_ma_20 = tlt_close_ffill.rolling(20).mean()
    vix_ma_20 = vix_close_ffill.rolling(20).mean()
    df["tlt_trend_gap_20"] = (tlt_close_ffill / tlt_ma_20) - 1.0
    df["vix_trend_gap_20"] = (vix_close_ffill / vix_ma_20) - 1.0

    tlt_min_20 = tlt_close_ffill.rolling(20).min()
    tlt_max_20 = tlt_close_ffill.rolling(20).max()
    vix_min_20 = vix_close_ffill.rolling(20).min()
    vix_max_20 = vix_close_ffill.rolling(20).max()
    eps = 1e-12
    df["tlt_channel_pos_20"] = (tlt_close_ffill - tlt_min_20) / (tlt_max_20 - tlt_min_20 + eps)
    df["vix_channel_pos_20"] = (vix_close_ffill - vix_min_20) / (vix_max_20 - vix_min_20 + eps)

    # Expose recent rolling min/max in outputs for context.
    df["tlt_20d_min_close"] = tlt_min_20
    df["tlt_20d_max_close"] = tlt_max_20
    df["vix_20d_min_close"] = vix_min_20
    df["vix_20d_max_close"] = vix_max_20

    # FRED DGS10 is in percentage points; differencing captures daily rate change.
    df["us10y_change"] = df["DGS10"].diff()
    df["inflation_expectations"] = df.get("T10YIE", np.nan).ffill()
    # Real Yield = Nominal 10Y (DGS10) - Inflation Expectations (T10YIE)
    # Both are in percent, so simple subtraction works.
    #
    # NOTE (User Request):
    # The user asked about "Truflation" (real-time spot inflation).
    # We use T10YIE (Breakeven Inflation) instead because:
    # 1. History: We need 15+ years of data for the HMM (Truflation has ~1 year free).
    # 2. Relevance: TLT is a long-duration asset driven by *future* inflation expectations (10Y),
    #    not current spot prices. T10YIE captures the market's 10-year outlook.
    df["real_yield_10y"] = df["DGS10"] - df["inflation_expectations"]
    df["real_yield_10y"] = df["real_yield_10y"].ffill()
    
    df["credit_spread"] = df.get("BAMLH0A0HYM2", np.nan).ffill()

    if "T10Y2Y" in df.columns:
        df["curve_slope"] = df["T10Y2Y"]
    else:
        df["curve_slope"] = df.get("DGS10", np.nan) - df.get("DGS2", np.nan)

    global_cols = [c for c in ["JAPAN10Y", "GERMANY10Y", "UK10Y"] if c in df.columns]
    for col in global_cols:
        df[f"{col}_change"] = df[col].diff()

    if global_cols:
        change_cols = [f"{col}_change" for col in global_cols]
        df["global_yield_change_mean"] = df[change_cols].mean(axis=1, skipna=True)
        # Keep feature matrix usable when some global series are sparse.
        df["global_yield_change_mean"] = df["global_yield_change_mean"].fillna(0.0)
    else:
        # Optional global yields may be unavailable; use a neutral default.
        df["global_yield_change_mean"] = 0.0

    df["tlt_forward_h5_return"] = tlt_close_ffill.pct_change(horizon).shift(-horizon)
    df["vix_forward_h5_return"] = vix_close_ffill.pct_change(horizon).shift(-horizon)

    return df
