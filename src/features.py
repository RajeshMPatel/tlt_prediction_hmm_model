"""Feature engineering for TLT/VIX H5 direction modeling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_feature_dataset(raw_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Create model features and H5 forward returns from merged raw data."""
    horizon = int(cfg["features"]["horizon_days"])
    vol_window = int(cfg["features"]["realized_vol_window"])

    df = raw_df.copy().sort_index()

    df["tlt_return"] = df["tlt_close"].pct_change()
    df["vix_return"] = df["vix_close"].pct_change()
    df[f"realized_vol_{vol_window}d"] = df["tlt_return"].rolling(vol_window).std() * np.sqrt(252)

    # FRED DGS10 is in percentage points; differencing captures daily rate change.
    df["us10y_change"] = df["DGS10"].diff()

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

    df["tlt_forward_h5_return"] = df["tlt_close"].pct_change(horizon).shift(-horizon)
    df["vix_forward_h5_return"] = df["vix_close"].pct_change(horizon).shift(-horizon)

    return df
