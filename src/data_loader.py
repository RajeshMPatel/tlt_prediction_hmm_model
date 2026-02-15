"""Data download and caching utilities."""

from __future__ import annotations

import os
import time
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from fredapi import Fred


@dataclass
class DownloadResult:
    """Container for raw data artifacts."""

    raw_frame: pd.DataFrame
    metadata: dict[str, Any]


def _read_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        df = pd.read_parquet(path)
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        df.index = idx
        return df.sort_index()
    return None


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_index().to_parquet(path)


def _today_utc_date() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize(None).normalize()


def _coerce_cached_close_frame(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Coerce legacy cached market frames into one close column."""
    if df.empty:
        out = pd.DataFrame(index=df.index)
        out[column_name] = pd.Series(dtype=float)
        return out

    close_series: pd.Series
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        level1 = df.columns.get_level_values(1)
        if "Close" in level0:
            close_slice = df.xs("Close", axis=1, level=0, drop_level=True)
        elif "Close" in level1:
            close_slice = df.xs("Close", axis=1, level=1, drop_level=True)
        elif "Adj Close" in level0:
            close_slice = df.xs("Adj Close", axis=1, level=0, drop_level=True)
        elif "Adj Close" in level1:
            close_slice = df.xs("Adj Close", axis=1, level=1, drop_level=True)
        else:
            close_slice = df.iloc[:, :1]

        if isinstance(close_slice, pd.Series):
            close_series = close_slice
        else:
            close_series = close_slice.iloc[:, 0]
    else:
        if column_name in df.columns:
            close_data = df[column_name]
        elif "Close" in df.columns:
            close_data = df["Close"]
        elif "Adj Close" in df.columns:
            close_data = df["Adj Close"]
        else:
            close_data = df.iloc[:, 0]

        close_series = close_data if isinstance(close_data, pd.Series) else close_data.iloc[:, 0]

    out = close_series.to_frame(name=column_name)
    out[column_name] = pd.to_numeric(out[column_name], errors="coerce")
    return out.dropna(subset=[column_name]).sort_index()


def _ensure_single_level_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten column index to a single level for safe joins."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(part) for part in col if part is not None and str(part) != "").strip("_")
            for col in out.columns
        ]
    out = out.loc[:, ~out.columns.duplicated(keep="last")]
    return out


def _extract_close_frame(
    downloaded: pd.DataFrame,
    symbol: str,
    column_name: str,
) -> pd.DataFrame:
    """Normalize yfinance output to a single close-price column."""
    if isinstance(downloaded.columns, pd.MultiIndex):
        level0 = downloaded.columns.get_level_values(0)
        level1 = downloaded.columns.get_level_values(1)

        if "Close" in level0:
            close_slice = downloaded.xs("Close", axis=1, level=0, drop_level=True)
        elif "Close" in level1:
            close_slice = downloaded.xs("Close", axis=1, level=1, drop_level=True)
        elif "Adj Close" in level0:
            close_slice = downloaded.xs("Adj Close", axis=1, level=0, drop_level=True)
        elif "Adj Close" in level1:
            close_slice = downloaded.xs("Adj Close", axis=1, level=1, drop_level=True)
        else:
            raise RuntimeError(
                f"Unable to find Close/Adj Close in yfinance result for {symbol}."
            )

        if isinstance(close_slice, pd.Series):
            close_series = close_slice
        elif symbol in close_slice.columns:
            close_series = close_slice[symbol]
        elif close_slice.shape[1] == 1:
            close_series = close_slice.iloc[:, 0]
        else:
            raise RuntimeError(
                f"Ambiguous close columns for {symbol}: {list(close_slice.columns)}"
            )
    else:
        if "Close" in downloaded.columns:
            close_series = downloaded["Close"]
        elif "Adj Close" in downloaded.columns:
            close_series = downloaded["Adj Close"]
        else:
            raise RuntimeError(
                f"Unable to find Close/Adj Close in yfinance result for {symbol}."
            )
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

    close = close_series.to_frame(name=column_name)
    idx = pd.to_datetime(close.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    close.index = idx
    return close.sort_index()


def _download_yahoo_chart_request(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily data from Yahoo Chart API via direct HTTP request."""
    period1 = int(pd.Timestamp(start_date).timestamp())
    period2 = int(pd.Timestamp(end_date).timestamp())
    encoded_symbol = urllib.parse.quote(symbol)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history"
        "&includeAdjustedClose=true"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=20) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart API error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        return pd.DataFrame()

    result = results[0]
    timestamps = result.get("timestamp") or []
    if not timestamps:
        return pd.DataFrame()

    indicators = result.get("indicators", {})
    quote_list = indicators.get("quote") or []
    if not quote_list:
        return pd.DataFrame()

    quote = quote_list[0]
    closes = quote.get("close") or []
    if len(closes) != len(timestamps):
        return pd.DataFrame()

    idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None)
    df = pd.DataFrame({"Close": closes}, index=idx)
    df.index.name = "Date"
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df.sort_index()


def _download_yahoo_with_retries(
    symbol: str,
    start_date: str,
    end_date: str,
    max_retries: int,
    retry_base_seconds: float,
    retry_backoff: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download Yahoo chart data with retry/backoff for transient failures."""
    last_error: str | None = None
    attempt_details: list[dict[str, Any]] = []
    for attempt in range(1, max_retries + 1):
        try:
            downloaded = _download_yahoo_chart_request(
                symbol=symbol, start_date=start_date, end_date=end_date
            )
            if not downloaded.empty:
                return downloaded, {
                    "attempts": attempt,
                    "strategy": "yahoo_chart_api",
                    "attempt_details": attempt_details,
                }

            last_error = "empty dataframe returned"
            attempt_details.append({"attempt": attempt, "status": "empty"})
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            attempt_details.append(
                {"attempt": attempt, "status": "error", "error": last_error}
            )

        if attempt < max_retries:
            sleep_seconds = retry_base_seconds * (retry_backoff ** (attempt - 1))
            time.sleep(sleep_seconds)

    return pd.DataFrame(), {
        "attempts": max_retries,
        "strategy": "yahoo_chart_api",
        "attempt_details": attempt_details,
        "error": last_error,
    }


def _download_stooq_close(
    stooq_symbol: str,
    column_name: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily close series from Stooq as a final fallback source."""
    encoded_symbol = urllib.parse.quote(stooq_symbol)
    url = f"https://stooq.com/q/d/l/?s={encoded_symbol}&i=d"

    with urllib.request.urlopen(url, timeout=15) as response:  # noqa: S310
        csv_text = response.read().decode("utf-8")

    df = pd.read_csv(StringIO(csv_text))
    if df.empty or "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    close = df[["Close"]].rename(columns={"Close": column_name})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def _build_tlt_proxy_from_fred(
    fred_df: pd.DataFrame,
    column_name: str,
) -> pd.DataFrame:
    """Create a synthetic TLT close proxy from long-rate changes."""
    if "DGS20" in fred_df.columns and fred_df["DGS20"].notna().any():
        yield_series = fred_df["DGS20"].copy()
    elif "DGS30" in fred_df.columns and fred_df["DGS30"].notna().any():
        yield_series = fred_df["DGS30"].copy()
    else:
        return pd.DataFrame()

    # Approximate long-duration bond price moves from yield changes.
    duration = 17.0
    dy = yield_series.diff() / 100.0
    daily_ret = (-duration * dy).fillna(0.0).clip(-0.2, 0.2)
    proxy = (1.0 + daily_ret).cumprod() * 100.0
    out = proxy.to_frame(name=column_name).dropna()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def _extract_fred_vix_close(
    fred_df: pd.DataFrame,
    column_name: str,
) -> pd.DataFrame:
    """Use FRED VIXCLS as VIX close fallback when market data is blocked."""
    if "VIXCLS" not in fred_df.columns:
        return pd.DataFrame()
    series = pd.to_numeric(fred_df["VIXCLS"], errors="coerce").dropna()
    if series.empty:
        return pd.DataFrame()
    out = series.to_frame(name=column_name)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def download_yfinance_close(
    symbol: str,
    column_name: str,
    raw_dir: Path,
    start_date: str,
    force_refresh: bool = False,
    max_retries: int = 5,
    retry_base_seconds: float = 2.0,
    retry_backoff: float = 2.0,
    stooq_enabled: bool = False,
    stooq_symbol: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download and cache a single yfinance close series."""
    cache_path = raw_dir / f"{column_name}.parquet"
    cached = None if force_refresh else _read_parquet_if_exists(cache_path)
    if cached is not None:
        cached = _coerce_cached_close_frame(cached, column_name=column_name)

    if cached is not None and not cached.empty:
        last_dt = cached.index.max()
        pull_start = (last_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    else:
        pull_start = start_date

    end_date = (_today_utc_date() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    downloaded, download_meta = _download_yahoo_with_retries(
        symbol=symbol,
        start_date=pull_start,
        end_date=end_date,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_backoff=retry_backoff,
    )

    if downloaded.empty:
        stooq_error: str | None = None
        if stooq_enabled and stooq_symbol:
            try:
                downloaded = _download_stooq_close(
                    stooq_symbol=stooq_symbol,
                    column_name=column_name,
                    start_date=pull_start,
                    end_date=end_date,
                )
                if not downloaded.empty:
                    close = downloaded
                    if cached is not None:
                        merged = pd.concat([cached, close], axis=0)
                        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                    else:
                        merged = close.sort_index()
                    _save_parquet(merged, cache_path)
                    download_meta["strategy"] = "stooq"
                    download_meta["stooq_symbol"] = stooq_symbol
                    return merged, {
                        "source": "download+cache",
                        "symbol": symbol,
                        "path": str(cache_path),
                        "download_meta": download_meta,
                    }
            except Exception as exc:  # noqa: BLE001
                stooq_error = str(exc)

        download_meta["strategy"] = "failed"
        if stooq_error:
            download_meta["stooq_error"] = stooq_error

    if downloaded.empty:
        if cached is not None:
            return cached, {
                "source": "cache",
                "symbol": symbol,
                "download_meta": download_meta,
            }
        raise RuntimeError(
            f"No data returned from yfinance for {symbol} after retries/fallback. "
            f"Details: {download_meta}"
        )

    try:
        close = _extract_close_frame(
            downloaded=downloaded, symbol=symbol, column_name=column_name
        )
    except Exception as exc:  # noqa: BLE001
        parse_error = str(exc)
        stooq_error: str | None = None
        if stooq_enabled and stooq_symbol:
            try:
                close = _download_stooq_close(
                    stooq_symbol=stooq_symbol,
                    column_name=column_name,
                    start_date=pull_start,
                    end_date=end_date,
                )
                if not close.empty:
                    if cached is not None:
                        merged = pd.concat([cached, close], axis=0)
                        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                    else:
                        merged = close.sort_index()
                    _save_parquet(merged, cache_path)
                    download_meta["strategy"] = "stooq_after_parse_error"
                    download_meta["parse_error"] = parse_error
                    download_meta["stooq_symbol"] = stooq_symbol
                    return merged, {
                        "source": "download+cache",
                        "symbol": symbol,
                        "path": str(cache_path),
                        "download_meta": download_meta,
                    }
            except Exception as fallback_exc:  # noqa: BLE001
                stooq_error = str(fallback_exc)

        if cached is not None:
            download_meta["strategy"] = "cache_after_parse_error"
            download_meta["parse_error"] = parse_error
            if stooq_error:
                download_meta["stooq_error"] = stooq_error
            return cached, {
                "source": "cache",
                "symbol": symbol,
                "download_meta": download_meta,
            }
        raise RuntimeError(
            f"Unable to parse yfinance output for {symbol}. Details: {parse_error}. "
            f"Stooq error: {stooq_error}"
        ) from exc

    if cached is not None:
        merged = pd.concat([cached, close], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = close.sort_index()

    _save_parquet(merged, cache_path)
    return merged, {
        "source": "download+cache",
        "symbol": symbol,
        "path": str(cache_path),
        "download_meta": download_meta,
    }


def download_fred_series(
    series_map: dict[str, str | None],
    raw_dir: Path,
    start_date: str,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download and cache FRED series; handles missing keys/series gracefully."""
    cache_path = raw_dir / "fred_series.parquet"
    cached = None if force_refresh else _read_parquet_if_exists(cache_path)
    metadata: dict[str, Any] = {"path": str(cache_path), "series": {}, "warnings": []}

    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        if cached is not None:
            metadata["warnings"].append("FRED_API_KEY missing. Using cached FRED data.")
            return cached, metadata
        metadata["warnings"].append("FRED_API_KEY missing. FRED columns set to NaN.")
        idx = pd.DatetimeIndex([])
        empty = pd.DataFrame(index=idx)
        for alias in series_map.keys():
            empty[alias] = pd.Series(dtype=float)
        _save_parquet(empty, cache_path)
        return empty, metadata

    fred = Fred(api_key=fred_key)
    result = pd.DataFrame()
    for alias, series_id in series_map.items():
        if not series_id:
            metadata["warnings"].append(f"{alias} placeholder is empty; filling with NaN.")
            result[alias] = pd.Series(dtype=float)
            continue

        try:
            values = fred.get_series(series_id, observation_start=start_date)
            series = pd.Series(values, name=alias)
            series.index = pd.to_datetime(series.index).tz_localize(None)
            result = result.join(series, how="outer") if not result.empty else series.to_frame()
            metadata["series"][alias] = {"series_id": series_id, "status": "ok"}
        except Exception as exc:  # noqa: BLE001
            metadata["series"][alias] = {
                "series_id": series_id,
                "status": "failed",
                "error": str(exc),
            }
            metadata["warnings"].append(f"Failed FRED series {alias} ({series_id}).")
            result[alias] = pd.Series(dtype=float)

    result = result.sort_index()
    if cached is not None and not cached.empty:
        merged = pd.concat([cached, result], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = result

    _save_parquet(merged, cache_path)
    return merged, metadata


def load_raw_data(
    cfg: dict[str, Any],
    raw_dir: Path,
    force_refresh: bool = False,
) -> DownloadResult:
    """Download or update raw data from all sources and return merged frame."""
    start_date = cfg["data"]["start_date"]
    tlt_symbol = cfg["data"]["yfinance"]["tlt_symbol"]
    vix_symbol = cfg["data"]["yfinance"]["vix_symbol"]
    spy_symbol = cfg["data"]["yfinance"].get("spy_symbol", "SPY")
    dxy_symbol = cfg["data"]["yfinance"].get("dxy_symbol", "DX-Y.NYB")
    yf_cfg = cfg["data"].get("yfinance", {})
    max_retries = int(yf_cfg.get("max_retries", 5))
    retry_base_seconds = float(yf_cfg.get("retry_base_seconds", 2.0))
    retry_backoff = float(yf_cfg.get("retry_backoff", 2.0))
    request_pause_seconds = float(yf_cfg.get("request_pause_seconds", 0.5))
    stooq_cfg = cfg["data"].get("stooq", {})
    stooq_enabled = bool(stooq_cfg.get("enabled", False))

    required_fred = cfg["data"]["fred"]["required_series"]
    optional_global = cfg["data"]["fred"]["optional_global_10y_series"]
    fred_series_map = {**required_fred, **optional_global}

    fred_df, fred_meta = download_fred_series(
        series_map=fred_series_map,
        raw_dir=raw_dir,
        start_date=start_date,
        force_refresh=force_refresh,
    )
    tlt_df: pd.DataFrame
    tlt_meta: dict[str, Any]
    try:
        tlt_df, tlt_meta = download_yfinance_close(
            symbol=tlt_symbol,
            column_name="tlt_close",
            raw_dir=raw_dir,
            start_date=start_date,
            force_refresh=force_refresh,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_backoff=retry_backoff,
            stooq_enabled=stooq_enabled,
            stooq_symbol=stooq_cfg.get("tlt_symbol"),
        )
    except Exception as exc:  # noqa: BLE001
        tlt_proxy = _build_tlt_proxy_from_fred(fred_df=fred_df, column_name="tlt_close")
        if tlt_proxy.empty:
            raise
        tlt_df = tlt_proxy
        tlt_meta = {
            "source": "fred_proxy",
            "symbol": tlt_symbol,
            "warning": f"Used FRED-based TLT proxy due to market-source failure: {exc}",
        }

    if request_pause_seconds > 0:
        time.sleep(request_pause_seconds)

    vix_df: pd.DataFrame
    vix_meta: dict[str, Any]
    try:
        vix_df, vix_meta = download_yfinance_close(
            symbol=vix_symbol,
            column_name="vix_close",
            raw_dir=raw_dir,
            start_date=start_date,
            force_refresh=force_refresh,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_backoff=retry_backoff,
            stooq_enabled=stooq_enabled,
            stooq_symbol=stooq_cfg.get("vix_symbol"),
        )
    except Exception as exc:  # noqa: BLE001
        vix_fred = _extract_fred_vix_close(fred_df=fred_df, column_name="vix_close")
        if vix_fred.empty:
            raise
        vix_df = vix_fred
        vix_meta = {
            "source": "fred_vixcls",
            "symbol": vix_symbol,
            "warning": f"Used FRED VIXCLS due to market-source failure: {exc}",
        }

    if request_pause_seconds > 0:
        time.sleep(request_pause_seconds)

    spy_df: pd.DataFrame
    spy_meta: dict[str, Any]
    try:
        spy_df, spy_meta = download_yfinance_close(
            symbol=spy_symbol,
            column_name="spy_close",
            raw_dir=raw_dir,
            start_date=start_date,
            force_refresh=force_refresh,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_backoff=retry_backoff,
            stooq_enabled=stooq_enabled,
            stooq_symbol=stooq_cfg.get("spy_symbol"),
        )
    except Exception as exc:  # noqa: BLE001
        # SPY is critical for correlation, but we can survive without it (correlation=0)
        spy_df = pd.DataFrame()
        spy_meta = {
            "source": "failed",
            "symbol": spy_symbol,
            "warning": f"Failed to download SPY: {exc}",
        }

    if request_pause_seconds > 0:
        time.sleep(request_pause_seconds)

    dxy_df: pd.DataFrame
    dxy_meta: dict[str, Any]
    try:
        dxy_df, dxy_meta = download_yfinance_close(
            symbol=dxy_symbol,
            column_name="dxy_close",
            raw_dir=raw_dir,
            start_date=start_date,
            force_refresh=force_refresh,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_backoff=retry_backoff,
            stooq_enabled=stooq_enabled,
            stooq_symbol=stooq_cfg.get("dxy_symbol"),
        )
    except Exception as exc:  # noqa: BLE001
        # Fallback to FRED DTWEXBGS (Trade Weighted U.S. Dollar Index) if market data fails
        try:
            fred_key = os.getenv("FRED_API_KEY")
            if fred_key:
                fred = Fred(api_key=fred_key)
                fred_dxy = fred.get_series("DTWEXBGS", observation_start=start_date)
                if not fred_dxy.empty:
                    dxy_df = fred_dxy.to_frame(name="dxy_close")
                    dxy_df.index = pd.to_datetime(dxy_df.index).tz_localize(None)
                    dxy_meta = {
                        "source": "fred_dtwexbgs",
                        "symbol": "DTWEXBGS",
                        "warning": f"Used FRED Trade Weighted Dollar Index due to market-source failure: {exc}",
                    }
                else:
                    raise RuntimeError("FRED DXY fallback empty")
            else:
                raise RuntimeError("No FRED key for fallback")
        except Exception as fred_exc:
            dxy_df = pd.DataFrame()
            dxy_meta = {
                "source": "failed",
                "symbol": dxy_symbol,
                "warning": f"Failed to download DXY (and FRED fallback): {exc} / {fred_exc}",
            }

    tlt_df = _ensure_single_level_columns(tlt_df)
    vix_df = _ensure_single_level_columns(vix_df)
    spy_df = _ensure_single_level_columns(spy_df)
    dxy_df = _ensure_single_level_columns(dxy_df)
    fred_df = _ensure_single_level_columns(fred_df)
    merged = (
        tlt_df.join(vix_df, how="outer")
        .join(spy_df, how="outer")
        .join(dxy_df, how="outer")
        .join(fred_df, how="outer")
        .sort_index()
    )
    merged.index.name = "date"
    _save_parquet(merged, raw_dir / "merged_raw.parquet")

    metadata = {
        "tlt": tlt_meta,
        "vix": vix_meta,
        "spy": spy_meta,
        "dxy": dxy_meta,
        "fred": fred_meta,
        "last_timestamp": str(merged.index.max()) if not merged.empty else None,
        "rows": int(merged.shape[0]),
    }
    return DownloadResult(raw_frame=merged, metadata=metadata)
