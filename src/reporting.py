"""Reporting helpers for charts and daily JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _contiguous_regions(values: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    regions: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if values.empty:
        return regions
    start_idx = values.index[0]
    current = values.iloc[0]
    for idx, val in values.iloc[1:].items():
        if val != current:
            regions.append((start_idx, idx, str(current)))
            start_idx = idx
            current = val
    regions.append((start_idx, values.index[-1], str(current)))
    return regions


def _shade_regimes(ax: Any, regime_series: pd.Series, n_regimes: int) -> None:
    cmap = plt.cm.get_cmap("tab10", n_regimes)
    for start, end, regime in _contiguous_regions(regime_series):
        try:
            regime_id = int(regime.split("_")[-1])
        except ValueError:
            regime_id = 0
        ax.axvspan(start, end, color=cmap(regime_id), alpha=0.14, linewidth=0)


def save_regime_charts(
    model_frame: pd.DataFrame,
    posterior_df: pd.DataFrame,
    charts_dir: Path,
    lookback_days: int,
) -> dict[str, str]:
    """Save TLT/VIX charts with background regime shading."""
    charts_dir.mkdir(parents=True, exist_ok=True)

    joint = model_frame.join(posterior_df, how="left")
    joint = joint.tail(lookback_days)
    regime_series = joint["most_likely_regime"]
    n_regimes = posterior_df.shape[1]

    outputs: dict[str, str] = {}

    tlt_path = charts_dir / "tlt_regime_chart.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_regimes(ax, regime_series, n_regimes=n_regimes)
    ax.plot(joint.index, joint["tlt_close"], color="navy", linewidth=1.6, label="TLT close")
    ax.set_title("TLT with HMM Regime Shading")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(tlt_path, dpi=140)
    plt.close(fig)
    outputs["tlt_chart"] = str(tlt_path)

    vix_path = charts_dir / "vix_regime_chart.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_regimes(ax, regime_series, n_regimes=n_regimes)
    ax.plot(joint.index, joint["vix_close"], color="darkred", linewidth=1.6, label="VIX close")
    ax.set_title("VIX with HMM Regime Shading")
    ax.set_ylabel("Index")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(vix_path, dpi=140)
    plt.close(fig)
    outputs["vix_chart"] = str(vix_path)

    probs_path = charts_dir / "regime_probabilities_chart.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in posterior_df.columns:
        ax.plot(joint.index, joint[col], linewidth=1.1, label=col)
    ax.set_title("Posterior Regime Probabilities")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(probs_path, dpi=140)
    plt.close(fig)
    outputs["posterior_chart"] = str(probs_path)

    return outputs


def write_daily_outputs(
    latest_payload: dict[str, Any],
    daily_reports_dir: Path,
    pipeline_meta: dict[str, Any],
    chart_paths: dict[str, str],
) -> None:
    """Write latest probabilities and status/debug JSON files."""
    generated_at = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    latest = {
        **latest_payload,
        "generated_at_utc": generated_at,
        "last_data_update": pipeline_meta.get("last_data_timestamp"),
        "charts": chart_paths,
    }
    _json_dump(daily_reports_dir / "latest_probabilities.json", latest)

    status = {
        "generated_at_utc": generated_at,
        "last_data_timestamp": pipeline_meta.get("last_data_timestamp"),
        "raw_rows": pipeline_meta.get("raw_rows"),
        "model_rows": pipeline_meta.get("model_rows"),
        "actionable_signals": latest_payload.get("actionable_signals", {}),
        "warnings": pipeline_meta.get("warnings", []),
        "sources": pipeline_meta.get("sources", {}),
        "debug": pipeline_meta.get("debug", {}),
    }
    _json_dump(daily_reports_dir / "status.json", status)
