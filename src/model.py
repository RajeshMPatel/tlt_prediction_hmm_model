"""HMM model training and probability calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


@dataclass
class HMMOutputs:
    """Container for model outputs used by reporting and web app."""

    model_frame: pd.DataFrame
    posterior_probs: pd.DataFrame
    regime_up_table: pd.DataFrame
    latest: dict[str, Any]


def _weighted_up_frequency(weights: np.ndarray, target: np.ndarray) -> float:
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return float("nan")
    return float(np.sum(weights * target) / total_weight)


def run_hmm_direction_model(feature_df: pd.DataFrame, cfg: dict[str, Any]) -> HMMOutputs:
    """Fit GaussianHMM and compute H5 up/down directional probabilities."""
    feature_cols = cfg["features"]["feature_columns"]
    model_cfg = cfg["model"]
    n_regimes = int(model_cfg["n_regimes"])
    horizon = int(cfg["features"]["horizon_days"])

    target_cols = ["tlt_forward_h5_return", "vix_forward_h5_return"]
    required_cols = feature_cols + target_cols + ["tlt_close", "vix_close"]
    model_frame = feature_df.dropna(subset=required_cols).copy()
    if model_frame.empty:
        raise RuntimeError("Not enough non-null rows after feature/target filtering.")

    scaler = StandardScaler()
    x_values = scaler.fit_transform(model_frame[feature_cols].values)

    hmm = GaussianHMM(
        n_components=n_regimes,
        covariance_type=model_cfg.get("covariance_type", "full"),
        n_iter=int(model_cfg.get("n_iter", 500)),
        random_state=int(model_cfg.get("random_state", 42)),
    )
    hmm.fit(x_values)
    post = hmm.predict_proba(x_values)

    posterior_cols = [f"regime_{i}" for i in range(n_regimes)]
    posterior_df = pd.DataFrame(post, index=model_frame.index, columns=posterior_cols)

    smooth_cfg = model_cfg.get("smoothing", {})
    if bool(smooth_cfg.get("enabled", False)):
        span = int(smooth_cfg.get("span", 3))
        posterior_df = posterior_df.ewm(span=span, adjust=False).mean()
        posterior_df = posterior_df.div(posterior_df.sum(axis=1), axis=0)

    tlt_up = (model_frame["tlt_forward_h5_return"].values > 0).astype(float)
    vix_up = (model_frame["vix_forward_h5_return"].values > 0).astype(float)

    regime_rows: list[dict[str, float]] = []
    for i, col in enumerate(posterior_cols):
        weights = posterior_df[col].values
        regime_rows.append(
            {
                "regime": i,
                "tlt_up_freq": _weighted_up_frequency(weights, tlt_up),
                "vix_up_freq": _weighted_up_frequency(weights, vix_up),
            }
        )
    regime_table = pd.DataFrame(regime_rows).set_index("regime")

    tlt_up_probs = posterior_df.values @ regime_table["tlt_up_freq"].values
    vix_up_probs = posterior_df.values @ regime_table["vix_up_freq"].values
    tlt_up_probs = np.clip(tlt_up_probs, 0.0, 1.0)
    vix_up_probs = np.clip(vix_up_probs, 0.0, 1.0)

    model_frame["p_tlt_up_h5"] = tlt_up_probs
    model_frame["p_tlt_down_h5"] = 1.0 - tlt_up_probs
    model_frame["p_vix_up_h5"] = vix_up_probs
    model_frame["p_vix_down_h5"] = 1.0 - vix_up_probs
    model_frame["most_likely_regime"] = posterior_df.idxmax(axis=1)

    latest_idx = model_frame.index.max()
    latest_posterior = posterior_df.loc[latest_idx]
    latest = {
        "date": latest_idx.strftime("%Y-%m-%d"),
        "horizon_days": horizon,
        "probabilities": {
            "tlt_up_h5": float(model_frame.loc[latest_idx, "p_tlt_up_h5"]),
            "tlt_down_h5": float(model_frame.loc[latest_idx, "p_tlt_down_h5"]),
            "vix_up_h5": float(model_frame.loc[latest_idx, "p_vix_up_h5"]),
            "vix_down_h5": float(model_frame.loc[latest_idx, "p_vix_down_h5"]),
        },
        "regime_probabilities": {col: float(latest_posterior[col]) for col in posterior_cols},
        "most_likely_regime": str(model_frame.loc[latest_idx, "most_likely_regime"]),
    }

    return HMMOutputs(
        model_frame=model_frame,
        posterior_probs=posterior_df,
        regime_up_table=regime_table,
        latest=latest,
    )
