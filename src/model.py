"""HMM model training and probability calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import TimeSeriesSplit
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


def _build_hmm(
    model_cfg: dict[str, Any],
    n_regimes: int,
) -> GaussianHMM:
    """Create a configured GaussianHMM instance."""
    return GaussianHMM(
        n_components=n_regimes,
        covariance_type=model_cfg.get("covariance_type", "full"),
        n_iter=int(model_cfg.get("n_iter", 500)),
        random_state=int(model_cfg.get("random_state", 42)),
    )


def _fit_scaler_hmm_and_posterior(
    frame: pd.DataFrame,
    feature_cols: list[str],
    model_cfg: dict[str, Any],
    n_regimes: int,
) -> tuple[StandardScaler, GaussianHMM, pd.DataFrame]:
    """Fit scaler/HMM on frame and return posterior probabilities."""
    scaler = StandardScaler()
    x_values = scaler.fit_transform(frame[feature_cols].values)
    hmm = _build_hmm(model_cfg=model_cfg, n_regimes=n_regimes)
    hmm.fit(x_values)
    post = hmm.predict_proba(x_values)
    posterior_cols = [f"regime_{i}" for i in range(n_regimes)]
    posterior_df = pd.DataFrame(post, index=frame.index, columns=posterior_cols)
    return scaler, hmm, posterior_df


def _apply_smoothing(
    posterior_df: pd.DataFrame,
    model_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Apply optional posterior smoothing and re-normalize probabilities."""
    smooth_cfg = model_cfg.get("smoothing", {})
    if bool(smooth_cfg.get("enabled", False)):
        span = int(smooth_cfg.get("span", 3))
        smoothed = posterior_df.ewm(span=span, adjust=False).mean()
        return smoothed.div(smoothed.sum(axis=1), axis=0)
    return posterior_df


def _compute_regime_up_table(
    posterior_df: pd.DataFrame,
    tlt_up: np.ndarray,
    vix_up: np.ndarray,
) -> pd.DataFrame:
    """Estimate regime-conditional up frequencies."""
    regime_rows: list[dict[str, float]] = []
    for i, col in enumerate(posterior_df.columns):
        weights = posterior_df[col].values
        regime_rows.append(
            {
                "regime": i,
                "tlt_up_freq": _weighted_up_frequency(weights, tlt_up),
                "vix_up_freq": _weighted_up_frequency(weights, vix_up),
            }
        )
    return pd.DataFrame(regime_rows).set_index("regime")


def _attach_directional_probabilities(
    score_frame: pd.DataFrame,
    posterior_df: pd.DataFrame,
    regime_table: pd.DataFrame,
) -> pd.DataFrame:
    """Attach up/down directional probabilities to score frame."""
    out = score_frame.copy()
    tlt_up_probs = posterior_df.values @ regime_table["tlt_up_freq"].values
    vix_up_probs = posterior_df.values @ regime_table["vix_up_freq"].values
    tlt_up_probs = np.clip(tlt_up_probs, 0.0, 1.0)
    vix_up_probs = np.clip(vix_up_probs, 0.0, 1.0)

    out["p_tlt_up_h5"] = tlt_up_probs
    out["p_tlt_down_h5"] = 1.0 - tlt_up_probs
    out["p_vix_up_h5"] = vix_up_probs
    out["p_vix_down_h5"] = 1.0 - vix_up_probs
    out["most_likely_regime"] = posterior_df.idxmax(axis=1)
    return out


def _brier_score(prob: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prob - target) ** 2))


def _log_loss(prob: np.ndarray, target: np.ndarray) -> float:
    clipped = np.clip(prob, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)))


def _confidence_bucket_stats(prob: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Compute hit-rate/coverage for stronger-confidence subsets."""
    out: dict[str, Any] = {}
    for threshold in [0.55, 0.60, 0.65]:
        mask = np.maximum(prob, 1.0 - prob) >= threshold
        key = f"ge_{threshold:.2f}"
        if not np.any(mask):
            out[key] = {"coverage": 0.0, "rows": 0, "hit_rate": None}
            continue
        prob_sel = prob[mask]
        target_sel = target[mask]
        hit = np.mean((prob_sel >= 0.5) == (target_sel == 1.0))
        out[key] = {
            "coverage": float(np.mean(mask)),
            "rows": int(np.sum(mask)),
            "hit_rate": float(hit),
        }
    return out


def _run_time_series_backtest(
    labeled_df: pd.DataFrame,
    feature_cols: list[str],
    model_cfg: dict[str, Any],
    n_regimes: int,
) -> dict[str, Any]:
    """Run expanding-window time series backtest and report OOS metrics."""
    n_splits = int(model_cfg.get("backtest_splits", 5))
    min_train_rows = int(model_cfg.get("backtest_min_train_rows", 500))
    if labeled_df.shape[0] < max(min_train_rows + 50, n_splits + 20):
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "insufficient_rows",
            "rows": int(labeled_df.shape[0]),
        }

    tss = TimeSeriesSplit(n_splits=n_splits)
    tlt_probs_all: list[float] = []
    tlt_actual_all: list[float] = []
    vix_probs_all: list[float] = []
    vix_actual_all: list[float] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold_id, (train_idx, test_idx) in enumerate(tss.split(labeled_df), start=1):
        if len(train_idx) < min_train_rows or len(test_idx) == 0:
            continue

        train_df = labeled_df.iloc[train_idx].copy()
        test_df = labeled_df.iloc[test_idx].copy()

        scaler, hmm, train_post = _fit_scaler_hmm_and_posterior(
            frame=train_df,
            feature_cols=feature_cols,
            model_cfg=model_cfg,
            n_regimes=n_regimes,
        )
        train_post = _apply_smoothing(train_post, model_cfg=model_cfg)

        tlt_up_train = (train_df["tlt_forward_h5_return"].values > 0).astype(float)
        vix_up_train = (train_df["vix_forward_h5_return"].values > 0).astype(float)
        regime_table = _compute_regime_up_table(train_post, tlt_up_train, vix_up_train)

        x_test = scaler.transform(test_df[feature_cols].values)
        test_post_arr = hmm.predict_proba(x_test)
        posterior_cols = [f"regime_{i}" for i in range(n_regimes)]
        test_post = pd.DataFrame(test_post_arr, index=test_df.index, columns=posterior_cols)
        test_post = _apply_smoothing(test_post, model_cfg=model_cfg)

        tlt_probs = np.clip(test_post.values @ regime_table["tlt_up_freq"].values, 0.0, 1.0)
        vix_probs = np.clip(test_post.values @ regime_table["vix_up_freq"].values, 0.0, 1.0)
        tlt_actual = (test_df["tlt_forward_h5_return"].values > 0).astype(float)
        vix_actual = (test_df["vix_forward_h5_return"].values > 0).astype(float)

        tlt_probs_all.extend(tlt_probs.tolist())
        tlt_actual_all.extend(tlt_actual.tolist())
        vix_probs_all.extend(vix_probs.tolist())
        vix_actual_all.extend(vix_actual.tolist())

        fold_summaries.append(
            {
                "fold": fold_id,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "test_start": test_df.index.min().strftime("%Y-%m-%d"),
                "test_end": test_df.index.max().strftime("%Y-%m-%d"),
            }
        )

    if not tlt_probs_all or not vix_probs_all:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "no_valid_folds",
            "rows": int(labeled_df.shape[0]),
        }

    tlt_prob_arr = np.asarray(tlt_probs_all, dtype=float)
    tlt_act_arr = np.asarray(tlt_actual_all, dtype=float)
    vix_prob_arr = np.asarray(vix_probs_all, dtype=float)
    vix_act_arr = np.asarray(vix_actual_all, dtype=float)

    return {
        "enabled": True,
        "status": "ok",
        "n_splits": n_splits,
        "min_train_rows": min_train_rows,
        "oos_rows": int(len(tlt_prob_arr)),
        "tlt": {
            "hit_rate": float(np.mean((tlt_prob_arr >= 0.5) == (tlt_act_arr == 1.0))),
            "brier_score": _brier_score(tlt_prob_arr, tlt_act_arr),
            "log_loss": _log_loss(tlt_prob_arr, tlt_act_arr),
            "confidence_buckets": _confidence_bucket_stats(tlt_prob_arr, tlt_act_arr),
        },
        "vix": {
            "hit_rate": float(np.mean((vix_prob_arr >= 0.5) == (vix_act_arr == 1.0))),
            "brier_score": _brier_score(vix_prob_arr, vix_act_arr),
            "log_loss": _log_loss(vix_prob_arr, vix_act_arr),
            "confidence_buckets": _confidence_bucket_stats(vix_prob_arr, vix_act_arr),
        },
        "folds": fold_summaries,
    }


def run_hmm_direction_model(feature_df: pd.DataFrame, cfg: dict[str, Any]) -> HMMOutputs:
    """Fit GaussianHMM and compute H5 up/down directional probabilities."""
    feature_cols = cfg["features"]["feature_columns"]
    model_cfg = cfg["model"]
    n_regimes = int(model_cfg["n_regimes"])
    horizon = int(cfg["features"]["horizon_days"])

    target_cols = ["tlt_forward_h5_return", "vix_forward_h5_return"]
    train_required = feature_cols + target_cols + ["tlt_close", "vix_close"]
    score_required = feature_cols + ["tlt_close", "vix_close"]
    train_frame = feature_df.dropna(subset=train_required).copy()
    score_frame = feature_df.dropna(subset=score_required).copy()

    if train_frame.empty:
        raise RuntimeError("Not enough non-null rows after feature/target filtering.")
    if score_frame.empty:
        raise RuntimeError("Not enough non-null rows for live scoring.")

    scaler, hmm, train_posterior = _fit_scaler_hmm_and_posterior(
        frame=train_frame,
        feature_cols=feature_cols,
        model_cfg=model_cfg,
        n_regimes=n_regimes,
    )
    train_posterior = _apply_smoothing(train_posterior, model_cfg=model_cfg)

    x_score = scaler.transform(score_frame[feature_cols].values)
    score_post_arr = hmm.predict_proba(x_score)
    posterior_cols = [f"regime_{i}" for i in range(n_regimes)]
    posterior_df = pd.DataFrame(score_post_arr, index=score_frame.index, columns=posterior_cols)
    posterior_df = _apply_smoothing(posterior_df, model_cfg=model_cfg)

    tlt_up_train = (train_frame["tlt_forward_h5_return"].values > 0).astype(float)
    vix_up_train = (train_frame["vix_forward_h5_return"].values > 0).astype(float)
    regime_table = _compute_regime_up_table(train_posterior, tlt_up_train, vix_up_train)
    model_frame = _attach_directional_probabilities(
        score_frame=score_frame,
        posterior_df=posterior_df,
        regime_table=regime_table,
    )

    latest_idx = model_frame.index.max()
    latest_posterior = posterior_df.loc[latest_idx]
    latest_has_label = pd.notna(model_frame.loc[latest_idx, "tlt_forward_h5_return"]) and pd.notna(
        model_frame.loc[latest_idx, "vix_forward_h5_return"]
    )
    backtest = _run_time_series_backtest(
        labeled_df=train_frame,
        feature_cols=feature_cols,
        model_cfg=model_cfg,
        n_regimes=n_regimes,
    )
    latest = {
        "date": latest_idx.strftime("%Y-%m-%d"),
        "horizon_days": horizon,
        "train_window_end": train_frame.index.max().strftime("%Y-%m-%d"),
        "latest_row_has_h5_label": bool(latest_has_label),
        "train_rows": int(train_frame.shape[0]),
        "score_rows": int(score_frame.shape[0]),
        "probabilities": {
            "tlt_up_h5": float(model_frame.loc[latest_idx, "p_tlt_up_h5"]),
            "tlt_down_h5": float(model_frame.loc[latest_idx, "p_tlt_down_h5"]),
            "vix_up_h5": float(model_frame.loc[latest_idx, "p_vix_up_h5"]),
            "vix_down_h5": float(model_frame.loc[latest_idx, "p_vix_down_h5"]),
        },
        "regime_probabilities": {col: float(latest_posterior[col]) for col in posterior_cols},
        "most_likely_regime": str(model_frame.loc[latest_idx, "most_likely_regime"]),
        "market_context": {
            "tlt_close": float(model_frame.loc[latest_idx, "tlt_close"]),
            "vix_close": float(model_frame.loc[latest_idx, "vix_close"]),
            "tlt_20d_min_close": float(model_frame.loc[latest_idx, "tlt_20d_min_close"]),
            "tlt_20d_max_close": float(model_frame.loc[latest_idx, "tlt_20d_max_close"]),
            "vix_20d_min_close": float(model_frame.loc[latest_idx, "vix_20d_min_close"]),
            "vix_20d_max_close": float(model_frame.loc[latest_idx, "vix_20d_max_close"]),
            "tlt_trend_gap_20": float(model_frame.loc[latest_idx, "tlt_trend_gap_20"]),
            "vix_trend_gap_20": float(model_frame.loc[latest_idx, "vix_trend_gap_20"]),
        },
        "backtest": backtest,
    }

    return HMMOutputs(
        model_frame=model_frame,
        posterior_probs=posterior_df,
        regime_up_table=regime_table,
        latest=latest,
    )
