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


def _weighted_mean(weights: np.ndarray, values: np.ndarray) -> float:
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return float("nan")
    mask = np.isfinite(values)
    if not np.any(mask):
        return float("nan")
    w = weights[mask]
    v = values[mask]
    denom = float(np.sum(w))
    if denom <= 0:
        return float("nan")
    return float(np.sum(w * v) / denom)


def _build_regime_labels(
    train_frame: pd.DataFrame,
    train_posterior: pd.DataFrame,
    regime_table: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Create human-readable regime names from regime behavior."""
    profiles: list[dict[str, Any]] = []
    has_vol = "realized_vol_20d" in train_frame.columns
    vol_values = (
        train_frame["realized_vol_20d"].values
        if has_vol
        else np.full(train_frame.shape[0], np.nan, dtype=float)
    )

    for idx, col in enumerate(train_posterior.columns):
        weights = train_posterior[col].values
        tlt_up = float(regime_table.loc[idx, "tlt_up_freq"])
        vix_up = float(regime_table.loc[idx, "vix_up_freq"])
        avg_vol = _weighted_mean(weights, vol_values)
        profiles.append(
            {
                "key": col,
                "regime_idx": idx,
                "tlt_up_freq": tlt_up,
                "vix_up_freq": vix_up,
                "avg_realized_vol_20d": avg_vol,
            }
        )

    vol_arr = np.asarray([p["avg_realized_vol_20d"] for p in profiles], dtype=float)
    finite_mask = np.isfinite(vol_arr)
    high_vol_cut = float(np.nanpercentile(vol_arr, 67)) if np.any(finite_mask) else float("nan")
    low_vol_cut = float(np.nanpercentile(vol_arr, 33)) if np.any(finite_mask) else float("nan")

    label_map: dict[str, str] = {}
    profile_out: dict[str, dict[str, Any]] = {}
    for p in profiles:
        tlt_up = p["tlt_up_freq"]
        vix_up = p["vix_up_freq"]
        avg_vol = p["avg_realized_vol_20d"]

        if tlt_up >= 0.55 and vix_up <= 0.45:
            base = "Calm Risk-On"
        elif tlt_up <= 0.45 and vix_up >= 0.55:
            base = "Stress Risk-Off"
        elif tlt_up <= 0.45 and vix_up <= 0.45:
            base = "Growth Risk-On"
        elif tlt_up >= 0.55 and vix_up >= 0.55:
            base = "Cross-Asset Flight-to-Quality"
        else:
            base = "Mixed Transition"

        if np.isfinite(avg_vol) and np.isfinite(high_vol_cut) and avg_vol >= high_vol_cut:
            base = f"High-Vol {base}"
        elif np.isfinite(avg_vol) and np.isfinite(low_vol_cut) and avg_vol <= low_vol_cut:
            base = f"Low-Vol {base}"

        label = f"{base} [{p['regime_idx'] + 1}]"
        label_map[p["key"]] = label
        profile_out[label] = {
            "technical_key": p["key"],
            "tlt_up_freq": float(tlt_up),
            "vix_up_freq": float(vix_up),
            "avg_realized_vol_20d": (
                None if not np.isfinite(avg_vol) else float(avg_vol)
            ),
        }

    return label_map, profile_out


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


def _confidence_bucket_stats(
    prob: np.ndarray,
    target: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Compute hit-rate/coverage for stronger-confidence subsets."""
    use_thresholds = thresholds or [0.55, 0.60, 0.65]
    out: dict[str, Any] = {}
    for threshold in use_thresholds:
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


def _recent_confidence_bucket_stats(
    prob: np.ndarray,
    target: np.ndarray,
    thresholds: list[float],
    rolling_rows: int,
) -> dict[str, Any]:
    """Compute confidence-bucket stats on the most recent OOS rows."""
    if rolling_rows <= 0 or rolling_rows >= len(prob):
        recent_prob = prob
        recent_target = target
    else:
        recent_prob = prob[-rolling_rows:]
        recent_target = target[-rolling_rows:]

    return {
        "rows_considered": int(len(recent_prob)),
        "confidence_buckets": _confidence_bucket_stats(
            recent_prob, recent_target, thresholds=thresholds
        ),
    }


def _bucket_key(threshold: float) -> str:
    return f"ge_{threshold:.2f}"


def _pick_signal_from_policy(
    probability_up: float,
    asset_name: str,
    recent_buckets: dict[str, Any],
    decision_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Map probability + backtest bucket quality into actionable signal."""
    neutral_band = float(decision_cfg.get("neutral_band", 0.02))
    min_bucket_rows = int(decision_cfg.get("min_bucket_rows", 120))
    strong_cfg = decision_cfg.get("strong", {})
    moderate_cfg = decision_cfg.get("moderate", {})
    strong_min_hit = float(strong_cfg.get("min_hit_rate", 0.62))
    moderate_min_hit = float(moderate_cfg.get("min_hit_rate", 0.55))
    strong_candidates = [float(x) for x in strong_cfg.get("candidate_thresholds", [0.65, 0.60])]
    moderate_candidates = [
        float(x) for x in moderate_cfg.get("candidate_thresholds", [0.60, 0.55])
    ]

    direction = "UP" if probability_up >= 0.5 else "DOWN"
    confidence = float(max(probability_up, 1.0 - probability_up))
    edge = abs(probability_up - 0.5)
    if edge < neutral_band:
        return {
            "asset": asset_name,
            "action": "SKIP",
            "direction": direction,
            "probability_up": float(probability_up),
            "confidence": confidence,
            "selected_threshold": None,
            "expected_hit_rate": None,
            "coverage": None,
            "reason": "probability_too_close_to_50_50",
        }

    def select_candidate(
        candidates: list[float],
        min_hit_rate: float,
    ) -> tuple[float, dict[str, Any]] | tuple[None, None]:
        for threshold in candidates:
            if confidence < threshold:
                continue
            stats = recent_buckets.get(_bucket_key(threshold), {})
            rows = int(stats.get("rows", 0))
            hit_rate = stats.get("hit_rate")
            if rows < min_bucket_rows or hit_rate is None:
                continue
            if float(hit_rate) < min_hit_rate:
                continue
            return threshold, stats
        return None, None

    strong_threshold, strong_stats = select_candidate(strong_candidates, strong_min_hit)
    if strong_stats is not None:
        return {
            "asset": asset_name,
            "action": f"STRONG_{direction}",
            "direction": direction,
            "probability_up": float(probability_up),
            "confidence": confidence,
            "selected_threshold": float(strong_threshold),
            "expected_hit_rate": float(strong_stats["hit_rate"]),
            "coverage": float(strong_stats["coverage"]),
            "reason": "passes_strong_bucket_rule",
        }

    moderate_threshold, moderate_stats = select_candidate(
        moderate_candidates, moderate_min_hit
    )
    if moderate_stats is not None:
        return {
            "asset": asset_name,
            "action": f"MODERATE_{direction}",
            "direction": direction,
            "probability_up": float(probability_up),
            "confidence": confidence,
            "selected_threshold": float(moderate_threshold),
            "expected_hit_rate": float(moderate_stats["hit_rate"]),
            "coverage": float(moderate_stats["coverage"]),
            "reason": "passes_moderate_bucket_rule",
        }

    return {
        "asset": asset_name,
        "action": "SKIP",
        "direction": direction,
        "probability_up": float(probability_up),
        "confidence": confidence,
        "selected_threshold": None,
        "expected_hit_rate": None,
        "coverage": None,
        "reason": "no_valid_confidence_bucket",
    }


def _build_actionable_signals(
    latest_probs: dict[str, float],
    backtest: dict[str, Any],
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build a plain-English decision layer from backtest-calibrated thresholds."""
    empty_signal = {
        "action": "SKIP",
        "direction": None,
        "probability_up": None,
        "confidence": None,
        "selected_threshold": None,
        "expected_hit_rate": None,
        "coverage": None,
    }
    decision_cfg = model_cfg.get("decision_policy", {})
    if not bool(decision_cfg.get("enabled", True)):
        return {
            "policy_enabled": False,
            "tlt": {**empty_signal, "asset": "TLT", "reason": "policy_disabled"},
            "vix": {**empty_signal, "asset": "VIX", "reason": "policy_disabled"},
        }

    if backtest.get("status") != "ok":
        return {
            "policy_enabled": True,
            "tlt": {**empty_signal, "asset": "TLT", "reason": "backtest_unavailable"},
            "vix": {**empty_signal, "asset": "VIX", "reason": "backtest_unavailable"},
        }

    tlt_recent = backtest.get("tlt", {}).get("recent_confidence_buckets", {}).get(
        "confidence_buckets", {}
    )
    vix_recent = backtest.get("vix", {}).get("recent_confidence_buckets", {}).get(
        "confidence_buckets", {}
    )
    tlt_signal = _pick_signal_from_policy(
        probability_up=latest_probs["tlt_up_h5"],
        asset_name="TLT",
        recent_buckets=tlt_recent,
        decision_cfg=decision_cfg,
    )
    vix_signal = _pick_signal_from_policy(
        probability_up=latest_probs["vix_up_h5"],
        asset_name="VIX",
        recent_buckets=vix_recent,
        decision_cfg=decision_cfg,
    )
    return {
        "policy_enabled": True,
        "rolling_oos_rows": int(decision_cfg.get("rolling_oos_rows", 1000)),
        "min_bucket_rows": int(decision_cfg.get("min_bucket_rows", 120)),
        "tlt": tlt_signal,
        "vix": vix_signal,
    }


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
    decision_cfg = model_cfg.get("decision_policy", {})
    thresholds = [float(x) for x in decision_cfg.get("thresholds", [0.55, 0.60, 0.65])]
    rolling_oos_rows = int(decision_cfg.get("rolling_oos_rows", 1000))

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
            "confidence_buckets": _confidence_bucket_stats(
                tlt_prob_arr, tlt_act_arr, thresholds=thresholds
            ),
            "recent_confidence_buckets": _recent_confidence_bucket_stats(
                tlt_prob_arr,
                tlt_act_arr,
                thresholds=thresholds,
                rolling_rows=rolling_oos_rows,
            ),
        },
        "vix": {
            "hit_rate": float(np.mean((vix_prob_arr >= 0.5) == (vix_act_arr == 1.0))),
            "brier_score": _brier_score(vix_prob_arr, vix_act_arr),
            "log_loss": _log_loss(vix_prob_arr, vix_act_arr),
            "confidence_buckets": _confidence_bucket_stats(
                vix_prob_arr, vix_act_arr, thresholds=thresholds
            ),
            "recent_confidence_buckets": _recent_confidence_bucket_stats(
                vix_prob_arr,
                vix_act_arr,
                thresholds=thresholds,
                rolling_rows=rolling_oos_rows,
            ),
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
    regime_label_map, regime_profiles = _build_regime_labels(
        train_frame=train_frame,
        train_posterior=train_posterior,
        regime_table=regime_table,
    )
    model_frame = _attach_directional_probabilities(
        score_frame=score_frame,
        posterior_df=posterior_df,
        regime_table=regime_table,
    )

    latest_idx = model_frame.index.max()
    latest_posterior = posterior_df.loc[latest_idx]
    technical_regime_probs = {col: float(latest_posterior[col]) for col in posterior_cols}
    friendly_regime_probs = {
        regime_label_map.get(key, key): value
        for key, value in technical_regime_probs.items()
    }
    most_likely_regime_key = str(model_frame.loc[latest_idx, "most_likely_regime"])
    most_likely_regime_label = regime_label_map.get(
        most_likely_regime_key, most_likely_regime_key
    )
    latest_has_label = pd.notna(model_frame.loc[latest_idx, "tlt_forward_h5_return"]) and pd.notna(
        model_frame.loc[latest_idx, "vix_forward_h5_return"]
    )
    latest_probabilities = {
        "tlt_up_h5": float(model_frame.loc[latest_idx, "p_tlt_up_h5"]),
        "tlt_down_h5": float(model_frame.loc[latest_idx, "p_tlt_down_h5"]),
        "vix_up_h5": float(model_frame.loc[latest_idx, "p_vix_up_h5"]),
        "vix_down_h5": float(model_frame.loc[latest_idx, "p_vix_down_h5"]),
    }
    backtest = _run_time_series_backtest(
        labeled_df=train_frame,
        feature_cols=feature_cols,
        model_cfg=model_cfg,
        n_regimes=n_regimes,
    )
    actionable_signals = _build_actionable_signals(
        latest_probs=latest_probabilities,
        backtest=backtest,
        model_cfg=model_cfg,
    )
    latest = {
        "date": latest_idx.strftime("%Y-%m-%d"),
        "horizon_days": horizon,
        "train_window_end": train_frame.index.max().strftime("%Y-%m-%d"),
        "latest_row_has_h5_label": bool(latest_has_label),
        "train_rows": int(train_frame.shape[0]),
        "score_rows": int(score_frame.shape[0]),
        "probabilities": latest_probabilities,
        "regime_probabilities": friendly_regime_probs,
        "regime_probabilities_raw": technical_regime_probs,
        "regime_definitions": regime_profiles,
        "most_likely_regime": most_likely_regime_label,
        "most_likely_regime_key": most_likely_regime_key,
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
        "actionable_signals": actionable_signals,
        "backtest": backtest,
    }

    return HMMOutputs(
        model_frame=model_frame,
        posterior_probs=posterior_df,
        regime_up_table=regime_table,
        latest=latest,
    )
