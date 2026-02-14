"""Daily pipeline entrypoint for tlt_vix_web."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.config import ensure_directories, load_config, resolve_project_paths
from src.data_loader import load_raw_data
from src.features import build_feature_dataset
from src.model import run_hmm_direction_model
from src.reporting import save_regime_charts, write_daily_outputs


def run_pipeline(config_path: Path, force_refresh: bool = False) -> dict[str, Any]:
    """Run full daily workflow and persist outputs."""
    project_root = config_path.parent.parent
    cfg = load_config(config_path)
    paths = resolve_project_paths(project_root, cfg)
    ensure_directories(paths)

    raw_result = load_raw_data(cfg=cfg, raw_dir=paths["raw_data_dir"], force_refresh=force_refresh)
    raw_df = raw_result.raw_frame

    feature_df = build_feature_dataset(raw_df=raw_df, cfg=cfg)
    feature_path = paths["processed_data_dir"] / "feature_dataset.parquet"
    feature_df.to_parquet(feature_path)

    model_outputs = run_hmm_direction_model(feature_df=feature_df, cfg=cfg)
    model_path = paths["processed_data_dir"] / "model_dataset.parquet"
    posterior_path = paths["processed_data_dir"] / "posterior_probabilities.parquet"
    regime_up_path = paths["processed_data_dir"] / "regime_up_frequencies.parquet"

    model_outputs.model_frame.to_parquet(model_path)
    model_outputs.posterior_probs.to_parquet(posterior_path)
    model_outputs.regime_up_table.to_parquet(regime_up_path)

    chart_paths = save_regime_charts(
        model_frame=model_outputs.model_frame,
        posterior_df=model_outputs.posterior_probs,
        charts_dir=paths["charts_dir"],
        lookback_days=int(cfg["reporting"]["chart_lookback_days"]),
    )

    warnings = raw_result.metadata.get("fred", {}).get("warnings", [])
    pipeline_meta = {
        "last_data_timestamp": raw_result.metadata.get("last_timestamp"),
        "raw_rows": int(raw_df.shape[0]),
        "model_rows": int(model_outputs.model_frame.shape[0]),
        "warnings": warnings,
        "sources": raw_result.metadata,
        "debug": {
            "feature_path": str(feature_path),
            "model_path": str(model_path),
            "posterior_path": str(posterior_path),
            "regime_up_path": str(regime_up_path),
            "force_refresh": force_refresh,
        },
    }
    write_daily_outputs(
        latest_payload=model_outputs.latest,
        daily_reports_dir=paths["daily_reports_dir"],
        pipeline_meta=pipeline_meta,
        chart_paths=chart_paths,
    )
    return pipeline_meta


def parse_args() -> argparse.Namespace:
    """CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run tlt_vix_web daily pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cache and force fresh download.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI main entrypoint."""
    args = parse_args()
    run_pipeline(config_path=args.config, force_refresh=args.force_refresh)


if __name__ == "__main__":
    main()
