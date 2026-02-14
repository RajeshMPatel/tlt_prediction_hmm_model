"""Configuration utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration from disk."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_paths(project_root: Path, cfg: dict[str, Any]) -> dict[str, Path]:
    """Resolve configured relative paths to absolute paths."""
    raw = project_root / cfg["paths"]["raw_data_dir"]
    processed = project_root / cfg["paths"]["processed_data_dir"]
    charts = project_root / cfg["paths"]["charts_dir"]
    daily = project_root / cfg["paths"]["daily_reports_dir"]
    return {
        "raw_data_dir": raw,
        "processed_data_dir": processed,
        "charts_dir": charts,
        "daily_reports_dir": daily,
    }


def ensure_directories(paths: dict[str, Path]) -> None:
    """Create required directories if they do not exist."""
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
