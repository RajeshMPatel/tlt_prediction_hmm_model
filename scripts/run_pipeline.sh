#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-4}"
mkdir -p "${MPLCONFIGDIR}"

python "${PROJECT_ROOT}/src/pipeline.py" --config "${PROJECT_ROOT}/config/config.yaml" "$@"
