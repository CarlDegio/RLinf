#!/usr/bin/env bash
# Export official PrimeBot validation actions. Additional flags go to Python.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/workspace/base_model/pi05_base_rlinf_torch/openpi_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
exec "${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}" \
    "${REPO_ROOT}/examples/sft/predict_primebot_validation.py" "$@"
