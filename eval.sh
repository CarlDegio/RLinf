#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME="primebot_eval_openpi_pi05_task01"

if [[ $# -ne 1 ]]; then
    echo "Usage: bash eval.sh /path/to/checkpoints/global_step_N/actor" >&2
    exit 2
fi

MODEL_PATH="$(realpath "$1")"
if [[ ! -f "${MODEL_PATH}/model_state_dict/full_weights.pt" && \
      ! -f "${MODEL_PATH}/actor/model_state_dict/full_weights.pt" ]]; then
    echo "Checkpoint not found under ${MODEL_PATH}" >&2
    echo "Expected model_state_dict/full_weights.pt or actor/model_state_dict/full_weights.pt" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable is not available: ${PYTHON_BIN}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/workspace/base_model/pi05_base_rlinf_torch/openpi_cache}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# This launcher is intentionally single-machine. Prevent Ray's address="auto"
# discovery from attaching to a stale or unrelated multi-node training cluster.
export RAY_ADDRESS="local"

EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${REPO_ROOT}/logs}"
LOG_DIR="${EVAL_LOG_ROOT}/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval.log"

CMD=(
    "${PYTHON_BIN}"
    "${REPO_ROOT}/examples/sft/train_vla_sft.py"
    --config-path "${REPO_ROOT}/examples/sft/config"
    --config-name "${CONFIG_NAME}"
    "actor.model.model_path=${MODEL_PATH}"
    "runner.logger.log_path=${LOG_DIR}"
    "runner.logger.logger_backends=[]"
)

echo "Task: task01 (open washing machine)"
echo "Checkpoint: ${MODEL_PATH}"
echo "Ray address: ${RAY_ADDRESS}"
echo "Log: ${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
