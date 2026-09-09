#!/usr/bin/env bash

set -o pipefail

set -euo pipefail

REPO_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_PATH}"
source "${REPO_PATH}/.venv/bin/activate"

# 必须在 ray start 之前设置
export RLINF_NODE_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

trap 'ray stop >/dev/null 2>&1 || true' EXIT INT TERM

ray start --head --port=6379 --disable-usage-stats --include-dashboard=false
ray status

#然后在启动训练/评估的 shell 中设置：

export RAY_ADDRESS=auto

#再执行训练：

OPENPI_DATA_HOME=/mnt/workspace/base_model/pi05_base_rlinf_torch/openpi_cache \
bash examples/sft/run_vla_sft.sh primebot_sft_openpi_pi05_task03

