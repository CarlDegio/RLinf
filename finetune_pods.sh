#!/usr/bin/env bash

# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Launch RLinf SFT across Arena PyTorchJob Pods through one Ray cluster.
# Arena runs this script once per Pod. Do not wrap it in torchrun: RLinf asks
# Ray to create one FSDP actor per GPU after all Pods have joined the cluster.

set -Eeuo pipefail

REPO_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

die() {
    echo "[finetune_pods] ERROR: $*" >&2
    exit 1
}

NODE_RANK="${PET_NODE_RANK:-${RANK:-}}"
[[ -n "${NODE_RANK}" ]] || die "PET_NODE_RANK or RANK must be set by Arena."
[[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || die "Node rank must be an integer, got '${NODE_RANK}'."

NUM_NODES=4
GPUS_PER_NODE=8
if [[ -n "${PET_NNODES:-}" && "${PET_NNODES}" != "${NUM_NODES}" ]]; then
    die "Arena must provide exactly ${NUM_NODES} Pods; PET_NNODES=${PET_NNODES}."
fi
(( NODE_RANK < NUM_NODES )) || die "Node rank ${NODE_RANK} is outside [0, ${NUM_NODES})."

POD_IP="${POD_IP:-$(hostname -I | awk '{print $1}')}"
[[ -n "${POD_IP}" ]] || die "Could not determine this Pod's IP address."

HEAD_HOST="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
if (( NODE_RANK == 0 )) && [[ -z "${HEAD_HOST}" ]]; then
    HEAD_HOST="${POD_IP}"
fi
[[ -n "${HEAD_HOST}" ]] || die "PET_MASTER_ADDR or MASTER_ADDR must be set on worker Pods."

# Arena exposes PET_MASTER_PORT through the master Service. Reusing that port
# for Ray avoids depending on an additional Service port.
RAY_PORT="${RAY_PORT:-${PET_MASTER_PORT:-${MASTER_PORT:-29500}}}"
[[ "${RAY_PORT}" =~ ^[0-9]+$ ]] || die "Ray port must be an integer, got '${RAY_PORT}'."

EXPECTED_GPUS=$((NUM_NODES * GPUS_PER_NODE))
CONFIG_NAME="${CONFIG_NAME:-primebot_sft_openpi_pi05_task02_delta}"
RAY_CLUSTER_TIMEOUT_SECONDS="${RAY_CLUSTER_TIMEOUT_SECONDS:-900}"
RAY_STATUS_INTERVAL_SECONDS="${RAY_STATUS_INTERVAL_SECONDS:-10}"
RAY_WORKER_START_TIMEOUT_SECONDS="${RAY_WORKER_START_TIMEOUT_SECONDS:-300}"
WORKER_FAILURE_LIMIT="${WORKER_FAILURE_LIMIT:-3}"
LOG_ROOT="${LOG_ROOT:-${REPO_PATH}/logs}"

[[ -x "${REPO_PATH}/.venv/bin/python" ]] || die "Missing executable ${REPO_PATH}/.venv/bin/python."
cd "${REPO_PATH}"
source "${REPO_PATH}/.venv/bin/activate"

RAY_BIN="${RAY_BIN:-ray}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# RLinf reads this value while Ray creates its node metadata, so it must be set
# before `ray start` on every Pod.
export RLINF_NODE_RANK="${NODE_RANK}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/workspace/base_model/pi05_base_rlinf_torch/openpi_cache}"

detected_gpus="$("${PYTHON_BIN}" -c '
import ray  # noqa: F401
import torch

torch.empty(1, device="cuda")
print(torch.cuda.device_count())
')" || die "The project environment could not import Ray/Torch or initialize CUDA."
detected_gpus="$(printf '%s\n' "${detected_gpus}" | tail -n 1)"
[[ "${detected_gpus}" == "${GPUS_PER_NODE}" ]] || \
    die "Expected ${GPUS_PER_NODE} visible GPUs in this Pod, found ${detected_gpus}."

RAY_START_ATTEMPTED=0
cleanup() {
    if (( RAY_START_ATTEMPTED == 1 )); then
        "${RAY_BIN}" stop >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_ray_head() {
    echo "[finetune_pods] Starting Ray head on ${POD_IP}:${RAY_PORT} (node rank ${NODE_RANK})."
    RAY_START_ATTEMPTED=1
    "${RAY_BIN}" start \
        --head \
        --node-ip-address="${POD_IP}" \
        --port="${RAY_PORT}" \
        --disable-usage-stats \
        --include-dashboard=false
}

start_ray_worker() {
    local deadline=$((SECONDS + RAY_WORKER_START_TIMEOUT_SECONDS))
    echo "[finetune_pods] Connecting Ray worker ${NODE_RANK} to ${HEAD_HOST}:${RAY_PORT}."
    RAY_START_ATTEMPTED=1
    while ! "${RAY_BIN}" start \
        --address="${HEAD_HOST}:${RAY_PORT}" \
        --node-ip-address="${POD_IP}" \
        --disable-usage-stats; do
        "${RAY_BIN}" stop >/dev/null 2>&1 || true
        if (( SECONDS >= deadline )); then
            die "Timed out connecting Ray worker ${NODE_RANK} to ${HEAD_HOST}:${RAY_PORT}."
        fi
        sleep "${RAY_STATUS_INTERVAL_SECONDS}"
    done
}

ray_cluster_size() {
    local status_output active_nodes total_gpus
    status_output="$("${RAY_BIN}" status --address="${HEAD_HOST}:${RAY_PORT}" 2>/dev/null)" || return 1
    active_nodes="$(printf '%s\n' "${status_output}" | awk '
        /^Active:$/ { in_active=1; next }
        /^Pending:$/ { in_active=0 }
        in_active && index($0, "node_") { count++ }
        END { print count + 0 }
    ')"
    total_gpus="$(printf '%s\n' "${status_output}" | awk '
        $2 == "GPU" {
            split($1, values, "/")
            printf "%.0f", values[2]
            exit
        }
    ')"
    [[ -n "${total_gpus}" ]] || return 1
    printf '%s %s\n' "${active_nodes}" "${total_gpus}"
}

wait_for_full_cluster() {
    local deadline=$((SECONDS + RAY_CLUSTER_TIMEOUT_SECONDS))
    local cluster_size active_nodes total_gpus
    while true; do
        if cluster_size="$(ray_cluster_size)"; then
            read -r active_nodes total_gpus <<< "${cluster_size}"
            echo "[finetune_pods] Ray cluster: ${active_nodes}/${NUM_NODES} nodes, ${total_gpus}/${EXPECTED_GPUS} GPUs."
            if [[ "${active_nodes}" == "${NUM_NODES}" && "${total_gpus}" == "${EXPECTED_GPUS}" ]]; then
                return 0
            fi
        fi
        if (( SECONDS >= deadline )); then
            die "Timed out waiting for ${NUM_NODES} Ray nodes and ${EXPECTED_GPUS} GPUs."
        fi
        sleep "${RAY_STATUS_INTERVAL_SECONDS}"
    done
}

wait_for_head_shutdown() {
    local failures=0
    echo "[finetune_pods] Worker ${NODE_RANK} joined; waiting for the head job to finish."
    while (( failures < WORKER_FAILURE_LIMIT )); do
        if "${RAY_BIN}" status --address="${HEAD_HOST}:${RAY_PORT}" >/dev/null 2>&1; then
            failures=0
        else
            failures=$((failures + 1))
        fi
        if (( failures < WORKER_FAILURE_LIMIT )); then
            sleep "${RAY_STATUS_INTERVAL_SECONDS}"
        fi
    done
    echo "[finetune_pods] Ray head is unavailable; worker ${NODE_RANK} is exiting."
}

run_training() {
    local log_dir log_file
    log_dir="${LOG_ROOT}/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}-4nodes"
    log_file="${log_dir}/run_embodiment.log"
    mkdir -p "${log_dir}"

    local command=(
        "${PYTHON_BIN}"
        "${REPO_PATH}/examples/sft/train_vla_sft.py"
        --config-path "${REPO_PATH}/examples/sft/config/"
        --config-name "${CONFIG_NAME}"
        "cluster.num_nodes=${NUM_NODES}"
        "runner.logger.log_path=${log_dir}"
    )
    printf '%q ' "${command[@]}" > "${log_file}"
    printf '\n' >> "${log_file}"
    echo "[finetune_pods] Launching SFT on ${NUM_NODES} nodes / ${EXPECTED_GPUS} GPUs."
    "${command[@]}" 2>&1 | tee -a "${log_file}"
}

if (( NODE_RANK == 0 )); then
    # Use the Pod IP for local status checks; workers use Arena's master DNS.
    HEAD_HOST="${POD_IP}"
    export RAY_ADDRESS="${HEAD_HOST}:${RAY_PORT}"
    start_ray_head
    wait_for_full_cluster
    run_training
else
    export RAY_ADDRESS="${HEAD_HOST}:${RAY_PORT}"
    start_ray_worker
    wait_for_head_shutdown
fi
