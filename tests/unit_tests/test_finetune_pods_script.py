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

"""Behavior tests for the Arena multi-Pod fine-tuning entrypoint."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "finetune_pods.sh"
_ARENA_GUIDES = (
    _REPO_ROOT / "docs/source-en/rst_source/guides/multi_node.rst",
    _REPO_ROOT / "docs/source-zh/rst_source/guides/multi_node.rst",
)


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(content)
    path.chmod(0o755)
    return path


@pytest.fixture
def pod_runtime(tmp_path: Path) -> dict[str, str]:
    """Create deterministic Ray/Python boundaries for exercising the script."""
    call_log = tmp_path / "calls.log"
    ray_bin = _write_executable(
        tmp_path / "ray",
        """#!/usr/bin/env bash
set -u
printf 'ray rank=%s node_rank=%s command=%s\\n' \
  "${RANK:-unset}" "${RLINF_NODE_RANK:-unset}" "$*" >> "${CALL_LOG}"
if [[ "${1:-}" == "start" && " $* " == *" --address="* \
  && " $* " == *" --include-dashboard="* ]]; then
  printf '%s\\n' '--include-dashboard is only valid with --head' >&2
  exit 1
fi
if [[ "${1:-}" == "start" && "${FAKE_RAY_START:-ok}" == "fail" ]]; then
  exit 1
fi
case "${1:-}" in
  start|stop)
    exit 0
    ;;
  status)
    if [[ "${FAKE_RAY_STATUS:-ready}" == "down" ]]; then
      exit 1
    fi
    if [[ "${FAKE_RAY_STATUS:-ready}" == "partial" ]]; then
      nodes=3
      gpus=24
    else
      nodes=4
      gpus=32
    fi
    printf '%s\\n' 'Node status' '---------------------------------------------------------------' 'Active:'
    for ((i=0; i<nodes; i++)); do printf ' node_%s\\n' "$i"; done
    printf '%s\\n' 'Pending:' ' (no pending nodes)' 'Resources' '---------------------------------------------------------------'
    printf ' 0.0/%s.0 GPU\\n' "$gpus"
    exit 0
    ;;
esac
exit 2
""",
    )
    python_bin = _write_executable(
        tmp_path / "python",
        """#!/usr/bin/env bash
set -u
printf 'python rank=%s node_rank=%s command=%s\\n' \
  "${RANK:-unset}" "${RLINF_NODE_RANK:-unset}" "$*" >> "${CALL_LOG}"
if [[ "${1:-}" == "-c" ]]; then
  printf '%s\\n' "${FAKE_GPU_COUNT:-8}"
fi
exit 0
""",
    )
    return {
        "CALL_LOG": str(call_log),
        "RAY_BIN": str(ray_bin),
        "PYTHON_BIN": str(python_bin),
        "LOG_ROOT": str(tmp_path / "logs"),
        "RAY_STATUS_INTERVAL_SECONDS": "0.01",
        "RAY_CLUSTER_TIMEOUT_SECONDS": "1",
        "RAY_WORKER_START_TIMEOUT_SECONDS": "1",
        "WORKER_FAILURE_LIMIT": "1",
    }


def _run_script(
    pod_runtime: dict[str, str], **updates: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(pod_runtime)
    env.update(updates)
    return subprocess.run(
        ["bash", str(_SCRIPT)],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_head_starts_one_ray_cluster_and_launches_training_once(
    pod_runtime: dict[str, str],
) -> None:
    """Catch a head branch that starts training before all 32 GPUs join."""
    result = _run_script(
        pod_runtime,
        PET_NODE_RANK="0",
        PET_NNODES="4",
        PET_MASTER_ADDR="pi05-master-0",
        PET_MASTER_PORT="23456",
        POD_IP="10.0.0.1",
        RANK="0",
        FAKE_RAY_STATUS="ready",
    )

    calls = Path(pod_runtime["CALL_LOG"]).read_text()
    assert result.returncode == 0, result.stderr
    assert (
        "ray rank=0 node_rank=0 command=start --head "
        "--node-ip-address=10.0.0.1 --port=23456" in calls
    )
    assert "train_vla_sft.py" in calls
    assert "cluster.num_nodes=4" in calls
    assert calls.count("train_vla_sft.py") == 1


def test_worker_joins_head_without_launching_training(
    pod_runtime: dict[str, str],
) -> None:
    """Catch worker Pods accidentally launching duplicate SFT controllers."""
    result = _run_script(
        pod_runtime,
        PET_NODE_RANK="2",
        PET_NNODES="4",
        PET_MASTER_ADDR="pi05-master-0",
        PET_MASTER_PORT="23456",
        POD_IP="10.0.0.3",
        RANK="2",
        FAKE_RAY_STATUS="down",
    )

    calls = Path(pod_runtime["CALL_LOG"]).read_text()
    assert result.returncode == 0, result.stderr
    assert (
        "ray rank=2 node_rank=2 command=start --address=pi05-master-0:23456 "
        "--node-ip-address=10.0.0.3" in calls
    )
    assert "train_vla_sft.py" not in calls


def test_missing_arena_node_rank_fails_before_starting_ray(
    pod_runtime: dict[str, str],
) -> None:
    """Catch an unset node rank being silently treated as rank zero."""
    result = _run_script(
        pod_runtime,
        PET_NODE_RANK="",
        RANK="",
        PET_MASTER_ADDR="pi05-master-0",
        POD_IP="10.0.0.1",
    )

    assert result.returncode != 0
    assert "PET_NODE_RANK or RANK" in result.stderr
    assert not Path(pod_runtime["CALL_LOG"]).exists()


def test_gpu_count_mismatch_fails_before_starting_ray(
    pod_runtime: dict[str, str],
) -> None:
    """Catch a partially allocated Pod entering the distributed job."""
    result = _run_script(
        pod_runtime,
        PET_NODE_RANK="0",
        PET_NNODES="4",
        PET_MASTER_ADDR="pi05-master-0",
        PET_MASTER_PORT="23456",
        POD_IP="10.0.0.1",
        RANK="0",
        FAKE_GPU_COUNT="4",
    )

    calls = Path(pod_runtime["CALL_LOG"]).read_text()
    assert result.returncode != 0
    assert "Expected 8 visible GPUs in this Pod, found 4" in result.stderr
    assert not any(line.startswith("ray rank=") for line in calls.splitlines())


def test_head_times_out_when_the_full_cluster_never_joins(
    pod_runtime: dict[str, str],
) -> None:
    """Catch training starting on a partial Ray cluster."""
    result = _run_script(
        pod_runtime,
        PET_NODE_RANK="0",
        PET_NNODES="4",
        PET_MASTER_ADDR="pi05-master-0",
        PET_MASTER_PORT="23456",
        POD_IP="10.0.0.1",
        RANK="0",
        FAKE_RAY_STATUS="partial",
    )

    calls = Path(pod_runtime["CALL_LOG"]).read_text()
    assert result.returncode != 0
    assert "Timed out waiting for 4 Ray nodes and 32 GPUs" in result.stderr
    assert "train_vla_sft.py" not in calls


def test_rejects_non_four_node_arena_topology_before_starting_ray(
    pod_runtime: dict[str, str],
) -> None:
    """Catch an Arena topology that cannot provide the required 32 GPUs."""
    result = _run_script(
        pod_runtime,
        PET_NODE_RANK="0",
        PET_NNODES="3",
        PET_MASTER_ADDR="pi05-master-0",
        PET_MASTER_PORT="23456",
        POD_IP="10.0.0.1",
        RANK="0",
    )

    assert result.returncode != 0
    assert "Arena must provide exactly 4 Pods; PET_NNODES=3" in result.stderr
    assert not Path(pod_runtime["CALL_LOG"]).exists()


def test_failed_ray_start_is_cleaned_up(
    pod_runtime: dict[str, str],
) -> None:
    """Catch a failed Ray start leaking state into retry or Pod shutdown."""
    result = _run_script(
        pod_runtime,
        PET_NODE_RANK="2",
        PET_NNODES="4",
        PET_MASTER_ADDR="pi05-master-0",
        PET_MASTER_PORT="23456",
        POD_IP="10.0.0.3",
        RANK="2",
        FAKE_RAY_START="fail",
        RAY_WORKER_START_TIMEOUT_SECONDS="1",
    )

    calls = Path(pod_runtime["CALL_LOG"]).read_text()
    assert result.returncode != 0
    assert "Timed out connecting Ray worker 2" in result.stderr
    assert "command=stop" in calls


def test_arena_guides_launch_one_script_process_per_pod() -> None:
    """Catch Arena launching eight competing Ray entrypoints in each Pod."""
    for guide in _ARENA_GUIDES:
        content = guide.read_text()
        assert "--nproc-per-node=1" in content
        assert "--nproc-per-node=8" not in content
