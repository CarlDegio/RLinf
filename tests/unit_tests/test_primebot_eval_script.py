# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the single-machine PrimeBot task01 eval launcher."""

import os
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "examples/sft/config"
EVAL_SCRIPT = REPO_ROOT / "eval.sh"


def test_task01_eval_config_selects_nonoverlapping_raw_action_mse() -> None:
    """A config regression must not silently restore per-frame flow loss."""
    with initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="primebot_eval_openpi_pi05_task01")

    assert cfg.data.train_data_paths is None
    assert list(cfg.data.val_data_paths) == [
        "/mnt/workspace/dataset/PrimeBotHouseholdByTask/train/"
        "task_01_open_washing_machine"
    ]
    assert cfg.data.eval_metric == "denormalized_action_mse"
    assert cfg.data.eval_action_chunk_size == 25
    assert cfg.cluster.num_nodes == 1
    assert cfg.cluster.component_placement.actor == "0-7"
    assert cfg.actor.eval_batch_size == 1
    assert cfg.actor.model.openpi.asset_id == "primebot/task_01_open_washing_machine"


def test_eval_script_launches_task01_checkpoint(tmp_path: Path) -> None:
    """The launcher must pass the selected actor checkpoint to task01 eval."""
    actor_dir = tmp_path / "global_step_15000" / "actor"
    weights = actor_dir / "model_state_dict" / "full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.touch()
    env = {
        **os.environ,
        "PYTHON_BIN": "/bin/echo",
        "EVAL_LOG_ROOT": str(tmp_path / "logs"),
    }

    result = subprocess.run(
        ["bash", str(EVAL_SCRIPT), str(actor_dir)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config-name primebot_eval_openpi_pi05_task01" in result.stdout
    assert f"actor.model.model_path={actor_dir}" in result.stdout
    assert "runner.logger.logger_backends=[]" in result.stdout
    assert "Ray address: local" in result.stdout
