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

"""Tests for chunk-level PrimeBot denormalized action MSE."""

import contextlib
import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.datasets.openpi_rlinf.primebot import primebot_sft_data_loader
from rlinf.data.datasets.openpi_rlinf.primebot.primebot_sft_dataset import (
    CAMERA_KEY_MAP,
)
from rlinf.scheduler.worker.worker import Worker
from rlinf.workers.sft import fsdp_vla_sft_worker


def test_action_mse_uses_only_valid_25d_nonoverlapping_targets() -> None:
    """Padded timesteps and model-only dimensions must not affect the MSE."""
    accumulator = fsdp_vla_sft_worker.ActionMSEAccumulator(action_dim=25)
    predictions = torch.zeros(2, 30, 25)
    targets = torch.zeros(2, 25, 25)
    targets[0] = 1.0
    targets[1, :2] = 2.0
    targets[1, 2:] = 1_000.0
    predictions[:, 25:] = 1_000.0

    accumulator.update(
        predictions,
        targets,
        valid_steps=torch.tensor([25, 2]),
        trajectory_names=["task01/episode_000001", "task01/episode_000002"],
    )
    metrics = accumulator.compute()

    assert metrics["action_mse"] == pytest.approx(825.0 / 675.0)
    assert metrics["action_mse/trajectory/task01/episode_000001"] == 1.0
    assert metrics["action_mse/trajectory/task01/episode_000002"] == 4.0
    assert metrics["num_chunks"] == 2
    assert metrics["num_action_steps"] == 27
    assert metrics["num_action_elements"] == 675
    for dim in range(25):
        assert metrics[f"action_mse/dim_{dim:02d}"] == pytest.approx(33.0 / 27.0)


def test_eval_transform_and_collate_preserve_raw_trajectory_chunk() -> None:
    """Normalization must not replace the raw MSE target or chunk metadata."""
    raw_actions = np.arange(25 * 25, dtype=np.float32).reshape(25, 25)
    source_item = {
        "images": {},
        "state": np.zeros(89, dtype=np.float32),
        "actions": np.zeros((30, 25), dtype=np.float32),
        "prompt": "open the washer",
        "task_name": "task01",
        "eval_actions": raw_actions,
        "eval_valid_steps": 13,
        "trajectory_name": "task01/episode_000123",
    }
    transformed_item = {
        "image": {key: np.zeros((3, 4, 4), dtype=np.float32) for key in CAMERA_KEY_MAP},
        "image_mask": {key: np.array(True) for key in CAMERA_KEY_MAP},
        "state": np.zeros(32, dtype=np.float32),
        "tokenized_prompt": np.zeros(256, dtype=np.int64),
        "tokenized_prompt_mask": np.ones(256, dtype=np.bool_),
        "actions": np.zeros((30, 32), dtype=np.float32),
    }
    dataset = primebot_sft_data_loader._TransformedDataset(
        [source_item], lambda _item: dict(transformed_item)
    )

    batch = primebot_sft_data_loader._collate(
        [next(iter(dataset))], include_task_names=True
    )

    torch.testing.assert_close(batch["eval_actions"][0], torch.from_numpy(raw_actions))
    assert batch["eval_valid_steps"].tolist() == [13]
    assert batch["trajectory_names"] == ["task01/episode_000123"]


def test_action_mse_accumulators_merge_disjoint_trajectories() -> None:
    """Distributed ranks must preserve per-trajectory and aggregate statistics."""
    first = fsdp_vla_sft_worker.ActionMSEAccumulator(action_dim=25)
    second = fsdp_vla_sft_worker.ActionMSEAccumulator(action_dim=25)
    first.update(
        torch.zeros(1, 30, 25),
        torch.ones(1, 25, 25),
        valid_steps=torch.tensor([25]),
        trajectory_names=["task01/episode_000001"],
    )
    second.update(
        torch.zeros(1, 30, 25),
        torch.full((1, 25, 25), 2.0),
        valid_steps=torch.tensor([5]),
        trajectory_names=["task01/episode_000002"],
    )

    first.merge_state(second.state_dict())
    metrics = first.compute()

    assert metrics["action_mse"] == pytest.approx((625.0 + 500.0) / 750.0)
    assert metrics["num_trajectories"] == 2
    assert metrics["num_chunks"] == 2
    assert metrics["num_action_steps"] == 30


def test_worker_samples_denormalized_actions_for_mse() -> None:
    """The worker must request sampled actions, not flow-matching loss."""

    class FakeModel:
        def __call__(self, **kwargs):
            assert kwargs["return_denormalized_actions"] is True
            assert kwargs["train"] is False
            return {"predicted_actions": torch.ones(1, 30, 25)}

    worker = types.SimpleNamespace(
        model=FakeModel(),
        amp_context=contextlib.nullcontext(),
        eval_action_generator=torch.Generator().manual_seed(42),
    )
    batch = {
        "observation": "observation",
        "actions": torch.zeros(1, 30, 32),
        "eval_actions": torch.zeros(1, 25, 25),
        "eval_valid_steps": torch.tensor([13]),
        "trajectory_names": ["task01/episode_000123"],
    }

    prediction, target, valid_steps, trajectory_names = (
        fsdp_vla_sft_worker.FSDPVlaSftWorker.get_eval_action_mse_output(worker, batch)
    )

    assert prediction.shape == (1, 30, 25)
    assert target.shape == (1, 25, 25)
    assert valid_steps.tolist() == [13]
    assert trajectory_names == ["task01/episode_000123"]


def test_denormalized_eval_records_public_run_eval_duration(monkeypatch) -> None:
    """Runner duration consumption must find the public ``run_eval`` timer."""

    class FakeModel:
        @staticmethod
        def eval() -> None:
            return None

    worker = types.SimpleNamespace(
        eval_data_loader=[{"sample": 0}],
        eval_batch_size=1,
        device=torch.device("cpu"),
        cfg=OmegaConf.create({"actor": {"seed": 42}}),
        _rank=0,
        _timer_metrics={},
        _trace_category="test",
        model=FakeModel(),
    )
    worker.worker_timer = types.MethodType(Worker.worker_timer, worker)
    worker.get_eval_action_mse_output = lambda _batch: (
        torch.zeros(1, 30, 25),
        torch.zeros(1, 25, 25),
        torch.tensor([25]),
        ["task01/episode_000001"],
    )
    data_config = types.SimpleNamespace(
        eval_action_chunk_size=25,
        action_dim=25,
        num_samples_per_rank=(1,),
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value: output.__setitem__(0, value),
    )

    fsdp_vla_sft_worker.FSDPVlaSftWorker._run_denormalized_action_mse_eval(
        worker, data_config
    )

    assert Worker.pop_execution_time(worker, "run_eval") >= 0.0
