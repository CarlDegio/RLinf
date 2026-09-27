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

"""Behavioral checks for the PrimeBot delta training budgets and LR schedules."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler
from rlinf.utils.runner_utils import check_progress


@pytest.fixture(
    params=[
        ("01", [5000, 10000]),
        ("02", [5000, 10000]),
        ("03", [2500, 5000, 7500, 10000]),
    ],
    ids=["task01", "task02", "task03"],
)
def delta_config(request):
    task, saved_steps = request.param
    config_dir = Path(__file__).resolve().parents[2] / "examples/sft/config"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=f"primebot_sft_openpi_pi05_task{task}_delta")
    return cfg, saved_steps


def test_delta_training_saves_on_schedule_and_stops_at_10000(delta_config) -> None:
    """Catch a wrong training budget, save cadence, or missing final checkpoint."""
    cfg, expected_saved_steps = delta_config
    runner = cfg.runner
    saved_steps = []
    finished_steps = []
    for step in range(1, 10001):
        _, save, done = check_progress(
            step, runner.max_steps, runner.val_check_interval, runner.save_interval, 1.0
        )
        if save:
            saved_steps.append(step)
        if done:
            finished_steps.append(step)
    assert saved_steps == expected_saved_steps
    assert finished_steps == [10000]
    assert cfg.actor.global_batch_size == 512
    assert 512 % (cfg.actor.micro_batch_size * 32) == 0


def test_delta_lr_warms_up_then_finishes_decay_at_step_10000(delta_config) -> None:
    """Catch scheduler/runner budgets diverging or warmup changing accidentally."""
    config, _ = delta_config
    cfg = config.actor.optim
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=cfg.lr)
    scheduler = get_lr_scheduler(
        lr_scheduler=cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=cfg.total_training_steps,
        min_lr=cfg.min_lr,
    )
    expected = {0: 2.5e-5 / 1001, 1000: 2.5e-5, 5500: 1.375e-5, 10000: 2.5e-6}
    for step in range(10001):
        if step in expected:
            assert scheduler.get_last_lr()[0] == pytest.approx(expected[step])
        if step < 10000:
            optimizer.step()
            scheduler.step()
