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

"""Tests for low-overhead SFT actor phase timing."""

import contextlib
import types

import torch
from omegaconf import OmegaConf

from rlinf.runners import sft_runner
from rlinf.scheduler import Worker
from rlinf.utils.distributed import ScopedTimer
from rlinf.workers.sft import fsdp_sft_worker


def test_sft_worker_records_actor_phase_durations(monkeypatch) -> None:
    """A training step must expose each actionable actor phase."""

    class FakeModel:
        @staticmethod
        def train() -> None:
            return None

    class FakeOptimizer:
        param_groups = [{"lr": 1e-4}]

        @staticmethod
        def zero_grad(*, set_to_none: bool) -> None:
            assert set_to_none is True

    class FakeLrScheduler:
        @staticmethod
        def step() -> None:
            return None

    worker = types.SimpleNamespace(
        _timer_metrics={},
        _trace_category="test",
        model=FakeModel(),
        gradient_accumulation=2,
        data_iter=iter([{"sample": 0}, {"sample": 1}]),
        _data_iter_offset=0,
        _data_epoch=0,
        grad_scaler=types.SimpleNamespace(scale=lambda loss: loss),
        optimizer=FakeOptimizer(),
        lr_scheduler=FakeLrScheduler(),
        global_step=1,
    )
    worker.worker_timer = types.MethodType(Worker.worker_timer, worker)
    worker.before_micro_batch = lambda *_args, **_kwargs: contextlib.nullcontext()
    worker.get_train_model_output = lambda _batch: (
        torch.tensor(1.0, requires_grad=True),
        {"loss": 1.0},
    )
    worker.optimizer_step = lambda: (torch.tensor(1.0), [1e-4])
    monkeypatch.setattr(
        fsdp_sft_worker,
        "all_reduce_dict",
        lambda metrics, **_kwargs: metrics,
    )

    fsdp_sft_worker.FSDPSftWorker.run_training(worker)

    durations = Worker.pop_execution_times(worker)
    assert set(durations) == {
        "run_training",
        "data",
        "forward",
        "backward",
        "optimizer",
        "metrics_reduce",
    }
    assert all(duration >= 0.0 for duration in durations.values())


def test_sft_runner_consumes_actor_phase_times_before_eval(monkeypatch) -> None:
    """Training timers must be consumed before the shared actor starts eval."""

    class TrainHandle:
        consumed = False

        @staticmethod
        def wait():
            return [{"loss": 1.0}]

        def consume_durations(self):
            self.consumed = True
            return {
                "run_training": 5.0,
                "data": 0.1,
                "forward": 1.2,
                "backward": 2.3,
                "optimizer": 0.8,
                "metrics_reduce": 0.05,
            }

        @staticmethod
        def consume_duration():
            raise AssertionError("runner must consume all actor timing tags together")

    class EvalHandle:
        @staticmethod
        def wait():
            return [{"eval_accuracy": 0.5}]

        @staticmethod
        def consume_duration():
            return 0.4

    train_handle = TrainHandle()

    class FakeActor:
        @staticmethod
        def set_global_step(_step: int) -> None:
            return None

        @staticmethod
        def run_training():
            return train_handle

        @staticmethod
        def run_eval():
            assert train_handle.consumed is True
            return EvalHandle()

    class FakeMetricLogger:
        def __init__(self) -> None:
            self.records = []

        def log(self, metrics, step: int) -> None:
            self.records.append((dict(metrics), step))

        @staticmethod
        def finish() -> None:
            return None

    class FakeProgressBar:
        @staticmethod
        def set_postfix(_metrics, refresh: bool) -> None:
            assert refresh is False

        @staticmethod
        def update(_increment: int) -> None:
            return None

    monkeypatch.setattr(sft_runner, "tqdm", lambda **_kwargs: FakeProgressBar())
    monkeypatch.setattr(
        sft_runner,
        "check_progress",
        lambda *_args, **_kwargs: (True, False, False),
    )

    runner = sft_runner.SFTRunner.__new__(sft_runner.SFTRunner)
    runner.cfg = OmegaConf.create(
        {"runner": {"val_check_interval": 1, "save_interval": -1}}
    )
    runner.actor = FakeActor()
    runner.global_step = 0
    runner.max_steps = 1
    runner.timer = ScopedTimer(reduction="max", sync_cuda=False)
    runner.metric_logger = FakeMetricLogger()
    runner.early_stop = None

    runner.run()

    time_metrics = runner.metric_logger.records[0][0]
    assert time_metrics["time/training"] == 5.0
    assert time_metrics["time/actor/data"] == 0.1
    assert time_metrics["time/actor/forward"] == 1.2
    assert time_metrics["time/actor/backward"] == 2.3
    assert time_metrics["time/actor/optimizer"] == 0.8
    assert time_metrics["time/actor/metrics_reduce"] == 0.05
