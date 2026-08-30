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
import math
import os
from collections import defaultdict
from typing import Any

import torch
from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class ActionMSEAccumulator:
    """Accumulate exact 25-D action MSE over valid trajectory chunks."""

    def __init__(self, action_dim: int) -> None:
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}.")
        self.action_dim = action_dim
        self.dimension_sum_squared_error = torch.zeros(action_dim, dtype=torch.float64)
        self.num_action_steps = 0
        self.num_chunks = 0
        self.trajectory_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        *,
        valid_steps: torch.Tensor,
        trajectory_names: list[str],
    ) -> None:
        """Add one batch, excluding padded timesteps and extra action dimensions."""
        if predictions.ndim != 3 or targets.ndim != 3:
            raise ValueError("Action MSE inputs must have shape [B, H, D].")
        batch_size = predictions.shape[0]
        if (
            targets.shape[0] != batch_size
            or len(valid_steps) != batch_size
            or len(trajectory_names) != batch_size
        ):
            raise ValueError("Action MSE batch metadata does not match batch size.")
        if (
            predictions.shape[-1] < self.action_dim
            or targets.shape[-1] < self.action_dim
        ):
            raise ValueError(
                f"Action MSE requires at least {self.action_dim} action dimensions."
            )

        predictions = predictions.detach().to(device="cpu", dtype=torch.float64)
        targets = targets.detach().to(device="cpu", dtype=torch.float64)
        for index, (valid, trajectory_name) in enumerate(
            zip(valid_steps.tolist(), trajectory_names, strict=True)
        ):
            valid = int(valid)
            max_horizon = min(predictions.shape[1], targets.shape[1])
            if not 0 < valid <= max_horizon:
                raise ValueError(
                    f"valid_steps must be in [1, {max_horizon}], got {valid}."
                )
            squared_error = torch.square(
                predictions[index, :valid, : self.action_dim]
                - targets[index, :valid, : self.action_dim]
            )
            self.dimension_sum_squared_error += squared_error.sum(dim=0)
            self.num_action_steps += valid
            self.num_chunks += 1
            trajectory = self.trajectory_stats[trajectory_name]
            trajectory[0] += squared_error.sum().item()
            trajectory[1] += squared_error.numel()

    def compute(self) -> dict[str, float | int]:
        """Return aggregate, per-dimension, and per-trajectory metrics."""
        num_elements = self.num_action_steps * self.action_dim
        if num_elements == 0:
            raise ValueError("Cannot compute action MSE without valid actions.")
        total_squared_error = self.dimension_sum_squared_error.sum().item()
        metrics: dict[str, float | int] = {
            "action_mse": total_squared_error / num_elements,
            "num_trajectories": len(self.trajectory_stats),
            "num_chunks": self.num_chunks,
            "num_action_steps": self.num_action_steps,
            "num_action_elements": num_elements,
        }
        for dim, squared_error in enumerate(self.dimension_sum_squared_error.tolist()):
            metrics[f"action_mse/dim_{dim:02d}"] = squared_error / self.num_action_steps
        for trajectory_name, (squared_error, count) in sorted(
            self.trajectory_stats.items()
        ):
            metrics[f"action_mse/trajectory/{trajectory_name}"] = squared_error / count
        return metrics

    def state_dict(self) -> dict[str, Any]:
        """Return a CPU-only state suitable for distributed object gathering."""
        return {
            "action_dim": self.action_dim,
            "dimension_sum_squared_error": (self.dimension_sum_squared_error.tolist()),
            "num_action_steps": self.num_action_steps,
            "num_chunks": self.num_chunks,
            "trajectory_stats": {
                name: list(values) for name, values in self.trajectory_stats.items()
            },
        }

    def merge_state(self, state: dict[str, Any]) -> None:
        """Merge statistics produced by another worker rank."""
        if int(state["action_dim"]) != self.action_dim:
            raise ValueError(
                "Cannot merge action MSE states with different action dimensions."
            )
        self.dimension_sum_squared_error += torch.tensor(
            state["dimension_sum_squared_error"], dtype=torch.float64
        )
        self.num_action_steps += int(state["num_action_steps"])
        self.num_chunks += int(state["num_chunks"])
        for trajectory_name, (squared_error, count) in state[
            "trajectory_stats"
        ].items():
            trajectory = self.trajectory_stats[trajectory_name]
            trajectory[0] += float(squared_error)
            trajectory[1] += float(count)


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.OPENPI_RLINF:
            from rlinf.data.datasets.openpi_rlinf import (
                build_openpi_rlinf_sft_dataloader,
            )

            return build_openpi_rlinf_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.OPENPI:
            from rlinf.data.datasets.openpi_rlinf import (
                build_official_openpi_sft_dataloader,
            )

            return build_official_openpi_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.LINGBOTVLA:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif model_type == SupportedModel.DREAMZERO:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.EVO1:
            from rlinf.models.embodiment.evo1.sft_builder import (
                build_evo1_sft_dataloader,
            )

            return build_evo1_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, list[str]]:
        """Return per-sample action loss and task labels for a VLA eval batch."""
        with torch.no_grad(), self.amp_context:
            output = self.model(
                forward_type=ForwardType.SFT,
                data=batch,
                train=False,
                return_per_sample_loss=True,
            )
        if not isinstance(output, dict) or "per_sample_loss" not in output:
            raise TypeError(
                "VLA SFT evaluation requires a model output containing "
                "'per_sample_loss'."
            )
        task_names = batch.get("task_names")
        if task_names is None:
            raise ValueError("VLA SFT evaluation batch is missing 'task_names'.")
        per_sample_loss = output["per_sample_loss"].detach().float()
        if per_sample_loss.shape != (len(task_names),):
            raise ValueError(
                "Per-sample loss/task label mismatch: "
                f"loss={tuple(per_sample_loss.shape)}, tasks={len(task_names)}."
            )
        return per_sample_loss, list(task_names)

    def get_eval_action_mse_output(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        """Sample raw-scale action chunks and return their trajectory targets."""
        required = {"eval_actions", "eval_valid_steps", "trajectory_names"}
        missing = required - set(batch)
        if missing:
            raise ValueError(
                f"PrimeBot action-MSE eval batch is missing {sorted(missing)}."
            )
        with torch.no_grad(), self.amp_context:
            output = self.model(
                forward_type=ForwardType.SFT,
                data=batch,
                train=False,
                return_denormalized_actions=True,
                rng=self.eval_action_generator,
            )
        if not isinstance(output, dict) or "predicted_actions" not in output:
            raise TypeError(
                "PrimeBot action-MSE evaluation requires 'predicted_actions'."
            )
        return (
            output["predicted_actions"],
            batch["eval_actions"],
            batch["eval_valid_steps"],
            list(batch["trajectory_names"]),
        )

    def run_eval(self):
        """Evaluate finite PrimeBot streams without entering the training loop."""
        assert self.eval_data_loader is not None, "eval_data_loader is not set"
        data_config = self.eval_data_config
        if getattr(data_config, "split", None) != "eval":
            raise NotImplementedError(
                "Standalone action-loss evaluation currently supports only the "
                "finite PrimeBot eval loader."
            )

        eval_metric = str(self.cfg.data.get("eval_metric", "flow_matching_loss"))
        if eval_metric == "denormalized_action_mse":
            return self._run_denormalized_action_mse_eval(data_config)
        if eval_metric != "flow_matching_loss":
            raise ValueError(f"Unsupported PrimeBot eval_metric={eval_metric!r}.")

        local_steps = len(self.eval_data_loader)
        max_steps = max(
            math.ceil(count / self.eval_batch_size)
            for count in data_config.num_samples_per_rank
        )
        if local_steps > max_steps:
            raise RuntimeError(
                f"Local eval loader has {local_steps} steps, exceeding {max_steps}."
            )

        sums = dict.fromkeys(data_config.task_names, 0.0)
        counts = dict.fromkeys(data_config.task_names, 0.0)
        eval_data_iter = iter(self.eval_data_loader)
        last_batch = None

        with self.worker_timer():
            eval_pbar = tqdm(
                total=max_steps,
                desc="Evaluate action loss",
                dynamic_ncols=True,
                disable=self._rank != 0,
            )
            self.model.eval()
            for step in range(max_steps):
                real_batch = step < local_steps
                if real_batch:
                    try:
                        batch = next(eval_data_iter)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "PrimeBot eval loader ended before its declared length."
                        ) from exc
                    last_batch = batch
                else:
                    # FSDP ranks must issue the same number of forward calls. Reuse
                    # the final local batch only to participate in collectives; its
                    # loss is deliberately excluded from metrics.
                    if last_batch is None:
                        raise RuntimeError("PrimeBot eval rank received no batches.")
                    batch = last_batch

                per_sample_loss, task_names = self.get_eval_model_output(batch)
                if real_batch:
                    for loss, task_name in zip(
                        per_sample_loss.tolist(), task_names, strict=True
                    ):
                        sums[task_name] += loss
                        counts[task_name] += 1.0
                eval_pbar.update(1)
            eval_pbar.close()

        reduction_values = {
            **{f"sum/{task_name}": value for task_name, value in sums.items()},
            **{f"count/{task_name}": value for task_name, value in counts.items()},
        }
        reduced = all_reduce_dict(
            reduction_values,
            dtype=torch.float64,
            op=torch.distributed.ReduceOp.SUM,
        )
        total_sum = sum(reduced[f"sum/{name}"] for name in data_config.task_names)
        total_count = sum(reduced[f"count/{name}"] for name in data_config.task_names)
        metrics = {
            "loss": total_sum / max(1.0, total_count),
            "num_samples": int(total_count),
        }
        for task_name in data_config.task_names:
            task_count = reduced[f"count/{task_name}"]
            metrics[f"loss/{task_name}"] = reduced[f"sum/{task_name}"] / max(
                1.0, task_count
            )
            metrics[f"num_samples/{task_name}"] = int(task_count)
        return metrics

    def _run_denormalized_action_mse_eval(self, data_config):
        """Evaluate non-overlapping raw-scale action chunks over full trajectories."""
        if data_config.eval_action_chunk_size is None:
            raise ValueError(
                "denormalized_action_mse requires data.eval_action_chunk_size."
            )
        if data_config.eval_action_chunk_size != 25 or data_config.action_dim != 25:
            raise ValueError(
                "PrimeBot action-MSE eval currently requires 25-step chunks and "
                "25 action dimensions."
            )

        local_steps = len(self.eval_data_loader)
        max_steps = max(
            math.ceil(count / self.eval_batch_size)
            for count in data_config.num_samples_per_rank
        )
        if local_steps > max_steps:
            raise RuntimeError(
                f"Local eval loader has {local_steps} steps, exceeding {max_steps}."
            )

        self.eval_action_generator = torch.Generator(device=self.device)
        self.eval_action_generator.manual_seed(int(self.cfg.actor.seed) + self._rank)
        accumulator = ActionMSEAccumulator(action_dim=data_config.action_dim)
        eval_data_iter = iter(self.eval_data_loader)
        last_batch = None

        with self.worker_timer("run_eval"):
            eval_pbar = tqdm(
                total=max_steps,
                desc="Evaluate denormalized action MSE",
                dynamic_ncols=True,
                disable=self._rank != 0,
            )
            self.model.eval()
            for step in range(max_steps):
                real_batch = step < local_steps
                if real_batch:
                    try:
                        batch = next(eval_data_iter)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "PrimeBot action-MSE loader ended before its declared length."
                        ) from exc
                    last_batch = batch
                else:
                    if last_batch is None:
                        raise RuntimeError("PrimeBot eval rank received no batches.")
                    batch = last_batch

                prediction, target, valid_steps, trajectory_names = (
                    self.get_eval_action_mse_output(batch)
                )
                if real_batch:
                    accumulator.update(
                        prediction,
                        target,
                        valid_steps=valid_steps,
                        trajectory_names=trajectory_names,
                    )
                eval_pbar.update(1)
            eval_pbar.close()

        gathered_states = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_states, accumulator.state_dict())
        merged = ActionMSEAccumulator(action_dim=data_config.action_dim)
        for state in gathered_states:
            merged.merge_state(state)
        return merged.compute()

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        with self.amp_context:
            output = self.model(forward_type=ForwardType.SFT, data=batch)

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]

        step_metrics = {"loss": loss.detach().item()}
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        step_metrics[key] = value.detach().item()
                elif isinstance(value, (float, int)):
                    step_metrics[key] = value
        return loss, step_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            if self._rank == 0:
                torch.save(all_states, os.path.join(save_path, "data.pt"))

            torch.distributed.barrier()

            rng_state = get_rng_state()
            all_rng_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_rng_states, rng_state)
            if self._rank == 0:
                torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))

            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)

        if isinstance(self.data_loader, StatefulDataLoader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            state = all_states[self._rank]
            self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)

            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])

            torch.distributed.barrier()

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type in (SupportedModel.OPENPI_RLINF, SupportedModel.OPENPI):
            if model_type == SupportedModel.OPENPI_RLINF:
                from rlinf.data.datasets.openpi_rlinf import (
                    get_official_openpi_sft_num_batches,
                    is_official_openpi_sft_dataloader,
                )

                num_batches = (
                    get_official_openpi_sft_num_batches(self.data_loader)
                    if is_official_openpi_sft_dataloader(self.data_loader)
                    else len(self.data_loader)
                )
            else:
                from rlinf.data.datasets.openpi_rlinf import (
                    get_official_openpi_sft_num_batches,
                )

                num_batches = get_official_openpi_sft_num_batches(self.data_loader)
        else:
            return super().get_max_steps_per_epoch()
        return max(1, num_batches // self.gradient_accumulation)
