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

"""PrimeBot streaming SFT loader for OpenPI_RLinf."""

from __future__ import annotations

import dataclasses
import functools
import logging
import multiprocessing
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from openpi.transforms import compose

from rlinf.data.datasets.openpi_rlinf.primebot.primebot_sft_dataset import (
    CAMERA_KEY_MAP,
    PrimeBotSftIterableDataset,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.transforms_pipeline import (
    build_openpi_transforms,
)

logger = logging.getLogger(__name__)


def _resolve_roots(data_paths: Any) -> list[str]:
    if OmegaConf.is_config(data_paths):
        data_paths = OmegaConf.to_container(data_paths, resolve=True)
    if isinstance(data_paths, (str, Path)):
        return [str(data_paths)]
    if not isinstance(data_paths, Sequence):
        raise TypeError(f"Unsupported PrimeBot data paths: {type(data_paths)!r}.")
    roots = []
    for item in data_paths:
        if isinstance(item, (str, Path)):
            roots.append(str(item))
        elif isinstance(item, dict) and "dataset_path" in item:
            roots.append(str(item["dataset_path"]))
        else:
            raise TypeError(
                "Each PrimeBot train_data_paths item must be a path or a mapping "
                f"with dataset_path; got {item!r}."
            )
    return roots


class _TransformedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: PrimeBotSftIterableDataset, transform) -> None:
        super().__init__()
        self._dataset = dataset
        self._transform = transform

    def __iter__(self):
        for item in self._dataset:
            task_name = item["task_name"]
            transformed = self._transform(item)
            transformed["_primebot_task_name"] = task_name
            yield transformed

    def __len__(self) -> int:
        return len(self._dataset)


def _collate(items, *, include_task_names: bool = False):
    if not items:
        raise ValueError("Cannot collate an empty PrimeBot SFT batch.")

    def stack(key: str, dtype=None) -> torch.Tensor:
        return torch.from_numpy(
            np.stack([np.asarray(item[key], dtype=dtype) for item in items])
        )

    images = {
        key: torch.from_numpy(
            np.stack([np.asarray(item["image"][key]) for item in items])
        )
        for key in CAMERA_KEY_MAP
    }
    image_masks = {
        key: torch.from_numpy(
            np.stack(
                [np.asarray(item["image_mask"][key], dtype=np.bool_) for item in items]
            )
        )
        for key in CAMERA_KEY_MAP
    }
    observation = Observation.from_dict(
        {
            "image": images,
            "image_mask": image_masks,
            "state": stack("state", np.float32),
            "tokenized_prompt": stack("tokenized_prompt", np.int64).long(),
            "tokenized_prompt_mask": stack("tokenized_prompt_mask", np.bool_),
        }
    )
    actions = stack("actions", np.float32)
    if include_task_names:
        return {
            "observation": observation,
            "actions": actions,
            "task_names": [item["_primebot_task_name"] for item in items],
        }
    return observation, actions


@dataclasses.dataclass(frozen=True)
class PrimeBotSftDataConfig:
    roots: tuple[str, ...]
    action_dim: int
    action_horizon: int
    max_token_len: int
    task_sampling_weights: tuple[float, ...]
    split: str
    task_names: tuple[str, ...]
    num_samples: int
    num_samples_per_rank: tuple[int, ...]


class PrimeBotSftDataLoader:
    """Expose an infinite train loader or exact finite eval loader."""

    def __init__(self, torch_loader, data_config: PrimeBotSftDataConfig) -> None:
        self._torch_loader = torch_loader
        self._data_config = data_config

    def data_config(self) -> PrimeBotSftDataConfig:
        return self._data_config

    @property
    def torch_loader(self):
        return self._torch_loader

    def __iter__(self):
        return iter(self._torch_loader)

    def __len__(self) -> int:
        return len(self._torch_loader)


def build_primebot_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[PrimeBotSftDataLoader, PrimeBotSftDataConfig]:
    """Build the PrimeBot episode/video streaming SFT loader."""
    roots = _resolve_roots(data_paths)
    model_cfg = cfg.actor.model
    data_cfg = cfg.data
    action_horizon = int(model_cfg.num_action_chunks)
    if action_horizon != 30:
        raise ValueError(
            f"PrimeBot config requires action_horizon=30, got {action_horizon}."
        )
    if int(model_cfg.openpi.model_action_dim) != 32:
        raise ValueError("PrimeBot Pi0.5 must retain the pretrained 32-D action head.")
    if int(model_cfg.action_dim) != 25:
        raise ValueError("PrimeBot environment action_dim must be 25.")
    if int(model_cfg.openpi.max_token_len) != 512:
        raise ValueError("PrimeBot Pi0.5 max_token_len must be 512.")

    weights = OmegaConf.select(cfg, "data.task_sampling_weights", default=None)
    if weights is not None:
        weights = list(weights)
    dataset = PrimeBotSftIterableDataset(
        roots,
        action_horizon=action_horizon,
        task_sampling_weights=weights,
        shuffle=not eval_dataset,
        seed=int(cfg.actor.seed),
        dist_rank=rank,
        dist_world_size=world_size,
        split="eval" if eval_dataset else "train",
        eval_episodes_per_task=int(data_cfg.get("eval_episodes_per_task", 100)),
    )

    data_kwargs = OmegaConf.select(model_cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)
    else:
        data_kwargs = {}
    # get_openpi_config(model_path=...) can discover the default mixed stats
    # while constructing DataConfig. Pass the experiment-selected path into the
    # config itself so a single-task config cannot silently retain mixed stats.
    selected_norm_stats = (
        Path(str(model_cfg.openpi.assets_dir))
        / str(model_cfg.openpi.asset_id)
        / "norm_stats.json"
    )
    data_kwargs["norm_stats_path"] = str(selected_norm_stats)
    input_transforms, _ = build_openpi_transforms(
        str(model_cfg.model_path),
        str(model_cfg.openpi.config_name),
        data_kwargs=data_kwargs,
        norm_stats_dir=str(model_cfg.openpi.assets_dir),
        norm_stats_asset_id=str(model_cfg.openpi.asset_id),
    )
    source = _TransformedDataset(dataset, compose(input_transforms))

    batch_size = (
        int(cfg.actor.eval_batch_size)
        if eval_dataset
        else int(cfg.actor.micro_batch_size)
    )
    num_workers = (
        int(data_cfg.get("eval_num_workers", 0))
        if eval_dataset
        else int(data_cfg.num_workers)
    )
    context = multiprocessing.get_context("spawn") if num_workers > 0 else None
    torch_loader = torch.utils.data.DataLoader(
        source,
        batch_size=batch_size,
        num_workers=num_workers,
        multiprocessing_context=context,
        persistent_workers=num_workers > 0,
        collate_fn=functools.partial(_collate, include_task_names=eval_dataset),
        drop_last=not eval_dataset,
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2))
        if num_workers > 0
        else None,
    )
    resolved = PrimeBotSftDataConfig(
        roots=tuple(roots),
        action_dim=25,
        action_horizon=action_horizon,
        max_token_len=512,
        task_sampling_weights=dataset.task_sampling_weights,
        split=dataset.split,
        task_names=dataset.task_names,
        num_samples=sum(repository.total_frames for repository in dataset.repositories),
        num_samples_per_rank=(dataset.eval_rank_frame_counts if eval_dataset else ()),
    )
    logger.info(
        "PrimeBot SFT loader: split=%s, roots=%s, sampling_weights=%s, "
        "batch_size=%d, num_workers=%d, horizon=%d, samples_per_rank=%s",
        dataset.split,
        roots,
        dataset.task_sampling_weights,
        batch_size,
        num_workers,
        action_horizon,
        dataset.eval_rank_frame_counts if eval_dataset else "infinite",
    )
    loader = PrimeBotSftDataLoader(torch_loader, resolved)
    return loader, resolved
