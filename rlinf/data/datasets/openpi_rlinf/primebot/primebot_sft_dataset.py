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

"""Episode-streaming PrimeBot Household LeRobot v2.1 dataset.

The stock LeRobot loader builds global timestamp/index tensors for all frames.
PrimeBot contains more than 32 million frames, so this loader instead keeps a
small episode manifest, reads one episode parquet at a time, and decodes each
camera sequentially. Mixed-task training samples the four task streams with
explicit probabilities instead of inheriting their strongly imbalanced frame
counts. The training/evaluation split is defined over globally sorted episode
indices, independent of the LeRobot chunk directory layout.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import get_worker_info

from rlinf.models.embodiment.openpi.policies.primebot_policy import (
    COMPACT_ACTION_DIM,
    pack_primebot_action,
)

logger = logging.getLogger(__name__)

CAMERA_KEY_MAP = {
    "base_0_rgb": "observation.images.x2w_camera_head_realsense_compressed",
    "left_wrist_0_rgb": (
        "observation.images.x2w_camera_left_wrist_zedxonegs_rgb_raw_image_compressed"
    ),
    "right_wrist_0_rgb": (
        "observation.images.x2w_camera_right_wrist_zedxonegs_rgb_raw_image_compressed"
    ),
}


@dataclasses.dataclass(frozen=True)
class _Episode:
    index: int
    length: int


@dataclasses.dataclass(frozen=True)
class _TaskRepository:
    root: Path
    fps: int
    chunks_size: int
    data_path: str
    video_path: str
    task_prompts: dict[int, str]
    episodes: tuple[_Episode, ...]
    total_frames: int


def _read_jsonlines(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _load_repository(root: str | Path) -> _TaskRepository:
    root = Path(root).expanduser().resolve()
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"PrimeBot metadata not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if str(info.get("codebase_version")) != "v2.1":
        raise ValueError(
            f"PrimeBot loader expects LeRobot v2.1, got {info.get('codebase_version')!r} "
            f"at {root}."
        )
    features = info.get("features", {})
    required = {"observation.state", "action", *CAMERA_KEY_MAP.values()}
    missing = required - set(features)
    if missing:
        raise ValueError(
            f"PrimeBot repository {root} is missing features {sorted(missing)}."
        )
    if tuple(features["observation.state"]["shape"]) != (89,):
        raise ValueError(f"PrimeBot state must be 89-D at {root}.")
    if tuple(features["action"]["shape"]) != (89,):
        raise ValueError(f"PrimeBot raw action must be 89-D at {root}.")

    task_prompts = {
        int(item["task_index"]): str(item["task"])
        for item in _read_jsonlines(root / "meta/tasks.jsonl")
    }
    episodes = tuple(
        sorted(
            (
                _Episode(index=int(item["episode_index"]), length=int(item["length"]))
                for item in _read_jsonlines(root / "meta/episodes.jsonl")
            ),
            key=lambda episode: episode.index,
        )
    )
    if not episodes:
        raise ValueError(f"PrimeBot repository has no episodes: {root}")
    return _TaskRepository(
        root=root,
        fps=int(info["fps"]),
        chunks_size=int(info["chunks_size"]),
        data_path=str(info["data_path"]),
        video_path=str(info["video_path"]),
        task_prompts=task_prompts,
        episodes=episodes,
        total_frames=int(info["total_frames"]),
    )


def _list_column_to_numpy(column: pa.ChunkedArray, width: int) -> np.ndarray:
    array = column.combine_chunks()
    if len(array) == 0:
        return np.empty((0, width), dtype=np.float32)
    try:
        values = array.values.to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=np.float32).reshape(len(array), width)
    except (AttributeError, ValueError):
        return np.asarray(array.to_pylist(), dtype=np.float32)


class PrimeBotSftIterableDataset(torch.utils.data.IterableDataset):
    """Stream one or more PrimeBot task repositories without global indexing."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        action_horizon: int,
        task_sampling_weights: Sequence[float] | None,
        shuffle: bool,
        seed: int,
        dist_rank: int,
        dist_world_size: int,
        split: str = "train",
        eval_episodes_per_task: int = 100,
    ) -> None:
        super().__init__()
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}.")
        if not roots:
            raise ValueError("At least one PrimeBot task repository is required.")
        if split not in {"train", "eval"}:
            raise ValueError(
                f"PrimeBot split must be 'train' or 'eval', got {split!r}."
            )
        if eval_episodes_per_task <= 0:
            raise ValueError(
                "eval_episodes_per_task must be positive, got "
                f"{eval_episodes_per_task}."
            )
        if dist_world_size <= 0 or not 0 <= dist_rank < dist_world_size:
            raise ValueError(f"Invalid distributed rank {dist_rank}/{dist_world_size}.")

        repositories = []
        for root in roots:
            repository = _load_repository(root)
            if len(repository.episodes) <= eval_episodes_per_task:
                raise ValueError(
                    f"PrimeBot repository {repository.root} has "
                    f"{len(repository.episodes)} episodes, but the requested held-out "
                    f"tail is {eval_episodes_per_task}; no training data would remain."
                )
            if split == "train":
                episodes = repository.episodes[:-eval_episodes_per_task]
            else:
                episodes = repository.episodes[-eval_episodes_per_task:]
            repositories.append(
                dataclasses.replace(
                    repository,
                    episodes=episodes,
                    total_frames=sum(episode.length for episode in episodes),
                )
            )
        self._repositories = tuple(repositories)
        fps_values = {repository.fps for repository in self._repositories}
        if fps_values != {30}:
            raise ValueError(
                f"PrimeBot action horizon assumes 30 FPS, got {fps_values}."
            )

        if task_sampling_weights is None:
            weights = np.ones(len(self._repositories), dtype=np.float64)
        else:
            weights = np.asarray(task_sampling_weights, dtype=np.float64)
        if weights.shape != (len(self._repositories),):
            raise ValueError(
                "task_sampling_weights must have one entry per repository: "
                f"got {weights.shape}, expected {(len(self._repositories),)}."
            )
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError(
                "PrimeBot task sampling weights must be finite and positive."
            )

        self._task_sampling_weights = tuple((weights / weights.sum()).tolist())
        self._action_horizon = action_horizon
        self._shuffle = shuffle
        self._seed = seed
        self._dist_rank = dist_rank
        self._dist_world_size = dist_world_size
        self._split = split
        self._eval_assignments = self._build_eval_assignments()
        self._eval_rank_frame_counts = tuple(
            sum(episode.length for _, episode in assignments)
            for assignments in self._eval_assignments
        )
        if split == "eval" and any(
            count == 0 for count in self._eval_rank_frame_counts
        ):
            raise ValueError(
                "Every distributed rank must receive held-out PrimeBot frames; "
                f"got per-rank counts {self._eval_rank_frame_counts}."
            )

        if split == "train":
            # A balanced conceptual epoch. This affects len(loader), not the
            # infinite iterator used by max_steps-based SFT.
            self._epoch_num_samples = len(self._repositories) * max(
                repository.total_frames for repository in self._repositories
            )
        else:
            self._epoch_num_samples = self._eval_rank_frame_counts[dist_rank]

    @property
    def repositories(self) -> tuple[_TaskRepository, ...]:
        return self._repositories

    @property
    def task_sampling_weights(self) -> tuple[float, ...]:
        return self._task_sampling_weights

    @property
    def split(self) -> str:
        return self._split

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(repository.root.name for repository in self._repositories)

    @property
    def eval_rank_frame_counts(self) -> tuple[int, ...]:
        return self._eval_rank_frame_counts

    def __len__(self) -> int:
        return self._epoch_num_samples

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        if self._split == "eval":
            assignments = self._eval_assignments[self._dist_rank]
            for repository_index, episode in assignments[worker_id::num_workers]:
                yield from self._stream_episode(
                    self._repositories[repository_index], episode
                )
            return

        global_worker_id = self._dist_rank * num_workers + worker_id
        global_num_workers = self._dist_world_size * num_workers
        rng = random.Random(self._seed + 104_729 * global_worker_id)

        streams = [
            self._repository_stream(
                repository,
                global_worker_id=global_worker_id,
                global_num_workers=global_num_workers,
                rng=random.Random(rng.randrange(2**63)),
            )
            for repository in self._repositories
        ]
        while True:
            repository_index = rng.choices(
                range(len(streams)), weights=self._task_sampling_weights, k=1
            )[0]
            yield next(streams[repository_index])

    def _build_eval_assignments(
        self,
    ) -> tuple[tuple[tuple[int, _Episode], ...], ...]:
        """Greedily balance held-out episodes over ranks by frame count."""
        if self._split != "eval":
            return tuple(() for _ in range(self._dist_world_size))

        candidates = [
            (repository_index, episode)
            for repository_index, repository in enumerate(self._repositories)
            for episode in repository.episodes
        ]
        candidates.sort(key=lambda item: (-item[1].length, item[0], item[1].index))
        assignments: list[list[tuple[int, _Episode]]] = [
            [] for _ in range(self._dist_world_size)
        ]
        frame_counts = [0] * self._dist_world_size
        for item in candidates:
            rank = min(range(self._dist_world_size), key=lambda i: (frame_counts[i], i))
            assignments[rank].append(item)
            frame_counts[rank] += item[1].length

        # Reading episodes in repository/index order is deterministic and avoids
        # repeatedly switching between task roots after balancing.
        for rank_assignments in assignments:
            rank_assignments.sort(key=lambda item: (item[0], item[1].index))
        return tuple(tuple(items) for items in assignments)

    def _repository_stream(
        self,
        repository: _TaskRepository,
        *,
        global_worker_id: int,
        global_num_workers: int,
        rng: random.Random,
    ) -> Iterator[dict]:
        episodes = list(repository.episodes[global_worker_id::global_num_workers])
        if not episodes:
            raise RuntimeError(
                f"Worker {global_worker_id}/{global_num_workers} received no episodes "
                f"from {repository.root}. Reduce data.num_workers."
            )
        while True:
            if self._shuffle:
                rng.shuffle(episodes)
            for episode in episodes:
                yield from self._stream_episode(repository, episode)

    def _stream_episode(
        self, repository: _TaskRepository, episode: _Episode
    ) -> Iterator[dict]:
        episode_chunk = episode.index // repository.chunks_size
        format_args = {
            "episode_chunk": episode_chunk,
            "episode_index": episode.index,
        }
        parquet_path = repository.root / repository.data_path.format(**format_args)
        table = pq.read_table(
            parquet_path,
            columns=["observation.state", "action", "task_index"],
        )
        states = _list_column_to_numpy(table["observation.state"], 89)
        actions = pack_primebot_action(_list_column_to_numpy(table["action"], 89))
        task_indices = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
        frame_count = len(states)
        if frame_count != episode.length:
            raise ValueError(
                f"Episode length mismatch for {parquet_path}: metadata={episode.length}, "
                f"parquet={frame_count}."
            )
        if actions.shape != (frame_count, COMPACT_ACTION_DIM):
            raise ValueError(f"Unexpected compact action shape {actions.shape}.")

        # Import PyAV in the worker process. Suspended task generators retain one
        # decoder per camera, allowing exact per-sample uniform task interleaving
        # without loading an episode's full-resolution video into RAM.
        import av

        containers = []
        decoders = []
        try:
            for camera_key in CAMERA_KEY_MAP.values():
                path = repository.root / repository.video_path.format(
                    video_key=camera_key, **format_args
                )
                container = av.open(str(path))
                containers.append(container)
                decoders.append(iter(container.decode(video=0)))

            offsets = np.arange(self._action_horizon)
            for frame_index in range(frame_count):
                images = {}
                for model_key, decoder in zip(CAMERA_KEY_MAP, decoders, strict=True):
                    try:
                        images[model_key] = next(decoder).to_ndarray(format="rgb24")
                    except StopIteration as exc:
                        raise RuntimeError(
                            f"Video ended before parquet at frame {frame_index}: "
                            f"episode={episode.index}, root={repository.root}."
                        ) from exc

                action_indices = np.minimum(frame_index + offsets, frame_count - 1)
                task_index = int(task_indices[frame_index])
                try:
                    prompt = repository.task_prompts[task_index]
                except KeyError as exc:
                    raise KeyError(
                        f"task_index={task_index} is absent from "
                        f"{repository.root / 'meta/tasks.jsonl'}."
                    ) from exc
                yield {
                    "images": images,
                    "state": states[frame_index],
                    "actions": actions[action_indices],
                    "prompt": prompt,
                    "task_name": repository.root.name,
                }
        finally:
            for container in containers:
                container.close()
