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

"""Tests for the global PrimeBot episode-tail train/eval split."""

import json
from pathlib import Path

from rlinf.data.datasets.openpi_rlinf.primebot import primebot_sft_dataset
from rlinf.data.datasets.openpi_rlinf.primebot.primebot_sft_dataset import (
    CAMERA_KEY_MAP,
    PrimeBotSftIterableDataset,
)
from toolkits.lerobot.calculate_primebot_norm_stats import _episode_paths


def _write_repository(root: Path, lengths: list[int]) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    features = {
        "observation.state": {"shape": [89]},
        "action": {"shape": [89]},
        **{
            camera_key: {"shape": [3, 224, 224]}
            for camera_key in CAMERA_KEY_MAP.values()
        },
    }
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "chunks_size": 2,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "total_frames": sum(lengths),
        "features": features,
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "test task"}) + "\n",
        encoding="utf-8",
    )
    # Deliberately reverse metadata order and cross several chunk boundaries.
    episodes = [
        {"episode_index": index, "length": lengths[index]}
        for index in reversed(range(len(lengths)))
    ]
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes),
        encoding="utf-8",
    )


def _build_dataset(
    roots: list[Path], *, split: str, rank: int = 0, world_size: int = 1
) -> PrimeBotSftIterableDataset:
    return PrimeBotSftIterableDataset(
        roots,
        action_horizon=30,
        task_sampling_weights=None,
        shuffle=False,
        seed=42,
        dist_rank=rank,
        dist_world_size=world_size,
        split=split,
        eval_episodes_per_task=3,
    )


def test_global_tail_split_ignores_metadata_and_chunk_order(tmp_path: Path) -> None:
    roots = [tmp_path / "task_a", tmp_path / "task_b"]
    _write_repository(roots[0], [1, 2, 3, 4, 5, 6, 7, 8])
    _write_repository(roots[1], [8, 7, 6, 5, 4, 3, 2, 1])

    train = _build_dataset(roots, split="train")
    evaluate = _build_dataset(roots, split="eval")

    for repository in train.repositories:
        assert [episode.index for episode in repository.episodes] == list(range(5))
    for repository in evaluate.repositories:
        assert [episode.index for episode in repository.episodes] == [5, 6, 7]

    assert [repository.total_frames for repository in train.repositories] == [15, 30]
    assert [repository.total_frames for repository in evaluate.repositories] == [21, 6]


def test_eval_rank_assignments_are_disjoint_and_complete(tmp_path: Path) -> None:
    roots = [tmp_path / "task_a", tmp_path / "task_b"]
    _write_repository(roots[0], [1, 1, 1, 1, 1, 10, 8, 6])
    _write_repository(roots[1], [1, 1, 1, 1, 1, 9, 7, 5])

    evaluate = _build_dataset(roots, split="eval", world_size=2)
    assigned = []
    for rank_assignments in evaluate._eval_assignments:
        assigned.append(
            {
                (repository_index, episode.index)
                for repository_index, episode in rank_assignments
            }
        )

    assert assigned[0].isdisjoint(assigned[1])
    assert assigned[0] | assigned[1] == {
        (repository_index, episode_index)
        for repository_index in range(2)
        for episode_index in (5, 6, 7)
    }
    assert sum(evaluate.eval_rank_frame_counts) == 45
    assert (
        max(evaluate.eval_rank_frame_counts) - min(evaluate.eval_rank_frame_counts) <= 1
    )


def test_norm_stats_use_the_same_training_split(tmp_path: Path) -> None:
    root = tmp_path / "task_a"
    lengths = [1, 2, 3, 4, 5, 6, 7, 8]
    _write_repository(root, lengths)

    paths, frame_count = _episode_paths(root, eval_episodes_per_task=3)

    assert [path.name for path in paths] == [
        f"episode_{episode_index:06d}.parquet" for episode_index in range(5)
    ]
    assert frame_count == sum(lengths[:5])


def test_chunked_eval_covers_each_trajectory_action_once() -> None:
    """A wrong stride or tail length would skip or double-count actions."""
    chunks = primebot_sft_dataset._build_eval_action_chunks(63, chunk_size=25)

    assert chunks == ((0, 25), (25, 25), (50, 13))


def test_chunked_eval_length_counts_inference_observations_not_frames(
    tmp_path: Path,
) -> None:
    """Eval progress must count t=0,25,... inference calls for each episode."""
    root = tmp_path / "task_a"
    _write_repository(root, [1, 1, 1, 1, 1, 26, 50, 63])

    evaluate = PrimeBotSftIterableDataset(
        [root],
        action_horizon=30,
        task_sampling_weights=None,
        shuffle=False,
        seed=42,
        dist_rank=0,
        dist_world_size=1,
        split="eval",
        eval_episodes_per_task=3,
        eval_action_chunk_size=25,
    )

    assert len(evaluate) == 2 + 2 + 3
    assert evaluate.eval_rank_sample_counts == (7,)
