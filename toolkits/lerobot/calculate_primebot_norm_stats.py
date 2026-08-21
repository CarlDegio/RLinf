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

"""Compute PrimeBot state/action normalization statistics from parquet only.

This follows OpenPI's ``RunningStats`` implementation. Images are deliberately
not decoded. Per-task files use every training frame after excluding each
task's globally last held-out episodes. The mixed file gives each task equal
statistical weight, matching the mixed loader's uniform task sampling.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from openpi.shared import normalize

logger = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = Path("/mnt/workspace/dataset/PrimeBotHouseholdByTask/train")
DEFAULT_OUTPUT_ROOT = Path("/mnt/workspace/base_model/pi05_base_rlinf_torch")


def _pack_primebot_action(action: np.ndarray) -> np.ndarray:
    """Pack joint positions and wheel velocities into the 25-D policy action."""
    return np.concatenate([action[..., :22], action[..., 83:86]], axis=-1)


@dataclasses.dataclass(frozen=True)
class _TaskResult:
    name: str
    count: int
    state: normalize.NormStats
    actions: normalize.NormStats


def _list_column_to_numpy(column: pa.Array | pa.ChunkedArray, width: int) -> np.ndarray:
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    try:
        values = column.values.to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=np.float32).reshape(len(column), width)
    except (AttributeError, ValueError):
        return np.asarray(column.to_pylist(), dtype=np.float32)


def _episode_paths(
    task_root: Path, eval_episodes_per_task: int
) -> tuple[list[Path], int]:
    info = json.loads((task_root / "meta/info.json").read_text(encoding="utf-8"))
    episodes = [
        json.loads(line)
        for line in (task_root / "meta/episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    episodes.sort(key=lambda item: int(item["episode_index"]))
    if len(episodes) <= eval_episodes_per_task:
        raise ValueError(
            f"{task_root} has {len(episodes)} episodes, but holding out "
            f"{eval_episodes_per_task} would leave no training episodes."
        )
    training_episodes = episodes[:-eval_episodes_per_task]
    paths = [
        task_root
        / info["data_path"].format(
            episode_chunk=int(item["episode_index"]) // int(info["chunks_size"]),
            episode_index=int(item["episode_index"]),
        )
        for item in training_episodes
    ]
    return paths, sum(int(item["length"]) for item in training_episodes)


def _compute_task(
    task_root: Path,
    *,
    mixed_sample_frames: int,
    max_frames: int | None,
    eval_episodes_per_task: int,
) -> tuple[_TaskResult, tuple[np.ndarray, np.ndarray]]:
    paths, declared_total = _episode_paths(task_root, eval_episodes_per_task)
    target_total = min(declared_total, max_frames) if max_frames else declared_total
    sample_count = min(mixed_sample_frames, target_total)
    sample_positions = np.linspace(0, target_total - 1, sample_count, dtype=np.int64)
    sample_state_parts: list[np.ndarray] = []
    sample_action_parts: list[np.ndarray] = []
    stats_state_parts: list[np.ndarray] = []
    stats_action_parts: list[np.ndarray] = []
    buffered_frames = 0
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    count = 0

    def flush_stats() -> None:
        nonlocal buffered_frames
        if not stats_state_parts:
            return
        state_stats.update(np.concatenate(stats_state_parts, axis=0))
        action_stats.update(np.concatenate(stats_action_parts, axis=0))
        stats_state_parts.clear()
        stats_action_parts.clear()
        buffered_frames = 0

    for episode_number, path in enumerate(paths, start=1):
        if count >= target_total:
            break
        table = pq.read_table(path, columns=["observation.state", "action"])
        states = _list_column_to_numpy(table["observation.state"], 89)
        actions = _pack_primebot_action(_list_column_to_numpy(table["action"], 89))
        keep = min(len(states), target_total - count)
        states, actions = states[:keep], actions[:keep]
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
            raise ValueError(f"Non-finite state/action values found in {path}.")
        stats_state_parts.append(states)
        stats_action_parts.append(actions)
        buffered_frames += keep
        if buffered_frames >= 100_000:
            flush_stats()

        left = int(np.searchsorted(sample_positions, count, side="left"))
        right = int(np.searchsorted(sample_positions, count + keep, side="left"))
        if right > left:
            local = sample_positions[left:right] - count
            sample_state_parts.append(states[local])
            sample_action_parts.append(actions[local])
        count += keep
        if episode_number % 1000 == 0:
            logger.info(
                "%s: processed %d/%d episodes, %d/%d frames",
                task_root.name,
                episode_number,
                len(paths),
                count,
                target_total,
            )

    if count != target_total:
        raise ValueError(
            f"Frame count mismatch for {task_root}: read={count}, expected={target_total}."
        )
    flush_stats()
    result = _TaskResult(
        name=task_root.name,
        count=count,
        state=state_stats.get_statistics(),
        actions=action_stats.get_statistics(),
    )
    return result, (
        np.concatenate(sample_state_parts, axis=0),
        np.concatenate(sample_action_parts, axis=0),
    )


def _uniform_mixed_stats(
    results: list[_TaskResult],
    samples: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, normalize.NormStats]:
    mixed_sample_stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }
    for states, actions in samples:
        mixed_sample_stats["state"].update(states)
        mixed_sample_stats["actions"].update(actions)

    output = {}
    for key in ("state", "actions"):
        task_stats = [getattr(result, key) for result in results]
        means = np.stack([stats.mean for stats in task_stats])
        second_moments = np.stack(
            [stats.std**2 + stats.mean**2 for stats in task_stats]
        )
        mean = means.mean(axis=0)
        variance = np.maximum(second_moments.mean(axis=0) - mean**2, 0.0)
        sampled = mixed_sample_stats[key].get_statistics()
        output[key] = normalize.NormStats(
            mean=mean,
            std=np.sqrt(variance),
            q01=sampled.q01,
            q99=sampled.q99,
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--eval-episodes-per-task",
        type=int,
        default=100,
        help="Globally last episodes per task excluded from normalization.",
    )
    parser.add_argument(
        "--mixed-sample-frames",
        type=int,
        default=200_000,
        help="Frames sampled per task for the mixed quantiles (means/std stay exact).",
    )
    parser.add_argument(
        "--max-frames-per-task",
        type=int,
        default=None,
        help="Development-only cap; omit for production statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    task_roots = sorted(
        path
        for path in args.dataset_root.iterdir()
        if path.is_dir() and (path / "meta/info.json").is_file()
    )
    if len(task_roots) != 4:
        raise ValueError(
            f"Expected four PrimeBot task repositories under {args.dataset_root}, "
            f"found {[path.name for path in task_roots]}."
        )

    results = []
    samples = []
    for task_root in task_roots:
        logger.info("Computing normalization statistics for %s", task_root)
        result, sample = _compute_task(
            task_root,
            mixed_sample_frames=args.mixed_sample_frames,
            max_frames=args.max_frames_per_task,
            eval_episodes_per_task=args.eval_episodes_per_task,
        )
        normalize.save(
            args.output_root / "primebot" / result.name,
            {"state": result.state, "actions": result.actions},
        )
        results.append(result)
        samples.append(sample)

    mixed_stats = _uniform_mixed_stats(results, samples)
    normalize.save(args.output_root / "primebot/mixed_uniform", mixed_stats)
    # Keep the checkpoint-root stats consistent with the mixed PrimeBot config.
    normalize.save(args.output_root, mixed_stats)
    summary = {
        "normalization": "quantile",
        "split": "train",
        "eval_episodes_per_task": args.eval_episodes_per_task,
        "state_dim": 89,
        "raw_action_dim": 89,
        "model_action_data_dim": 25,
        "action_indices": [*range(22), 83, 84, 85],
        "mixed_task_weighting": "uniform",
        "tasks": {result.name: result.count for result in results},
    }
    summary_path = args.output_root / "primebot/norm_stats_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote PrimeBot normalization assets under %s", args.output_root)


if __name__ == "__main__":
    main()
