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

"""Compute single-task PrimeBot joint-delta statistics without decoding video.

Every training observation contributes a full action chunk. Joint targets are
relative to that observation's raw state; wheel velocities stay absolute. Chunks
repeat the last action at an episode's tail, exactly as the SFT loader does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from openpi.shared import normalize

from rlinf.models.embodiment.openpi.policies.primebot_policy import (
    pack_primebot_action,
    primebot_joint_delta_actions,
)
from toolkits.lerobot.calculate_primebot_norm_stats import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    _episode_paths,
    _list_column_to_numpy,
)

logger = logging.getLogger(__name__)
TASK_NAME = "task_03_put_items_in_washer"


def compute_delta_norm_stats(
    task_root: Path,
    *,
    action_horizon: int = 30,
    eval_episodes_per_task: int = 100,
    frames_per_batch: int = 1024,
) -> tuple[dict[str, normalize.NormStats], dict]:
    """Scan all training chunks and return pre-clipping statistics and metadata.

    Uses OpenPI's streaming moments and approximate histogram quantiles. Float64
    accumulators avoid cancellation for almost-constant state/action dimensions.
    Temporary chunk storage is bounded by ``frames_per_batch * action_horizon``.
    """
    if action_horizon <= 0 or frames_per_batch <= 0 or eval_episodes_per_task <= 0:
        raise ValueError(
            "horizon, batch size and held-out episode count must be positive."
        )
    task_root = Path(task_root)
    paths, expected_frames = _episode_paths(task_root, eval_episodes_per_task)
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    training_frames = 0
    offsets = np.arange(action_horizon)
    for episode_number, path in enumerate(paths, start=1):
        table = pq.read_table(path, columns=["observation.state", "action"])
        states = _list_column_to_numpy(table["observation.state"], 89)
        actions = pack_primebot_action(_list_column_to_numpy(table["action"], 89))
        if (
            states.shape != (len(actions), 89)
            or not len(states)
            or not np.isfinite(states).all()
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"Invalid state/action data in {path}.")
        state_stats.update(states.astype(np.float64))
        for start in range(0, len(states), frames_per_batch):
            stop = min(start + frames_per_batch, len(states))
            indices = np.minimum(
                np.arange(start, stop)[:, None] + offsets[None, :], len(states) - 1
            )
            delta = primebot_joint_delta_actions(actions[indices], states[start:stop])
            action_stats.update(delta.astype(np.float64))
        training_frames += len(states)
        if episode_number % 100 == 0 or episode_number == len(paths):
            logger.info(
                "%s: %d/%d episodes, %d/%d observation frames, %d action vectors",
                task_root.name,
                episode_number,
                len(paths),
                training_frames,
                expected_frames,
                training_frames * action_horizon,
            )
    if training_frames != expected_frames:
        raise ValueError(
            f"Training frame count mismatch: read {training_frames}, "
            f"metadata declares {expected_frames}."
        )

    stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }
    manifest = {
        "action_space": "joint_delta",
        "reference": "observation.state[t, 0:22] for all action[t+k] in the chunk",
        "action_horizon": action_horizon,
        "episode_tail": "repeat_last_action",
        "task_root": str(task_root.resolve()),
        "training_episodes": len(paths),
        "training_frames": training_frames,
        "action_vectors": training_frames * action_horizon,
        "eval_episodes_per_task": eval_episodes_per_task,
        "state_dim": 89,
        "action_dim": 25,
        "action_indices": [*range(22), 83, 84, 85],
        "joint_delta_indices": list(range(22)),
        "absolute_wheel_indices": [22, 23, 24],
        "quantiles": "OpenPI RunningStats histogram, before clipping",
        "action_dims_with_std_below_0_01": np.flatnonzero(
            stats["actions"].std < 0.01
        ).tolist(),
        "source_metadata_sha256": {
            name: hashlib.sha256((task_root / "meta" / name).read_bytes()).hexdigest()
            for name in ("info.json", "episodes.jsonl")
        },
    }
    return stats, manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-root", type=Path, default=DEFAULT_DATASET_ROOT / TASK_NAME
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "primebot" / f"{TASK_NAME}_delta_h30",
    )
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--eval-episodes-per-task", type=int, default=100)
    parser.add_argument("--frames-per-batch", type=int, default=1024)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    stats_path = args.output_dir / "norm_stats.json"
    manifest_path = args.output_dir / "norm_stats_manifest.json"
    if stats_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Normalization assets already exist at {args.output_dir}; "
            "choose a new --output-dir to avoid replacing experiment statistics."
        )
    stats, manifest = compute_delta_norm_stats(
        args.task_root,
        action_horizon=args.action_horizon,
        eval_episodes_per_task=args.eval_episodes_per_task,
        frames_per_batch=args.frames_per_batch,
    )
    normalize.save(args.output_dir, stats)
    manifest["norm_stats_sha256"] = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", stats_path)
    logger.info(
        "Action dimensions with std < 0.01: %s",
        manifest["action_dims_with_std_below_0_01"],
    )


if __name__ == "__main__":
    main()
