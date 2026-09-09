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

"""Boundary, routing, and submission integrity tests (no GPU required)."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.sft.predict_primebot_validation import (
    ACTION_SCHEMA,
    INSTRUCTIONS,
    build_chunks,
    export_episode,
    normalize_segments,
    read_partial,
    save_partial,
    select_latest_checkpoint,
)


def segment(start, end, task):
    return {
        "start_frame_index": start,
        "end_frame_index": end,
        "instruction": INSTRUCTIONS.get(task, task),
    }


def test_chunks_restart_at_instruction_boundary_and_merge_adjacent():
    segments = normalize_segments(
        [
            segment(0, 7, "Start remote operation."),
            segment(7, 31, "task01"),
            segment(31, 40, "task01"),
            segment(40, 69, "task02"),
            segment(69, 73, "Invalid"),
        ],
        73,
    )
    assert build_chunks(segments, "task01", 25) == [(7, 25), (32, 8)]
    assert build_chunks(segments, "task02", 25) == [(40, 25), (65, 4)]
    assert build_chunks(segments, "task09", 25) == []


@pytest.mark.parametrize(
    "segments",
    [
        [segment(1, 10, "task01")],
        [segment(0, 6, "task01"), segment(5, 10, "task02")],
        [segment(0, 11, "task01")],
        [segment(0, 9, "task01")],
    ],
)
def test_invalid_annotation_coverage_rejected(segments):
    with pytest.raises(ValueError):
        normalize_segments(segments, 10)


def test_latest_uses_numeric_step_and_requires_complete_weight_file(tmp_path):
    for step in (9, 100, 20):
        p = tmp_path / f"global_step_{step}/actor/model_state_dict/full_weights.pt"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"weights")
    (tmp_path / "global_step_999/actor").mkdir(parents=True)
    assert select_latest_checkpoint(tmp_path).parent.name == "global_step_100"


def test_export_preserves_ids_and_zeros_non_target_frames(tmp_path):
    source = tmp_path / "input.parquet"
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([8] * 12, type=pa.int64()),
                "frame_index": pa.array(range(12), type=pa.int64()),
                "index": pa.array(range(300, 312), type=pa.int64()),
            }
        ),
        source,
    )
    episode = {
        "dataset": "valid",
        "episode_index": 8,
        "length": 12,
        "parquet": str(source),
        "relative": "chunk-000/episode_000008.parquet",
        "segments": normalize_segments(
            [
                segment(0, 2, "Start remote operation."),
                segment(2, 5, "task01"),
                segment(5, 8, "Invalid"),
                segment(8, 12, "task02"),
            ],
            12,
        ),
    }
    for task, start, end, value in [("task01", 2, 5, 1), ("task02", 8, 12, 2)]:
        mask = np.zeros(12, dtype=bool)
        mask[start:end] = True
        actions = np.zeros((12, 25), np.float32)
        actions[mask] = value
        save_partial(tmp_path, task, episode, actions, mask)
    report = export_episode(tmp_path, episode)
    result = pq.read_table(tmp_path / "actions/valid/chunk-000/episode_000008.parquet")
    assert result.schema == ACTION_SCHEMA
    for key in ("episode_index", "frame_index", "index"):
        assert result[key].equals(pq.read_table(source)[key])
    actions = np.asarray(result["action"].to_pylist())
    assert np.all(actions[:2] == 0) and np.all(actions[5:8] == 0)
    assert np.all(actions[2:5] == 1) and np.all(actions[8:] == 2)
    assert report["predicted_frames"] == 7
    assert report["zero_filled_frames"] == 5


def test_corrupt_partial_cannot_be_silently_resumed(tmp_path):
    episode = {
        "dataset": "valid",
        "episode_index": 0,
        "length": 4,
        "segments": normalize_segments([segment(0, 4, "task01")], 4),
    }
    save_partial(
        tmp_path, "task01", episode, np.ones((4, 25), np.float32), np.ones(4, bool)
    )
    actions, mask = read_partial(tmp_path, "task01", episode)
    assert actions.shape == (4, 25) and mask.all()
    actions[0, 0] = np.nan
    with pytest.raises(ValueError):
        save_partial(tmp_path, "task01", episode, actions, mask)
    with pytest.raises(ValueError):
        save_partial(
            tmp_path, "task01", episode, np.ones((4, 25), np.float32), np.zeros(4, bool)
        )


def test_repository_annotations_have_real_instructions_not_coarse_task_ids():
    # Fixture models the official info.json, where task_index never changes.
    data = json.loads('{"instruction_segments":{"0":[]}}')
    data["instruction_segments"]["0"] = [
        segment(0, 25, "task01"),
        segment(25, 50, "task09"),
    ]
    normalized = normalize_segments(data["instruction_segments"]["0"], 50)
    assert build_chunks(normalized, "task01", 25) == [(0, 25)]
    assert build_chunks(normalized, "task09", 25) == [(25, 25)]
