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

import json

import pytest

from toolkits.lerobot.check_primebot_norm_stats import analyze_norm_stats, main


def test_analyze_norm_stats_reports_only_degenerate_dimensions(tmp_path):
    """A zero or near-zero scale must be reported with its exact dimension."""
    path = tmp_path / "norm_stats.json"
    path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {
                        "mean": [0.0, 1.0, 2.0],
                        "std": [0.0, 5.0e-7, 0.5],
                        "q01": [0.0, 1.0, -1.0],
                        "q99": [0.0, 1.0000005, 3.0],
                    },
                    "actions": {
                        "mean": [0.0, 1.0],
                        "std": [0.1, 0.2],
                        "q01": [-1.0, 0.0],
                        "q99": [1.0, 2.0],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    findings = analyze_norm_stats(
        path,
        qspan_threshold=1.0e-6,
        std_threshold=1.0e-6,
    )

    assert [(item.field, item.dimension, item.check) for item in findings] == [
        ("state", 0, "qspan"),
        ("state", 0, "std"),
        ("state", 1, "qspan"),
        ("state", 1, "std"),
    ]
    assert findings[0].q01 == 0.0
    assert findings[0].q99 == 0.0
    assert findings[0].value == 0.0


def test_analyze_norm_stats_rejects_nonfinite_values(tmp_path):
    """NaN statistics must fail instead of silently escaping comparisons."""
    path = tmp_path / "norm_stats.json"
    valid = {
        "mean": [0.0],
        "std": [0.1],
        "q01": [-1.0],
        "q99": [1.0],
    }
    payload = {
        "norm_stats": {
            "state": {**valid, "std": [float("nan")]},
            "actions": valid,
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="state.std.*non-finite"):
        analyze_norm_stats(
            path,
            qspan_threshold=1.0e-6,
            std_threshold=1.0e-6,
        )


def test_cli_returns_failure_and_prints_degenerate_dimension(tmp_path, capsys):
    """A CI caller must see both a nonzero status and actionable values."""
    path = tmp_path / "norm_stats.json"
    path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {
                        "mean": [2.0],
                        "std": [0.0],
                        "q01": [2.0],
                        "q99": [2.0],
                    },
                    "actions": {
                        "mean": [0.0],
                        "std": [1.0],
                        "q01": [-1.0],
                        "q99": [1.0],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    status = main([str(path)])

    output = capsys.readouterr().out
    assert status == 1
    assert "state" in output
    assert "dim=0" in output
    assert "qspan=0" in output
    assert "std=0" in output


def test_cli_scans_four_primebot_tasks_by_default(tmp_path, capsys):
    """An assets root must resolve exactly the four task norm files."""
    tasks = (
        "task_01_open_washing_machine",
        "task_02_close_washing_machine",
        "task_03_put_items_in_washer",
        "task_09_fold_clothing",
    )
    valid = {
        "mean": [0.0],
        "std": [0.5],
        "q01": [-1.0],
        "q99": [1.0],
    }
    for task in tasks:
        directory = tmp_path / task
        directory.mkdir()
        (directory / "norm_stats.json").write_text(
            json.dumps({"norm_stats": {"state": valid, "actions": valid}}),
            encoding="utf-8",
        )

    status = main(["--assets-root", str(tmp_path)])

    output = capsys.readouterr().out
    assert status == 0
    assert (
        "SUMMARY files=4 dimensions=0 critical=0 constant=0 std_only=0 checks=0"
        in output
    )
    for task in tasks:
        assert task in output


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (
            {
                "mean": [0.0, 1.0],
                "std": [0.1],
                "q01": [-1.0],
                "q99": [1.0],
            },
            "same length",
        ),
        (
            {
                "mean": [0.0],
                "std": [-0.1],
                "q01": [-1.0],
                "q99": [1.0],
            },
            "std must be non-negative",
        ),
        (
            {
                "mean": [0.0],
                "std": [0.1],
                "q01": [1.0],
                "q99": [-1.0],
            },
            "q99 must be greater than or equal to q01",
        ),
    ],
)
def test_analyze_norm_stats_rejects_malformed_vectors(tmp_path, state, message):
    """Invalid vector geometry must not be presented as a scale warning."""
    path = tmp_path / "norm_stats.json"
    actions = {
        "mean": [0.0],
        "std": [1.0],
        "q01": [-1.0],
        "q99": [1.0],
    }
    path.write_text(
        json.dumps({"norm_stats": {"state": state, "actions": actions}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        analyze_norm_stats(
            path,
            qspan_threshold=1.0e-6,
            std_threshold=1.0e-6,
        )


def test_cli_sorts_collapsed_quantiles_before_constant_and_std_only(tmp_path, capsys):
    """The report must put data-dependent quantile hazards first."""
    path = tmp_path / "norm_stats.json"
    state = {
        "mean": [0.0, 0.0, 0.0],
        "std": [0.5, 0.0, 0.0],
        "q01": [0.0, 0.0, -1.0],
        "q99": [0.0, 0.0, 1.0],
    }
    actions = {
        "mean": [0.0],
        "std": [1.0],
        "q01": [-1.0],
        "q99": [1.0],
    }
    path.write_text(
        json.dumps({"norm_stats": {"state": state, "actions": actions}}),
        encoding="utf-8",
    )

    assert main([str(path)]) == 1

    output = capsys.readouterr().out
    critical = output.index("CRITICAL state dim=0")
    constant = output.index("CONSTANT state dim=1")
    std_only = output.index("STD_ONLY state dim=2")
    assert critical < constant < std_only
    assert (
        "SUMMARY files=1 dimensions=3 critical=1 constant=1 std_only=1 checks=4"
        in output
    )
