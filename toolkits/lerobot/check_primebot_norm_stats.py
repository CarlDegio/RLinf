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

"""Diagnose degenerate dimensions in PrimeBot normalization statistics."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections.abc import Sequence
from pathlib import Path

DEFAULT_ASSETS_ROOT = Path("/mnt/workspace/base_model/pi05_base_rlinf_torch/primebot")
DEFAULT_TASKS = (
    "task_01_open_washing_machine",
    "task_02_close_washing_machine",
    "task_03_put_items_in_washer",
    "task_09_fold_clothing",
)


@dataclasses.dataclass(frozen=True)
class Finding:
    """One dimension whose normalization scale is suspiciously small."""

    field: str
    dimension: int
    check: str
    value: float
    mean: float
    std: float
    q01: float
    q99: float


def _validate_field(path: Path, field: str, stats: dict) -> None:
    names = ("mean", "std", "q01", "q99")
    missing = [name for name in names if name not in stats]
    if missing:
        raise ValueError(f"{path}: {field} is missing statistics {missing}.")
    lengths = {name: len(stats[name]) for name in names}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"{path}: {field} statistic vectors must have the same length; "
            f"got {lengths}."
        )
    for statistic in names:
        if not all(math.isfinite(float(value)) for value in stats[statistic]):
            raise ValueError(f"{path}: {field}.{statistic} contains non-finite values.")
    if any(float(value) < 0 for value in stats["std"]):
        raise ValueError(f"{path}: {field}.std must be non-negative.")
    if any(
        float(q99) < float(q01)
        for q01, q99 in zip(stats["q01"], stats["q99"], strict=True)
    ):
        raise ValueError(f"{path}: {field}.q99 must be greater than or equal to q01.")


def analyze_norm_stats(
    path: Path,
    *,
    qspan_threshold: float,
    std_threshold: float,
) -> list[Finding]:
    """Return dimensions with a quantile span or standard deviation below a threshold."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats_by_field = payload.get("norm_stats", payload)
    findings = []
    for field in ("state", "actions"):
        stats = stats_by_field[field]
        _validate_field(path, field, stats)
        for dimension, (mean, std, q01, q99) in enumerate(
            zip(
                stats["mean"],
                stats["std"],
                stats["q01"],
                stats["q99"],
                strict=True,
            )
        ):
            qspan = float(q99) - float(q01)
            values = {"qspan": qspan, "std": float(std)}
            thresholds = {
                "qspan": qspan_threshold,
                "std": std_threshold,
            }
            for check in ("qspan", "std"):
                if values[check] <= thresholds[check]:
                    findings.append(
                        Finding(
                            field=field,
                            dimension=dimension,
                            check=check,
                            value=values[check],
                            mean=float(mean),
                            std=float(std),
                            q01=float(q01),
                            q99=float(q99),
                        )
                    )
    return findings


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Explicit norm_stats.json files. Defaults to the four PrimeBot tasks.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=DEFAULT_ASSETS_ROOT,
        help="Directory containing the four PrimeBot task asset directories.",
    )
    parser.add_argument("--qspan-threshold", type=float, default=1.0e-6)
    parser.add_argument("--std-threshold", type=float, default=1.0e-6)
    return parser.parse_args(argv)


def _classify_findings(items: list[Finding]) -> tuple[int, str]:
    checks = {finding.check for finding in items}
    if checks == {"qspan"}:
        return 0, "CRITICAL"
    if checks == {"qspan", "std"}:
        return 1, "CONSTANT"
    return 2, "STD_ONLY"


def _print_findings(path: Path, findings: list[Finding]) -> None:
    print(f"\n{path}")
    if not findings:
        print("  PASS")
        return

    dimensions: dict[tuple[str, int], list[Finding]] = {}
    for finding in findings:
        dimensions.setdefault((finding.field, finding.dimension), []).append(finding)

    rows = []
    for (field, dimension), items in dimensions.items():
        priority, severity = _classify_findings(items)
        rows.append((priority, field, dimension, severity, items))

    for _, field, dimension, severity, items in sorted(rows):
        item = items[0]
        checks = ",".join(finding.check for finding in items)
        qspan = item.q99 - item.q01
        print(
            f"  {severity} {field} dim={dimension} checks={checks} "
            f"mean={item.mean:.9g} std={item.std:.9g} "
            f"q01={item.q01:.9g} q99={item.q99:.9g} qspan={qspan:.9g}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the norm-stat diagnostics and return a process exit status."""
    args = _parse_args(argv)
    paths = args.paths or [
        args.assets_root / task / "norm_stats.json" for task in DEFAULT_TASKS
    ]

    all_findings: dict[tuple[Path, str, int], list[Finding]] = {}
    for path in paths:
        findings = analyze_norm_stats(
            path,
            qspan_threshold=args.qspan_threshold,
            std_threshold=args.std_threshold,
        )
        _print_findings(path, findings)
        for finding in findings:
            key = (path, finding.field, finding.dimension)
            all_findings.setdefault(key, []).append(finding)

    severity_counts = {"CRITICAL": 0, "CONSTANT": 0, "STD_ONLY": 0}
    for items in all_findings.values():
        _, severity = _classify_findings(items)
        severity_counts[severity] += 1
    checks = sum(len(items) for items in all_findings.values())
    print(
        f"\nSUMMARY files={len(paths)} dimensions={len(all_findings)} "
        f"critical={severity_counts['CRITICAL']} "
        f"constant={severity_counts['CONSTANT']} "
        f"std_only={severity_counts['STD_ONLY']} checks={checks}"
    )
    return int(bool(all_findings))


if __name__ == "__main__":
    raise SystemExit(main())
