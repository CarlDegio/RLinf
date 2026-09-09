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

"""Export complete official PrimeBot trajectories using four specialist policies.

Only instruction_segments controls routing. Hidden action targets are never read.
Each GPU owns an independent model replica; no Ray, FSDP, or optimizer is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
INSTRUCTIONS = {
    "task01": "Use the gripper to fully open the washing machine door.",
    "task02": "Close the washing machine door tightly with the gripper.",
    "task03": "Put these two pieces of clothing into the washer.",
    "task09": "Unfold the clothing and fold it neatly.",
}
CAMERAS = {
    "base_0_rgb": "observation.images.x2w_camera_head_realsense_compressed",
    "left_wrist_0_rgb": "observation.images.x2w_camera_left_wrist_zedxonegs_rgb_raw_image_compressed",
    "right_wrist_0_rgb": "observation.images.x2w_camera_right_wrist_zedxonegs_rgb_raw_image_compressed",
}
ACTION_SCHEMA = pa.schema(
    [
        ("episode_index", pa.int64()),
        ("frame_index", pa.int64()),
        ("index", pa.int64()),
        ("action", pa.list_(pa.float32(), 25)),
    ]
)


def normalize_segments(segments: list[dict], length: int) -> list[dict]:
    """Validate half-open full coverage and merge touching identical instructions."""
    result = []
    cursor = 0
    for segment in sorted(segments, key=lambda s: s["start_frame_index"]):
        start, end = segment["start_frame_index"], segment["end_frame_index"]
        instruction = segment["instruction"]
        if (
            type(start) is not int
            or type(end) is not int
            or not isinstance(instruction, str)
            or start != cursor
            or not 0 <= start <= end <= length
        ):
            raise ValueError(f"Invalid annotation coverage at {cursor}: {segment}")
        cursor = end
        if start == end:
            continue
        if result and result[-1]["instruction"] == instruction:
            result[-1]["end"] = end
        else:
            result.append({"start": start, "end": end, "instruction": instruction})
    if cursor != length:
        raise ValueError(f"Annotations cover {cursor} frames, expected {length}")
    return result


def build_chunks(segments: list[dict], task: str, size: int) -> list[tuple[int, int]]:
    """Restart the execution horizon at every distinct instruction boundary."""
    if not 0 < size <= 30:
        raise ValueError("chunk_size must be between 1 and 30")
    return [
        (start, min(size, s["end"] - start))
        for s in segments
        if s["instruction"] == INSTRUCTIONS[task]
        for start in range(s["start"], s["end"], size)
    ]


def task_mask(episode: dict, task: str) -> np.ndarray:
    """Return the exact scored frames owned by one specialist."""
    mask = np.zeros(episode["length"], dtype=bool)
    for segment in episode["segments"]:
        if segment["instruction"] == INSTRUCTIONS[task]:
            mask[segment["start"] : segment["end"]] = True
    return mask


def select_latest_checkpoint(root: Path) -> Path:
    """Choose the numerically largest step with a nonempty full weights file."""
    candidates = [
        p
        for p in root.glob("global_step_*/actor/model_state_dict/full_weights.pt")
        if p.stat().st_size > 0 and p.parents[2].name[12:].isdigit()
    ]
    if not candidates:
        raise FileNotFoundError(f"No full checkpoint under {root}")
    return max(candidates, key=lambda p: int(p.parents[2].name[12:])).parents[1]


def file_identity(path: Path) -> dict:
    """Fingerprint large immutable files without rereading all checkpoint bytes."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_manifest(config: dict, smoke: bool) -> dict:
    """Inspect every selected parquet and annotation before launching GPUs."""
    if set(config["tasks"]) != set(INSTRUCTIONS):
        raise ValueError("Configure exactly task01, task02, task03, and task09")
    if config["model"]["openpi"]["discrete_state_input"]:
        raise ValueError("This exporter requires PI0.5 without state input")
    build_chunks([], "task01", config["chunk_size"])
    episodes = []
    for name in config["datasets"]:
        root = Path(config["dataset_root"]) / name
        info = json.loads((root / "meta/info.json").read_text())
        if info["fps"] != 30:
            raise ValueError(f"Expected 30 FPS at {root}")
        annotations = info["instruction_segments"]
        entries = [
            json.loads(line)
            for line in (root / "meta/episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        selected = []
        for entry in sorted(entries, key=lambda e: e["episode_index"]):
            index, length = entry["episode_index"], entry["length"]
            args = {
                "episode_index": index,
                "episode_chunk": index // info["chunks_size"],
            }
            parquet = root / info["data_path"].format(**args)
            ids = pq.read_table(
                parquet, columns=["episode_index", "frame_index", "index"]
            )
            if (
                len(ids) != length
                or length <= 0
                or not np.all(ids["episode_index"].to_numpy() == index)
                or not np.array_equal(ids["frame_index"].to_numpy(), np.arange(length))
            ):
                raise ValueError(f"Invalid episode/frame IDs: {parquet}")
            videos = {
                key: str(root / info["video_path"].format(video_key=value, **args))
                for key, value in CAMERAS.items()
            }
            selected.append(
                {
                    "dataset": name,
                    "episode_index": index,
                    "length": length,
                    "parquet": str(parquet),
                    "relative": f"chunk-{args['episode_chunk']:03d}/{parquet.name}",
                    "segments": normalize_segments(annotations[str(index)], length),
                    "videos": videos,
                    "source_files": [
                        file_identity(parquet),
                        *[file_identity(Path(p)) for p in videos.values()],
                    ],
                }
            )
        if (
            len(selected) != info["total_episodes"]
            or sum(e["length"] for e in selected) != info["total_frames"]
        ):
            raise ValueError(f"Metadata totals do not match files: {root}")
        if smoke:
            # Choose complete episodes covering all available target instructions.
            chosen = {
                next(
                    (e["episode_index"] for e in selected if task_mask(e, t).any()),
                    None,
                )
                for t in INSTRUCTIONS
            }
            selected = [e for e in selected if e["episode_index"] in chosen]
        episodes.extend(selected)
    identities = {}
    for task, values in config["tasks"].items():
        root = ROOT / values["checkpoint_root"]
        actor = (
            Path(values["checkpoint"]).resolve()
            if values.get("checkpoint")
            else select_latest_checkpoint(root)
        )
        stats = (
            Path(config["model"]["openpi"]["assets_dir"])
            / values["asset_id"]
            / "norm_stats.json"
        )
        values["checkpoint"] = str(actor)
        values["norm_stats_path"] = str(stats)
        identities[task] = {
            "weights": file_identity(actor / "model_state_dict/full_weights.pt"),
            "norm_stats_sha256": hashlib.sha256(stats.read_bytes()).hexdigest(),
        }
    return {
        "version": 1,
        "smoke_test": smoke,
        "config": config,
        "models": identities,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "episodes": episodes,
    }


def partial_path(out: Path, task: str, episode: dict) -> Path:
    return (
        out
        / "partials"
        / task
        / episode["dataset"]
        / f"episode_{episode['episode_index']:06d}.npz"
    )


def validate_partial(
    actions: np.ndarray, mask: np.ndarray, episode: dict, task: str
) -> None:
    if (
        actions.shape != (episode["length"], 25)
        or actions.dtype != np.float32
        or not np.isfinite(actions).all()
        or mask.dtype != np.bool_
        or not np.array_equal(mask, task_mask(episode, task))
        or np.any(actions[~mask] != 0)
    ):
        raise ValueError(
            f"Invalid/incomplete predictions: {task}, {episode['episode_index']}"
        )


def save_partial(
    out: Path, task: str, episode: dict, actions: np.ndarray, mask: np.ndarray
) -> None:
    """Publish one complete episode/task atomically for safe restart."""
    validate_partial(actions, mask, episode, task)
    path = partial_path(out, task, episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, actions=actions, mask=mask)
    temporary.replace(path)


def read_partial(out: Path, task: str, episode: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(partial_path(out, task, episode), allow_pickle=False) as data:
        actions, mask = data["actions"], data["mask"]
    validate_partial(actions, mask, episode, task)
    return actions, mask


def export_episode(out: Path, episode: dict) -> dict:
    """Merge specialist predictions, leaving every non-scored row exactly zero."""
    actions = np.zeros((episode["length"], 25), dtype=np.float32)
    covered = np.zeros(episode["length"], dtype=bool)
    by_task = {}
    for task in INSTRUCTIONS:
        if task_mask(episode, task).any():
            predicted, mask = read_partial(out, task, episode)
            if np.any(covered & mask):
                raise ValueError("Overlapping specialist outputs")
            actions[mask] = predicted[mask]
            covered |= mask
            by_task[task] = int(mask.sum())
    source = pq.read_table(
        episode["parquet"], columns=["episode_index", "frame_index", "index"]
    )
    table = pa.Table.from_arrays(
        [
            *[
                source[key].cast(pa.int64())
                for key in ("episode_index", "frame_index", "index")
            ],
            pa.FixedSizeListArray.from_arrays(pa.array(actions.ravel()), 25),
        ],
        schema=ACTION_SCHEMA,
    )
    path = out / "actions" / episode["dataset"] / episode["relative"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pq.write_table(table, temporary)
    check = pq.read_table(temporary)
    if not check.equals(table):
        raise ValueError(f"Parquet roundtrip failed: {path}")
    temporary.replace(path)
    return {
        "dataset": episode["dataset"],
        "episode_index": episode["episode_index"],
        "frames": len(table),
        "predicted_frames": int(covered.sum()),
        "zero_filled_frames": int((~covered).sum()),
        "frames_by_task": by_task,
    }


def sample_episode(
    model: Any, transform: Any, episode: dict, task: str, config: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Decode recorded video sequentially and predict only instruction-aligned chunks."""
    import av
    import torch

    from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation

    chunks = dict(build_chunks(episode["segments"], task, config["chunk_size"]))
    actions = np.zeros((episode["length"], 25), dtype=np.float32)
    covered = np.zeros(episode["length"], dtype=bool)
    with ExitStack() as stack:
        containers = {
            key: stack.enter_context(av.open(path))
            for key, path in episode["videos"].items()
        }
        for container in containers.values():
            container.streams.video[0].codec_context.thread_count = config[
                "cpu_threads"
            ]
        decoders = {
            key: iter(container.decode(video=0))
            for key, container in containers.items()
        }
        for frame in range(max(chunks) + 1):
            try:
                decoded = {key: next(decoder) for key, decoder in decoders.items()}
            except StopIteration as exc:
                raise ValueError(
                    f"Video shorter than required frame {frame}: {episode['parquet']}"
                ) from exc
            if frame not in chunks:
                continue
            data = transform(
                {
                    "images": {
                        key: value.to_ndarray(format="rgb24")
                        for key, value in decoded.items()
                    },
                    "state": np.zeros(89, np.float32),
                    "prompt": INSTRUCTIONS[task],
                }
            )

            def tensor(value):
                return (
                    torch.from_numpy(np.asarray(value).copy()).unsqueeze(0).to("cuda")
                )

            observation = Observation.from_dict(
                {
                    "image": {
                        key: tensor(value) for key, value in data["image"].items()
                    },
                    "image_mask": {
                        key: tensor(value) for key, value in data["image_mask"].items()
                    },
                    "state": tensor(data["state"]),
                    "tokenized_prompt": tensor(data["tokenized_prompt"]).long(),
                    "tokenized_prompt_mask": tensor(data["tokenized_prompt_mask"]),
                }
            )
            # Seed per chunk: GPU count, scheduling and resume do not change noise.
            identity = f"{config['seed']}:{task}:{episode['dataset']}:{episode['episode_index']}:{frame}"
            seed = int.from_bytes(
                hashlib.sha256(identity.encode()).digest()[:8], "little"
            ) % (2**63)
            rng = torch.Generator(device="cuda").manual_seed(seed)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.model.sample_actions(
                    observation, num_steps=model.num_steps, rng=rng
                )
                prediction = (
                    model.denormalize_actions(prediction)[0].float().cpu().numpy()
                )
            valid = chunks[frame]
            if prediction.shape != (30, 25) or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid model prediction: {prediction.shape}")
            actions[frame : frame + valid] = prediction[:valid]
            covered[frame : frame + valid] = True
    return actions, covered


def run_worker(out: Path, task: str, rank: int, world: int) -> None:
    """Load one BF16 model on the single visible GPU and process its episode shard."""
    import torch
    from openpi.transforms import compose

    from rlinf.models.embodiment.openpi_rlinf import get_model
    from rlinf.models.embodiment.openpi_rlinf.transforms_pipeline import (
        build_openpi_transforms,
    )

    manifest = json.loads((out / "manifest.json").read_text())
    config = manifest["config"]
    episodes = [e for e in manifest["episodes"] if task_mask(e, task).any()][
        rank::world
    ]
    pending = []
    for episode in episodes:
        if partial_path(out, task, episode).exists():
            read_partial(out, task, episode)
        else:
            pending.append(episode)
    if not pending:
        LOGGER.info("%s rank %s: all assigned episodes already complete", task, rank)
        return
    torch.set_num_threads(config["cpu_threads"])
    torch.cuda.set_device(0)
    model_config = OmegaConf.create(config["model"])
    selected = config["tasks"][task]
    model_config.model_path = selected["checkpoint"]
    model_config.openpi.asset_id = selected["asset_id"]
    model_config.openpi_data.norm_stats_path = selected["norm_stats_path"]
    model = (
        get_model(model_config, torch_dtype=torch.bfloat16)
        .eval()
        .requires_grad_(False)
        .to("cuda")
    )
    inputs, _ = build_openpi_transforms(
        model_config.model_path,
        model_config.openpi.config_name,
        data_kwargs=OmegaConf.to_container(model_config.openpi_data, resolve=True),
    )
    transform = compose(inputs)
    LOGGER.info(
        "Loaded %s: GPU allocated %.2f GiB; %s pending episodes",
        task,
        torch.cuda.memory_allocated() / 2**30,
        len(pending),
    )
    for number, episode in enumerate(pending, 1):
        actions, mask = sample_episode(model, transform, episode, task, config)
        save_partial(out, task, episode, actions, mask)
        LOGGER.info(
            "%s rank %s: %s/%s %s episode %s, %s predicted frames",
            task,
            rank,
            number,
            len(pending),
            episode["dataset"],
            episode["episode_index"],
            mask.sum(),
        )


def launch_workers(out: Path, manifest: dict, gpus: list[str]) -> None:
    """Run task groups sequentially, cleaning up only this launcher's children."""
    config = manifest["config"]
    for task in INSTRUCTIONS:
        total = sum(task_mask(e, task).any() for e in manifest["episodes"])
        world = min(len(gpus), total)
        LOGGER.info("Starting %s: %s episodes on %s GPU workers", task, total, world)
        children = []
        with ExitStack() as stack:
            try:
                for rank in range(world):
                    log = stack.enter_context(
                        (out / f"{task}_worker{rank}.log").open("a")
                    )
                    env = dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=gpus[rank],
                        OMP_NUM_THREADS=str(config["cpu_threads"]),
                        PYTHONPATH=str(ROOT)
                        + os.pathsep
                        + os.environ.get("PYTHONPATH", ""),
                    )
                    children.append(
                        subprocess.Popen(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--output",
                                str(out),
                                "--worker",
                                task,
                                "--rank",
                                str(rank),
                                "--world",
                                str(world),
                            ],
                            env=env,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                    )
                while any(p.poll() is None for p in children):
                    if any(p.poll() not in (None, 0) for p in children):
                        raise RuntimeError(
                            f"{task} worker failed; inspect {out}/{task}_worker*.log"
                        )
                    time.sleep(2)
                if any(p.returncode != 0 for p in children):
                    raise RuntimeError(f"{task} worker failed; inspect its worker log")
            finally:
                for child in children:
                    if child.poll() is None:
                        child.terminate()
                for child in children:
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()


def main() -> None:
    """CLI for preflight, partial resume, smoke inference, and complete export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "examples/sft/config/primebot_validation_export.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--gpus",
        help="Comma-separated GPU IDs/UUIDs; defaults to CUDA_VISIBLE_DEVICES or YAML",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate inputs and record a manifest without loading models",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Select complete episodes covering each available target task",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse verified partials only if the manifest is identical",
    )
    parser.add_argument("--worker", choices=list(INSTRUCTIONS), help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--world", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.worker:
        run_worker(args.output.resolve(), args.worker, args.rank, args.world)
        return
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    gpus = (args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES", config["gpus"])).split(
        ","
    )
    gpus = [g.strip() for g in gpus]
    if not all(gpus) or "-1" in gpus or len(set(gpus)) != len(gpus):
        parser.error("GPU IDs must be nonempty and unique")
    out = (
        args.output
        or ROOT / "logs" / f"primebot_submission_{datetime.now():%Y%m%d-%H%M%S}"
    ).resolve()
    manifest = build_manifest(config, args.smoke_test)
    if out.exists():
        if not args.resume or not (out / "manifest.json").exists():
            parser.error("Output already exists; use a new directory or --resume")
        if json.loads((out / "manifest.json").read_text()) != manifest:
            parser.error(
                "Resume manifest differs: checkpoint, configuration, source files, or script changed"
            )
    else:
        out.mkdir(parents=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    handler = logging.FileHandler(out / "inference.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    LOGGER.info(
        "Output: %s; smoke=%s; episodes=%s; frames=%s",
        out,
        args.smoke_test,
        len(manifest["episodes"]),
        sum(e["length"] for e in manifest["episodes"]),
    )
    for task in INSTRUCTIONS:
        LOGGER.info(
            "%s checkpoint=%s predicted_frames=%s",
            task,
            config["tasks"][task]["checkpoint"],
            sum(int(task_mask(e, task).sum()) for e in manifest["episodes"]),
        )
    if args.plan_only:
        return
    launch_workers(out, manifest, gpus)
    reports = [export_episode(out, e) for e in manifest["episodes"]]
    report = {
        "complete": True,
        "smoke_test": args.smoke_test,
        "episodes": reports,
        "total_episodes": len(reports),
        "total_frames": sum(r["frames"] for r in reports),
        "predicted_frames": sum(r["predicted_frames"] for r in reports),
        "zero_filled_frames": sum(r["zero_filled_frames"] for r in reports),
    }
    (out / "coverage_report.json").write_text(json.dumps(report, indent=2))
    LOGGER.info(
        "Export complete: %s", {k: v for k, v in report.items() if k != "episodes"}
    )


if __name__ == "__main__":
    main()
