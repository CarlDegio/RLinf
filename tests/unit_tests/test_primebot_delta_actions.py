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

"""Numerical contracts for PrimeBot joint-delta supervision."""

import json
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from openpi import transforms
from openpi.shared.normalize import NormStats, save

from rlinf.models.embodiment.openpi.dataconfig import primebot_dataconfig
from rlinf.models.embodiment.openpi.policies.primebot_policy import (
    PrimeBotInputs,
    PrimeBotOutputs,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model import pi0
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi_rlinf.transforms_pipeline import (
    build_openpi_transforms,
)


def test_joint_delta_uses_one_observation_for_the_whole_chunk() -> None:
    """Subtract observation joints, preserving wheel commands and input arrays."""
    state = np.arange(89, dtype=np.float32)
    actions = np.stack([state + 2.0, state + 7.0])
    original = actions.copy()
    item = {
        "images": {
            key: np.zeros((4, 4, 3), dtype=np.uint8)
            for key in PrimeBotInputs.expected_cameras
        },
        "state": state,
        "actions": actions,
    }

    result = PrimeBotInputs(action_space="joint_delta")(item)

    np.testing.assert_array_equal(result["actions"][0, :22], np.full(22, 2.0))
    np.testing.assert_array_equal(result["actions"][1, :22], np.full(22, 7.0))
    np.testing.assert_array_equal(result["actions"][:, 22:], original[:, 83:86])
    np.testing.assert_array_equal(actions, original)
    restored = PrimeBotOutputs(action_space="joint_delta")(
        {"actions": result["actions"], "reference_state": state}
    )
    np.testing.assert_array_equal(restored["actions"][:, :22], original[:, :22])
    np.testing.assert_array_equal(restored["actions"][:, 22:], original[:, 83:86])


def test_joint_delta_outputs_require_raw_reference_state() -> None:
    """Do not silently export deltas as absolute joint positions."""
    with pytest.raises(ValueError, match="reference_state"):
        PrimeBotOutputs(action_space="joint_delta")(
            {"actions": np.zeros((30, 32), dtype=np.float32)}
        )


def test_delta_config_normalizes_after_subtracting_raw_state(tmp_path, monkeypatch):
    """Exercise YAML -> data config -> delta -> selective norm -> model padding."""

    class LocalTokenizer:
        def __init__(self, max_len):
            self.max_len = max_len

        def tokenize(self, prompt, state):
            assert state is None
            return np.zeros(self.max_len, dtype=np.int64), np.ones(self.max_len, bool)

    monkeypatch.setattr(
        primebot_dataconfig._tokenizer, "PaligemmaTokenizer", LocalTokenizer
    )
    config_dir = Path(__file__).resolve().parents[2] / "examples/sft/config"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name="primebot_sft_openpi_pi05_task03_delta")
    model_cfg = cfg.actor.model
    model_cfg.openpi.assets_dir = str(tmp_path)
    asset_dir = tmp_path / model_cfg.openpi.asset_id
    save(
        asset_dir,
        {
            "state": NormStats(
                mean=np.zeros(89),
                std=np.ones(89),
                q01=np.full(89, -100.0),
                q99=np.full(89, 100.0),
            ),
            "actions": NormStats(
                mean=np.zeros(25),
                std=np.array([0.009] + [1.0] * 24),
                q01=np.full(25, -2.0),
                q99=np.full(25, 2.0),
            ),
        },
    )
    inputs, _ = build_openpi_transforms(
        str(tmp_path),
        model_cfg.openpi.config_name,
        data_kwargs=OmegaConf.to_container(model_cfg.openpi_data, resolve=True),
    )
    state = np.full(89, 10.0, dtype=np.float32)
    actions = np.full((30, 25), 11.0, dtype=np.float32)
    actions[:, 0] = 10.5
    result = transforms.compose(inputs)(
        {
            "state": state,
            "actions": actions,
            "prompt": "put items in the washer",
            "images": {
                key: np.zeros((4, 4, 3), np.uint8)
                for key in PrimeBotInputs.expected_cameras
            },
        }
    )

    np.testing.assert_allclose(result["actions"][:, :22], 0.5, atol=1e-6)
    assert result["actions"].shape == (30, 32)
    assert result["tokenized_prompt"].shape == (256,)
    np.testing.assert_array_equal(result["actions"][:, 25:], 0.0)


def test_flow_loss_excludes_wheels_and_padding_from_value_and_gradient(
    monkeypatch,
) -> None:
    """The real Pi0 loss must average over 22 supervised dimensions only."""
    prediction = torch.cat(
        [torch.full((1, 2, 22), 2.0), torch.full((1, 2, 10), 100.0)], dim=-1
    ).requires_grad_()
    tokens = torch.zeros(1, 1, 32)
    mask = torch.ones(1, 1, dtype=torch.bool)
    ar_mask = torch.zeros(1, dtype=torch.bool)
    core = types.SimpleNamespace(
        embed_dtype=torch.float32,
        action_horizon=2,
        loss_action_dim=22,
        embed_prefix=lambda _obs: (tokens, mask, ar_mask),
        embed_suffix=lambda *_args: (tokens, mask, ar_mask, None),
        llm=lambda *_args, **_kwargs: ((None, prediction), None),
        action_out_proj=torch.nn.Identity(),
    )
    monkeypatch.setattr(pi0.model, "preprocess_observation", lambda obs, **_kwargs: obs)
    monkeypatch.setattr(pi0.model, "_observation_to_dtype", lambda obs, _dtype: obs)
    actions = torch.zeros(1, 2, 32)

    loss = pi0.Pi0.compute_loss(
        core,
        object(),
        actions,
        noise=torch.zeros_like(actions),
        time=torch.tensor([0.5]),
    )
    loss.mean().backward()

    torch.testing.assert_close(loss, torch.full((1, 2), 4.0))
    torch.testing.assert_close(
        prediction.grad[..., :22], torch.full((1, 2, 22), 4.0 / 44)
    )
    assert torch.count_nonzero(prediction.grad[..., 22:]) == 0


@pytest.mark.parametrize("loss_action_dim", [0, -1, 33, 2.5])
def test_loss_dimension_must_be_a_valid_prefix(loss_action_dim) -> None:
    with pytest.raises(ValueError, match="loss_action_dim"):
        Pi0Config(action_dim=32, loss_action_dim=loss_action_dim)


def test_delta_norm_covers_horizon_tail_padding_and_train_split(tmp_path) -> None:
    """Catch same-frame-only stats, crossing episodes, and held-out leakage."""
    from toolkits.lerobot.calculate_primebot_delta_norm_stats import (
        compute_delta_norm_stats,
    )

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(
        json.dumps({"data_path": "episode_{episode_index}.parquet", "chunks_size": 1})
    )
    episodes = []
    # Last episode has enormous values and must not enter either statistic.
    for index, (positions, commands) in enumerate(
        [([10, 20, 30], [11, 23, 36]), ([100, 200], [101, 202]), ([1e6], [-1e6])]
    ):
        states = np.zeros((len(positions), 89), dtype=np.float32)
        actions = np.zeros_like(states)
        states[:, 0] = positions
        actions[:, 0] = commands
        actions[:, 83] = np.arange(1, len(positions) + 1)
        pq.write_table(
            pa.table(
                {"observation.state": states.tolist(), "action": actions.tolist()}
            ),
            tmp_path / f"episode_{index}.parquet",
        )
        episodes.append(json.dumps({"episode_index": index, "length": len(states)}))
    (tmp_path / "meta/episodes.jsonl").write_text("\n".join(episodes) + "\n")

    stats, manifest = compute_delta_norm_stats(
        tmp_path, action_horizon=2, eval_episodes_per_task=1, frames_per_batch=1
    )

    expected_delta = np.array([1, 13, 3, 16, 6, 6, 1, 102, 2, 2])
    np.testing.assert_allclose(stats["actions"].mean[0], 15.2)
    np.testing.assert_allclose(stats["actions"].std[0], expected_delta.std())
    np.testing.assert_allclose(stats["state"].mean[0], 72.0)
    np.testing.assert_allclose(stats["actions"].mean[22], 2.1)
    assert stats["actions"].mean.shape == (25,)
    assert stats["state"].mean.shape == (89,)
    assert manifest["training_frames"] == 5
    assert manifest["action_vectors"] == 10
    assert manifest["training_episodes"] == 2
