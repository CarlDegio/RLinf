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

"""Tests for the PrimeBot Pi0.5 state/action contract."""

import pathlib
import types

import numpy as np
import torch
from openpi.models.model import ModelType
from openpi.training.config import DataConfig

from rlinf.models.embodiment.openpi.dataconfig import primebot_dataconfig
from rlinf.models.embodiment.openpi.policies.primebot_policy import (
    CropContinuousStateAfterTokenization,
    PrimeBotInputs,
    pack_primebot_action,
    unpack_primebot_action,
)
from rlinf.models.embodiment.openpi_rlinf.sft_action_model import (
    OpenPiPytorchSFTActionModel,
)


def test_primebot_action_pack_and_unpack() -> None:
    raw = np.arange(2 * 89, dtype=np.float32).reshape(2, 89)
    compact = pack_primebot_action(raw)
    np.testing.assert_array_equal(compact[:, :22], raw[:, :22])
    np.testing.assert_array_equal(compact[:, 22:25], raw[:, 83:86])

    restored = unpack_primebot_action(compact)
    np.testing.assert_array_equal(restored[:, :22], raw[:, :22])
    np.testing.assert_array_equal(restored[:, 83:86], raw[:, 83:86])
    assert np.count_nonzero(restored[:, 22:83]) == 0
    assert np.count_nonzero(restored[:, 86:]) == 0


def test_primebot_inputs_preserve_89d_state_and_pack_actions() -> None:
    item = {
        "images": {
            "base_0_rgb": np.zeros((8, 12, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.zeros((3, 8, 12), dtype=np.float32),
            "right_wrist_0_rgb": np.zeros((8, 12, 3), dtype=np.uint8),
        },
        "state": np.arange(89, dtype=np.float32),
        "actions": np.zeros((30, 89), dtype=np.float32),
        "prompt": "do the task",
    }
    transformed = PrimeBotInputs()(item)
    assert transformed["state"].shape == (89,)
    assert transformed["actions"].shape == (30, 25)
    assert transformed["image"]["left_wrist_0_rgb"].shape == (8, 12, 3)


def test_continuous_state_is_cropped_after_tokens_are_present() -> None:
    data = {
        "state": np.arange(89, dtype=np.float32),
        "tokenized_prompt": np.arange(256),
    }
    transformed = CropContinuousStateAfterTokenization(32)(data)
    assert transformed["state"].shape == (32,)
    assert transformed["tokenized_prompt"].shape == (256,)


def test_primebot_prompt_uses_256_tokens_without_state(monkeypatch) -> None:
    """The PrimeBot language prefix must not serialize the robot state."""
    received = {}

    class FakeTokenizer:
        def __init__(self, max_len):
            received["max_len"] = max_len

        def tokenize(self, prompt, state):
            received["prompt"] = prompt
            received["state"] = state
            return np.zeros(256, dtype=np.int64), np.ones(256, dtype=bool)

    monkeypatch.setattr(
        primebot_dataconfig._tokenizer, "PaligemmaTokenizer", FakeTokenizer
    )
    monkeypatch.setattr(
        primebot_dataconfig.LeRobotPrimeBotDataConfig,
        "create_base_config",
        lambda self, assets_dirs, model_config: DataConfig(),
    )
    model_config = types.SimpleNamespace(
        model_type=ModelType.PI05,
        discrete_state_input=False,
        max_token_len=256,
        action_dim=32,
    )

    data_config = primebot_dataconfig.LeRobotPrimeBotDataConfig().create(
        pathlib.Path("unused"), model_config
    )
    tokenize = data_config.model_transforms.inputs[2]
    transformed = tokenize(
        {"prompt": "open the washer", "state": np.arange(89, dtype=np.float32)}
    )

    assert received == {
        "max_len": 256,
        "prompt": "open the washer",
        "state": None,
    }
    assert transformed["tokenized_prompt"].shape == (256,)


def test_eval_sft_forward_returns_per_sample_action_loss() -> None:
    class FakeFlowModel:
        def __init__(self) -> None:
            self.train = None

        def compute_loss(self, observation, actions, *, train):
            del observation, actions
            self.train = train
            return torch.tensor([[1.0, 3.0], [2.0, 6.0]])

    class FakeSftModel:
        rlt_cfg = type("RltConfig", (), {"use_rlt": False})()
        model = FakeFlowModel()

        @staticmethod
        def _unpack_sft_batch(data):
            del data
            return "observation", "actions"

        @staticmethod
        def _observation_to_device(observation):
            return observation

        @staticmethod
        def _actions_to_device(actions):
            return actions

    fake = FakeSftModel()
    output = OpenPiPytorchSFTActionModel.sft_forward(
        fake,
        data=None,
        train=False,
        return_per_sample_loss=True,
    )

    torch.testing.assert_close(output["per_sample_loss"], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(output["loss"], torch.tensor(3.0))
    assert fake.model.train is False


def test_eval_sft_forward_returns_denormalized_25d_action_chunks() -> None:
    """The MSE path must sample actions and apply the deployment output scale."""

    class FakeFlowModel:
        action_dim = 32

        @staticmethod
        def sample_actions(observation, *, num_steps, rng):
            del observation, rng
            assert num_steps == 5
            values = torch.arange(30 * 32, dtype=torch.float32)
            return values.reshape(1, 30, 32)

    class FakeSftModel:
        model = FakeFlowModel()
        num_steps = 5

        @staticmethod
        def _unpack_sft_batch(data):
            del data
            return "observation", torch.zeros(1, 30, 32)

        @staticmethod
        def _observation_to_device(observation):
            return observation

        @staticmethod
        def _actions_to_device(actions):
            return actions

        @staticmethod
        def denormalize_actions(actions):
            return actions[..., :25] + 100.0

    output = OpenPiPytorchSFTActionModel.sft_forward(
        FakeSftModel(),
        data=None,
        train=False,
        return_denormalized_actions=True,
        rng=torch.Generator().manual_seed(42),
    )

    assert output["predicted_actions"].shape == (1, 30, 25)
    torch.testing.assert_close(
        output["predicted_actions"][0, 0],
        torch.arange(25, dtype=torch.float32) + 100.0,
    )


def test_sft_action_model_applies_output_transform_per_sample() -> None:
    """Sampled normalized actions must use the deployment inverse transform."""

    def output_transform(data):
        return {"actions": np.asarray(data["actions"])[..., :25] * 2.0 + 1.0}

    fake = types.SimpleNamespace()
    OpenPiPytorchSFTActionModel.setup_output_transform(fake, [output_transform])
    normalized = torch.zeros(2, 30, 32)

    denormalized = OpenPiPytorchSFTActionModel.denormalize_actions(fake, normalized)

    assert denormalized.shape == (2, 30, 25)
    assert denormalized.dtype == torch.float32
    torch.testing.assert_close(denormalized, torch.ones(2, 30, 25))


def test_sft_action_model_sends_only_bfloat16_actions_to_output_transform() -> None:
    """PI0.5 output transforms must receive FP32 actions without model state."""
    received = []

    def output_transform(data):
        received.append((set(data), data["actions"].dtype))
        return {"actions": data["actions"][..., :25]}

    fake = types.SimpleNamespace()
    OpenPiPytorchSFTActionModel.setup_output_transform(fake, [output_transform])
    normalized = torch.zeros(1, 30, 32, dtype=torch.bfloat16)

    denormalized = OpenPiPytorchSFTActionModel.denormalize_actions(fake, normalized)

    assert received == [({"actions"}, np.dtype(np.float32))]
    assert denormalized.dtype == torch.float32
