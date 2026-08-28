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

"""PrimeBot input/output transforms for OpenPI policies."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np
from openpi import transforms

RAW_ACTION_DIM = 89
COMPACT_ACTION_DIM = 25
JOINT_POSITION_SLICE = slice(0, 22)
WHEEL_VELOCITY_SLICE = slice(83, 86)


def pack_primebot_action(action: np.ndarray) -> np.ndarray:
    """Pack the 25 controlled PrimeBot dimensions from an 89-D raw action."""
    action = np.asarray(action)
    if action.shape[-1] != RAW_ACTION_DIM:
        raise ValueError(
            f"Expected a {RAW_ACTION_DIM}-D PrimeBot action, got {action.shape}."
        )
    return np.concatenate(
        [action[..., JOINT_POSITION_SLICE], action[..., WHEEL_VELOCITY_SLICE]],
        axis=-1,
    )


def unpack_primebot_action(action: np.ndarray) -> np.ndarray:
    """Expand a compact 25-D policy action into the raw 89-D command layout."""
    action = np.asarray(action)
    if action.shape[-1] < COMPACT_ACTION_DIM:
        raise ValueError(
            "Expected at least 25 compact PrimeBot action dimensions, got "
            f"{action.shape}."
        )
    raw = np.zeros((*action.shape[:-1], RAW_ACTION_DIM), dtype=action.dtype)
    raw[..., JOINT_POSITION_SLICE] = action[..., :22]
    raw[..., WHEEL_VELOCITY_SLICE] = action[..., 22:25]
    return raw


def _as_uint8_hwc(image: np.ndarray) -> np.ndarray:
    """Convert an HWC/CHW image to uint8 HWC without resizing it."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a rank-3 image, got {image.shape}.")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three image channels, got {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size and float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class PrimeBotInputs(transforms.DataTransformFn):
    """Convert PrimeBot observations to the canonical three-camera Pi0 input."""

    expected_cameras: ClassVar[tuple[str, ...]] = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        missing = set(self.expected_cameras) - set(images)
        extra = set(images) - set(self.expected_cameras)
        if missing or extra:
            raise ValueError(
                "PrimeBot images must contain exactly "
                f"{self.expected_cameras}; missing={sorted(missing)}, "
                f"extra={sorted(extra)}."
            )

        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != RAW_ACTION_DIM:
            raise ValueError(f"PrimeBot state must be 89-D, got {state.shape}.")
        result = {
            "image": {
                name: _as_uint8_hwc(images[name]) for name in self.expected_cameras
            },
            "image_mask": dict.fromkeys(self.expected_cameras, np.True_),
            # Keep all 89 state dimensions here. Pi0.5 normalizes and tokenizes
            # this value before the model-shape transform replaces the continuous
            # placeholder with 32 dimensions.
            "state": state,
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] == RAW_ACTION_DIM:
                actions = pack_primebot_action(actions)
            elif actions.shape[-1] != COMPACT_ACTION_DIM:
                raise ValueError(
                    "PrimeBot actions must be raw 89-D or compact 25-D, got "
                    f"{actions.shape}."
                )
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class PrimeBotOutputs(transforms.DataTransformFn):
    """Return the 25 controlled action dimensions after unnormalization."""

    def __call__(self, data: dict) -> dict:
        return {
            **data,
            "actions": np.asarray(data["actions"])[..., :COMPACT_ACTION_DIM],
        }


@dataclasses.dataclass(frozen=True)
class CropContinuousStateAfterTokenization(transforms.DataTransformFn):
    """Crop the continuous state after constructing the language prompt.

    Pi0.5 does not consume ``Observation.state`` in its action suffix, but the
    shared model interface still requires its final dimension to equal the
    model action dimension (32). PrimeBot does not serialize state into the
    prompt, so this value remains only as a structural placeholder.
    """

    model_action_dim: int

    def __call__(self, data: dict) -> dict:
        if "tokenized_prompt" not in data:
            raise ValueError(
                "PrimeBot continuous state may only be cropped after prompt "
                "tokenization."
            )
        state = np.asarray(data["state"])
        if state.shape[-1] < self.model_action_dim:
            raise ValueError(
                "PrimeBot state unexpectedly has fewer dimensions than the model: "
                f"state={state.shape}, model_action_dim={self.model_action_dim}."
            )
        data["state"] = state[..., : self.model_action_dim]
        return data
