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

"""OpenPI data configuration for PrimeBot Household LeRobot v2.1 data."""

from __future__ import annotations

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import primebot_policy


@dataclasses.dataclass(frozen=True)
class LeRobotPrimeBotDataConfig(DataConfigFactory):
    """Configure absolute or observation-relative PrimeBot actions for Pi0.5."""

    use_quantile_norm: bool = True
    action_norm_min_std: float = 0.01
    action_space: str = "absolute"

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        if model_config.model_type != _model.ModelType.PI05:
            raise ValueError(
                "PrimeBot currently supports only the Pi0.5 transform path."
            )
        if self.action_space not in {"absolute", "joint_delta"}:
            raise ValueError(
                f"Unsupported PrimeBot action_space: {self.action_space!r}"
            )
        data_transforms = _transforms.Group(
            inputs=[primebot_policy.PrimeBotInputs(action_space=self.action_space)],
            outputs=[primebot_policy.PrimeBotOutputs(action_space=self.action_space)],
        )
        # This is intentionally explicit instead of ModelTransformFactory: its
        # generic PadStatesAndActions rejects a state wider than action_dim. The
        # crop below happens after prompt tokenization. State is not serialized
        # into the prompt and is retained only until the shared interface can be
        # given its required 32-D structural placeholder.
        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=bool(model_config.discrete_state_input),
                ),
                primebot_policy.CropContinuousStateAfterTokenization(
                    model_config.action_dim
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ]
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=self.use_quantile_norm,
            action_sequence_keys=("action",),
        )
