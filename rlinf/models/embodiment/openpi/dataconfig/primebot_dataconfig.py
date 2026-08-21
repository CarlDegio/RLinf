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
    """Configure absolute-action PrimeBot transforms for Pi0.5."""

    use_quantile_norm: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        if model_config.model_type != _model.ModelType.PI05:
            raise ValueError(
                "PrimeBot currently supports only the Pi0.5 transform path."
            )
        if not getattr(model_config, "discrete_state_input", False):
            raise ValueError(
                "PrimeBot requires Pi0.5 discrete_state_input=True so all 89 state "
                "dimensions are encoded in the prompt."
            )

        data_transforms = _transforms.Group(
            inputs=[primebot_policy.PrimeBotInputs()],
            outputs=[primebot_policy.PrimeBotOutputs()],
        )
        # This is intentionally explicit instead of ModelTransformFactory: its
        # generic PadStatesAndActions rejects a state wider than action_dim. The
        # crop below happens after tokenization, so the tokenizer still receives
        # all 89 normalized state values.
        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=True,
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
