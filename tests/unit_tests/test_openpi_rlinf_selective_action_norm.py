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

"""Tests for action normalization selected once from loaded statistics."""

import types

import numpy as np
from openpi import transforms
from openpi.shared.normalize import NormStats

from rlinf.models.embodiment.openpi_rlinf import transforms_pipeline


def test_low_std_actions_keep_raw_scale_in_both_directions() -> None:
    """Only dimensions strictly below the threshold must bypass scaling."""
    norm_stats = {
        "actions": NormStats(
            mean=np.array([5.0, 10.0, 20.0]),
            std=np.array([0.009, 0.01, 0.02]),
            q01=np.array([0.0, 0.0, 0.0]),
            q99=np.array([10.0, 20.0, 40.0]),
        )
    }
    normalize = transforms_pipeline.NormalizeWithSelectiveActions(
        norm_stats,
        use_quantiles=True,
        action_norm_min_std=0.01,
    )
    unnormalize = transforms_pipeline.UnnormalizeWithSelectiveActions(
        norm_stats,
        use_quantiles=True,
        action_norm_min_std=0.01,
    )

    normalized = normalize({"actions": np.array([[5.0, 10.0, 20.0]])})
    np.testing.assert_allclose(normalized["actions"], [[5.0, 0.0, 0.0]], atol=1.0e-6)

    restored = unnormalize({"actions": np.array([[7.0, 0.0, 0.0]])})
    np.testing.assert_allclose(restored["actions"], [[7.0, 10.0, 20.0]], atol=1.0e-6)


def test_training_actions_are_clipped_before_selective_normalization() -> None:
    """Targets outside q01/q99 must be clipped before either scale path."""
    norm_stats = {
        "actions": NormStats(
            mean=np.array([5.0, 10.0, 20.0]),
            std=np.array([0.009, 0.01, 0.02]),
            q01=np.array([0.0, 0.0, 0.0]),
            q99=np.array([10.0, 20.0, 40.0]),
        )
    }
    normalize = transforms_pipeline.NormalizeWithSelectiveActions(
        norm_stats,
        use_quantiles=True,
        action_norm_min_std=0.01,
    )

    normalized = normalize({"actions": np.array([[-2.0, 30.0, 50.0]])})

    np.testing.assert_allclose(normalized["actions"], [[0.0, 1.0, 1.0]], atol=1.0e-6)


def test_transform_pipeline_uses_primebot_action_std_threshold(monkeypatch) -> None:
    """A configured threshold must select matching input and output transforms."""
    norm_stats = {
        "actions": NormStats(
            mean=np.array([0.0]),
            std=np.array([0.001]),
            q01=np.array([-1.0]),
            q99=np.array([1.0]),
        )
    }
    empty_group = transforms.Group(inputs=[], outputs=[])
    data_config = types.SimpleNamespace(
        norm_stats=norm_stats,
        asset_id="primebot/test",
        use_quantile_norm=True,
        data_transforms=empty_group,
        model_transforms=empty_group,
    )
    data_factory = types.SimpleNamespace(
        action_norm_min_std=0.01,
        create=lambda _assets, _model: data_config,
    )
    train_config = types.SimpleNamespace(
        model=object(),
        data=data_factory,
        assets_dirs="unused",
    )

    import rlinf.models.embodiment.openpi.dataconfig as dataconfig

    monkeypatch.setattr(
        dataconfig, "get_openpi_config", lambda *args, **kwargs: train_config
    )

    input_transforms, output_transforms = transforms_pipeline.build_openpi_transforms(
        "unused", "pi05_primebot"
    )

    assert isinstance(
        input_transforms[1], transforms_pipeline.NormalizeWithSelectiveActions
    )
    assert isinstance(
        output_transforms[0], transforms_pipeline.UnnormalizeWithSelectiveActions
    )
