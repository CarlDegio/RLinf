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

import pytest
import torch

from rlinf.hybrid_engines.fsdp import utils as fsdp_utils


def test_compile_fsdp_model_compiles_in_place_when_enabled():
    model = torch.nn.Linear(4, 2)
    model_id = id(model)
    inputs = torch.randn(3, 4)
    expected = model(inputs)
    fsdp_config = {
        "strategy": "fsdp",
        "use_orig_params": True,
        "torch_compile": {
            "enabled": True,
            "backend": "eager",
            "fullgraph": False,
            "dynamic": False,
        },
    }

    compiled_model = fsdp_utils.compile_fsdp_model(model, fsdp_config)

    assert id(compiled_model) == model_id
    assert model._compiled_call_impl is not None
    torch.testing.assert_close(compiled_model(inputs), expected)


def test_compile_fsdp_model_requires_orig_params_for_fsdp1():
    model = torch.nn.Linear(4, 2)
    fsdp_config = {
        "strategy": "fsdp",
        "use_orig_params": False,
        "torch_compile": {"enabled": True},
    }

    with pytest.raises(ValueError, match="use_orig_params=true"):
        fsdp_utils.compile_fsdp_model(model, fsdp_config)
