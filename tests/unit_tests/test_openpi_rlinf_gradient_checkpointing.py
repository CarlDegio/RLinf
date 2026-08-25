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
from omegaconf import OmegaConf
from torch import nn

from rlinf.hybrid_engines.fsdp import fsdp_model_manager
from rlinf.models.embodiment.openpi_rlinf.pi0_model import gemma, siglip
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0


class _CountingLayer(nn.Module):
    """Layer whose invocation count exposes activation recomputation."""

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.calls = 0
        self.linear = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.sin(self.linear(x))


class _CountingGemmaLayer(_CountingLayer):
    def forward(self, xs, kv_cache, positions, mask, adarms_cond):
        del positions, mask, adarms_cond
        return [super().forward(xs[0])], kv_cache


class _IdentityGemmaNorm(nn.Module):
    def forward(self, x, adarms_cond):
        del adarms_cond
        return x, None


def _make_pi0_shell(llm_depth: int = 4, vision_depth: int = 3) -> Pi0:
    model = Pi0.__new__(Pi0)
    nn.Module.__init__(model)
    model.llm = nn.Module()
    model.llm.layers = nn.ModuleList([nn.Identity() for _ in range(llm_depth)])
    model.img = nn.Module()
    model.img.encoder = nn.Module()
    model.img.encoder.layers = nn.ModuleList(
        [nn.Identity() for _ in range(vision_depth)]
    )
    return model


def _make_gemma_shell(depth: int = 3) -> gemma.Module:
    model = gemma.Module.__new__(gemma.Module)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([_CountingGemmaLayer() for _ in range(depth)])
    model.final_norms = nn.ModuleList([_IdentityGemmaNorm()])
    model.configs = [object()]
    model.embed_dtype = torch.float32
    model.gradient_checkpointing = True
    model.gradient_checkpointing_use_reentrant = False
    return model


def test_fsdp_builds_openpi_partial_checkpointing_kwargs() -> None:
    config = OmegaConf.create(
        {
            "gradient_checkpointing_use_reentrant": False,
            "gradient_checkpointing_llm_layers": 2,
            "gradient_checkpointing_vision_layers": 1,
        }
    )
    build_kwargs = getattr(
        fsdp_model_manager,
        "_build_gradient_checkpointing_kwargs",
        lambda unused_config: {},
    )

    assert build_kwargs(config) == {
        "use_reentrant": False,
        "llm_layers": 2,
        "vision_layers": 1,
    }


def test_pi0_enables_requested_checkpoint_layer_counts() -> None:
    model = _make_pi0_shell()

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "llm_layers": 2,
            "vision_layers": 1,
        }
    )

    assert model.llm.gradient_checkpointing is True
    assert model.llm.gradient_checkpointing_layers == 2
    assert model.img.encoder.gradient_checkpointing is True
    assert model.img.encoder.gradient_checkpointing_layers == 1


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("llm_layers", -1, "llm_layers"),
        ("llm_layers", 5, "llm_layers"),
        ("vision_layers", 4, "vision_layers"),
        ("vision_layers", True, "vision_layers"),
    ],
)
def test_pi0_rejects_invalid_checkpoint_layer_counts(
    key: str, value: int, match: str
) -> None:
    model = _make_pi0_shell()

    with pytest.raises(ValueError, match=match):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={key: value})


def test_gemma_recomputes_only_requested_checkpoint_layers() -> None:
    model = _make_gemma_shell()
    model.gradient_checkpointing_layers = 2
    embedded = [torch.randn(2, 3, 4, requires_grad=True)]
    positions = torch.zeros(2, 3, dtype=torch.long)
    mask = torch.ones(2, 3, 3, dtype=torch.bool)

    outputs, _ = model(embedded, positions, mask, [None])
    outputs[0].sum().backward()

    assert [layer.calls for layer in model.layers] == [2, 2, 1]


def test_siglip_recomputes_only_requested_checkpoint_layers() -> None:
    model = siglip.Encoder(dim=4, depth=3, num_heads=1, mlp_dim=8)
    model.layers = nn.ModuleList([_CountingLayer() for _ in range(3)])
    model.gradient_checkpointing = True
    model.gradient_checkpointing_layers = 1
    inputs = torch.randn(2, 3, 4, requires_grad=True)

    model(inputs).sum().backward()

    assert [layer.calls for layer in model.layers] == [2, 1, 1]
