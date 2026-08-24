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

"""Tests for the OpenPI-RLinf scaled dot-product attention path."""

import copy

import pytest
import torch

from rlinf.models.embodiment.openpi_rlinf.pi0_model.gemma import Attention, Config


def _attention_inputs() -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1234)
    inputs = [
        torch.randn(2, 3, 64, generator=generator, requires_grad=True),
        torch.randn(2, 2, 32, generator=generator, requires_grad=True),
    ]
    positions = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    mask = torch.tensor(
        [
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, True, True, True, False],
                [True, True, True, True, True],
            ],
            [
                [True, True, False, False, False],
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ],
        ]
    ).unsqueeze(1)
    return inputs, positions, mask


def _attention_configs(head_dim: int = 16) -> list[Config]:
    common = {
        "depth": 1,
        "mlp_dim": 128,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": head_dim,
    }
    return [Config(width=64, **common), Config(width=32, **common)]


def _forward_and_backward(
    attention: Attention,
    inputs: list[torch.Tensor],
    positions: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    outputs, _ = attention(inputs, positions, mask)
    loss = sum(output.square().mean() for output in outputs)
    gradients = torch.autograd.grad(loss, [*inputs, *attention.parameters()])
    return outputs, list(gradients)


def test_sdpa_attention_matches_eager_outputs_and_gradients(
    monkeypatch,
) -> None:
    """Catch SDPA mask/GQA changes that diverge from eager attention."""
    monkeypatch.delenv("PI_ATTN_IMPL", raising=False)
    eager = Attention(_attention_configs())
    monkeypatch.setenv("PI_ATTN_IMPL", "SDPA")
    sdpa = Attention(_attention_configs())
    sdpa.load_state_dict(copy.deepcopy(eager.state_dict()))

    eager_inputs, positions, mask = _attention_inputs()
    sdpa_inputs = [value.detach().clone().requires_grad_() for value in eager_inputs]

    eager_outputs, eager_gradients = _forward_and_backward(
        eager, eager_inputs, positions, mask
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        sdpa_outputs, sdpa_gradients = _forward_and_backward(
            sdpa, sdpa_inputs, positions, mask
        )

    operator_names = {event.key for event in profile.key_averages()}
    assert "aten::scaled_dot_product_attention" in operator_names
    for actual, expected in zip(sdpa_outputs, eager_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for actual, expected in zip(sdpa_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flash_attention_block_decomposition_matches_eager(
    monkeypatch,
) -> None:
    """Catch block decomposition or padding changes that alter attention."""
    pytest.importorskip("flash_attn")
    device = torch.device("cuda")
    dtype = torch.bfloat16

    monkeypatch.delenv("PI_ATTN_IMPL", raising=False)
    eager = Attention(_attention_configs(head_dim=256)).to(device=device, dtype=dtype)
    monkeypatch.setenv("PI_ATTN_IMPL", "FLASH_ATTN")
    flash = Attention(_attention_configs(head_dim=256)).to(device=device, dtype=dtype)
    flash.load_state_dict(copy.deepcopy(eager.state_dict()))

    generator = torch.Generator(device=device).manual_seed(5678)
    eager_inputs = [
        torch.randn(
            2,
            4,
            64,
            generator=generator,
            device=device,
            dtype=dtype,
            requires_grad=True,
        ),
        torch.randn(
            2,
            3,
            32,
            generator=generator,
            device=device,
            dtype=dtype,
            requires_grad=True,
        ),
    ]
    flash_inputs = [value.detach().clone().requires_grad_() for value in eager_inputs]
    positions = torch.arange(7, device=device).expand(2, -1)

    # Pi0.5 prefix-LM layout: prefix queries only see prefix keys, while all
    # suffix queries see both prefix and suffix. Prefix padding differs by batch.
    block_mask = torch.tensor(
        [
            [True, True, True, True, False, False, False],
            [True, True, True, True, False, False, False],
            [True, True, True, True, False, False, False],
            [True, True, True, True, False, False, False],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ],
        device=device,
    )
    valid_tokens = torch.tensor(
        [
            [True, True, True, False, True, True, True],
            [True, True, False, False, True, True, True],
        ],
        device=device,
    )
    mask = (
        block_mask.unsqueeze(0) & valid_tokens.unsqueeze(1) & valid_tokens.unsqueeze(2)
    ).unsqueeze(1)

    eager_outputs, eager_gradients = _forward_and_backward(
        eager, eager_inputs, positions, mask
    )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        flash_outputs, flash_gradients = _forward_and_backward(
            flash, flash_inputs, positions, mask
        )
    torch.cuda.synchronize()

    operator_names = {event.key for event in profile.key_averages()}
    assert "FlashAttnVarlenFunc" in operator_names
    assert "FlashAttnVarlenFuncBackward" in operator_names
    for actual, expected in zip(flash_outputs, eager_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    for actual, expected in zip(flash_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
