# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression tests for select_decode_mqa_query (mla_attention.py).

Pins the decode-query dispatch contract added for NoPE-only sparse MLA on
sm80 (adversarial review slice-5, 2026-09-01):

- empty rope part + configured rope_dim == 0 -> pass ql_nope through
- empty rope part + configured rope_dim != 0 -> loud RuntimeError
- empty rope part + fp8_attention -> loud NotImplementedError
- non-empty rope part + fp8 fused concat -> op called with q_scale
- non-empty rope part, plain path -> (ql_nope, q_pe) tuple
"""

import pytest
import torch

from vllm.v1.attention.backends.mla.mla_query_dispatch import (
    select_decode_mqa_query,
)


def _nope(n_tok: int = 4, n_heads: int = 16, dim: int = 512) -> torch.Tensor:
    return torch.zeros(n_tok, n_heads, dim)


def test_zero_rope_empty_pe_passes_nope_through() -> None:
    q_nope = _nope()
    q_pe = torch.zeros(4, 16, 0)
    out = select_decode_mqa_query(
        q_nope,
        q_pe,
        layer_name="layer",
        qk_rope_head_dim=0,
        fp8_attention=False,
        supports_quant_query_input=False,
        concat_quant_fp8_op=None,
        q_scale=1.0,
    )
    assert out is q_nope


def test_nonzero_rope_empty_pe_fails_loudly() -> None:
    q_nope = _nope()
    q_pe = torch.zeros(4, 16, 0)
    with pytest.raises(RuntimeError, match="qk_rope_head_dim=64"):
        select_decode_mqa_query(
            q_nope,
            q_pe,
            layer_name="layer",
            qk_rope_head_dim=64,
            fp8_attention=False,
            supports_quant_query_input=False,
            concat_quant_fp8_op=None,
            q_scale=1.0,
        )


def test_zero_rope_empty_pe_rejects_fp8_attention() -> None:
    q_nope = _nope()
    q_pe = torch.zeros(4, 16, 0)
    with pytest.raises(NotImplementedError, match="NoPE-only MLA"):
        select_decode_mqa_query(
            q_nope,
            q_pe,
            layer_name="layer",
            qk_rope_head_dim=0,
            fp8_attention=True,
            supports_quant_query_input=True,
            concat_quant_fp8_op=None,
            q_scale=1.0,
        )


def test_rope_fp8_path_calls_concat_op_with_scale() -> None:
    q_nope = _nope()
    q_pe = torch.zeros(4, 16, 64)
    calls: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    def fake_op(nope, pe, scale):
        calls.append((nope, pe, scale))
        return torch.zeros(4, 16, 576)

    out = select_decode_mqa_query(
        q_nope,
        q_pe,
        layer_name="layer",
        qk_rope_head_dim=64,
        fp8_attention=True,
        supports_quant_query_input=True,
        concat_quant_fp8_op=fake_op,
        q_scale=0.25,
    )
    assert out.shape == (4, 16, 576)
    assert len(calls) == 1
    assert calls[0][0] is q_nope
    assert calls[0][1] is q_pe
    assert calls[0][2] == 0.25


def test_rope_plain_path_returns_tuple() -> None:
    q_nope = _nope()
    q_pe = torch.zeros(4, 16, 64)
    out = select_decode_mqa_query(
        q_nope,
        q_pe,
        layer_name="layer",
        qk_rope_head_dim=64,
        fp8_attention=False,
        supports_quant_query_input=False,
        concat_quant_fp8_op=None,
        q_scale=1.0,
    )
    assert isinstance(out, tuple)
    assert out[0] is q_nope
    assert out[1] is q_pe
