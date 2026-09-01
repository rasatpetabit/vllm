# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure decode-query dispatch logic for sparse MLA.

Split out of ``mla_attention.py`` so the decision can be unit-tested on CPU
hosts without importing the GPU module chain (``fp8_sm80`` calls
``tl.constexpr`` at module scope).
"""

from collections.abc import Callable

import torch


def select_decode_mqa_query(
    mqa_ql_nope: torch.Tensor,
    mqa_q_pe: torch.Tensor,
    *,
    layer_name: str,
    qk_rope_head_dim: int,
    fp8_attention: bool,
    supports_quant_query_input: bool,
    concat_quant_fp8_op: Callable[..., torch.Tensor] | None,
    q_scale: float,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Decide how the decode query reaches the sparse MLA kernel.

    NoPE-only MLA (e.g. glm5next on sm80) has an empty rope part: pass the
    absorbed nope query through directly, because the sparse kernels' q
    concat op requires rope_dim 64. The empty-rope branch is tied to the
    configured geometry -- an empty rope part on a rope-configured model is
    an upstream construction bug and fails loudly, as does the fp8 fused
    concat (which needs a rope part).
    """
    if mqa_q_pe.shape[-1] == 0:
        if qk_rope_head_dim != 0:
            raise RuntimeError(
                f"{layer_name}: empty q_pe but qk_rope_head_dim="
                f"{qk_rope_head_dim}; refusing NoPE-only decode path"
            )
        if fp8_attention:
            raise NotImplementedError(
                f"{layer_name}: fp8_attention with NoPE-only MLA "
                "is not implemented; the fused fp8 concat requires a rope part"
            )
        return mqa_ql_nope
    if fp8_attention and supports_quant_query_input:
        assert mqa_ql_nope.shape[0] == mqa_q_pe.shape[0]
        assert mqa_ql_nope.shape[1] == mqa_q_pe.shape[1]
        assert concat_quant_fp8_op is not None
        return concat_quant_fp8_op(mqa_ql_nope, mqa_q_pe, q_scale)
    return (mqa_ql_nope, mqa_q_pe)
