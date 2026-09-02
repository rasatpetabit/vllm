# SPDX-License-Identifier: Apache-2.0
# FileCopyrightText: Copyright contributors to the vLLM project
"""Query-layout capability selection (design rev 7 section 4.4).

Two layers:

- Pure layout logic (runs on every host): ``query_layout_for``, the base
  declaration surface raising for undeclared MLA backends, and the
  declaration-derived ``sort_sparse_tail`` ordering that replaces the
  identity ``head_size == 512`` platform branch — specialists precede
  general-purpose backends, platform index next, name last (deterministic
  ties).
- Backend-bound pins (run where the backend chain imports; module skip on
  hosts without the compiled ops, executing in-image at T26): the declared
  layout sets of the sparse-MLA family, the layout rejection record shape,
  and the fail-closed raise for a missing declaration.
"""

from __future__ import annotations

import enum

import pytest

from vllm.v1.attention.backends.mla.query_layout import (
    QueryLayout,
    UnsupportedQueryLayout,
    query_layout_for,
    sort_sparse_tail,
    supported_query_layouts as base_declaration,
)


def test_layout_derivation() -> None:
    assert query_layout_for(0) is QueryLayout.NOPE_ONLY
    for rope in (32, 64, 128):
        assert query_layout_for(rope) is QueryLayout.ROPE


def test_base_declaration_raises_for_mla_backends() -> None:
    """Fail closed: an MLA backend without a declaration refuses selection
    (the raise, not a silent default)."""
    with pytest.raises(UnsupportedQueryLayout):
        base_declaration()


def test_sort_promotes_exact_specialist_then_platform_then_name() -> None:
    class B(enum.Enum):
        GENERAL_A = "general_a"
        SPECIALIST = "specialist"
        GENERAL_B = "general_b"
        UNDECLARED = "undeclared"

    requested = QueryLayout.NOPE_ONLY
    declared = {
        B.GENERAL_A: frozenset({QueryLayout.NOPE_ONLY, QueryLayout.ROPE}),
        B.SPECIALIST: frozenset({QueryLayout.NOPE_ONLY}),
        B.GENERAL_B: frozenset({QueryLayout.ROPE}),
    }
    platform = {B.GENERAL_A: 0, B.SPECIALIST: 1, B.GENERAL_B: 2, B.UNDECLARED: 3}
    tail = [B.GENERAL_A, B.GENERAL_B, B.UNDECLARED, B.SPECIALIST]
    ordered = sort_sparse_tail(tail, requested, declared, platform)
    # the exact specialist precedes every general-purpose backend
    assert ordered[0] is B.SPECIALIST
    # general-purpose order follows the platform index; the rope-only
    # backend stays present (its rejection is validate_configuration's
    # job, not the sort's); undeclared sorts last within its group
    assert ordered[1:] == [B.GENERAL_A, B.GENERAL_B, B.UNDECLARED]


def test_sort_no_specialist_preserves_platform_order() -> None:
    class B(enum.Enum):
        TRITON = "triton_mla_sparse"
        FLASHINFER = "flashinfer_mla_sparse_sm90"

    declared = {
        B.TRITON: frozenset({QueryLayout.NOPE_ONLY, QueryLayout.ROPE}),
        B.FLASHINFER: frozenset({QueryLayout.NOPE_ONLY, QueryLayout.ROPE}),
    }
    platform = {B.TRITON: 2, B.FLASHINFER: 3}
    tail = [B.FLASHINFER, B.TRITON]
    ordered = sort_sparse_tail(tail, QueryLayout.NOPE_ONLY, declared, platform)
    assert ordered == [B.TRITON, B.FLASHINFER]


def test_sort_synthetic_tie_resolves_deterministically() -> None:
    """Two synthetic backends with identical declarations and platform
    indices tie-break on the name — deterministically, with no
    identity-specific branch left anywhere in the ordering."""

    class B(enum.Enum):
        ZETA = "a_zeta"
        ALPHA = "b_alpha"

    declared = {
        B.ZETA: frozenset({QueryLayout.ROPE}),
        B.ALPHA: frozenset({QueryLayout.ROPE}),
    }
    platform = {B.ZETA: 5, B.ALPHA: 5}
    for _ in range(3):  # stable across repeated sorts
        ordered = sort_sparse_tail([B.ZETA, B.ALPHA], QueryLayout.ROPE, declared, platform)
        assert ordered == [B.ALPHA, B.ZETA]  # name ascending


def test_triton_sparse_query_layout_contract() -> None:
    """Named kernel-contract test proving Triton sparse declares BOTH
    layouts: the pure dispatch handles rope_dim 0 (NoPE pass-through) and
    64 (concat) — the NoPE branch exists since f7ea90dea6."""
    from vllm.v1.attention.backends.mla.mla_query_dispatch import (
        select_decode_mqa_query,
    )

    import torch

    nope = torch.zeros(2, 4, 128)
    empty_pe = torch.zeros(2, 4, 0)
    out = select_decode_mqa_query(
        nope, empty_pe, layer_name="t", qk_rope_head_dim=0,
        fp8_attention=False, supports_quant_query_input=False,
        concat_quant_fp8_op=None, q_scale=1.0,
    )
    assert out is nope  # NoPE pass-through

    pe = torch.zeros(2, 4, 64)
    q, k = select_decode_mqa_query(
        nope, pe, layer_name="t", qk_rope_head_dim=64,
        fp8_attention=False, supports_quant_query_input=False,
        concat_quant_fp8_op=None, q_scale=1.0,
    )
    assert q is nope and k is pe  # rope path returns the pair

    # the geometry-bug branch raises the stable type
    with pytest.raises(RuntimeError, match="refusing NoPE-only"):
        select_decode_mqa_query(
            nope, empty_pe, layer_name="t", qk_rope_head_dim=64,
            fp8_attention=False, supports_quant_query_input=False,
            concat_quant_fp8_op=None, q_scale=1.0,
        )
    # fp8 + NoPE raises the STABLE UnsupportedQueryLayout
    with pytest.raises(UnsupportedQueryLayout):
        select_decode_mqa_query(
            nope, empty_pe, layer_name="t", qk_rope_head_dim=0,
            fp8_attention=True, supports_quant_query_input=True,
            concat_quant_fp8_op=None, q_scale=1.0,
        )
    # quant-query kernel without a concat op raises (never a bare assert)
    with pytest.raises(UnsupportedQueryLayout):
        select_decode_mqa_query(
            nope, pe, layer_name="t", qk_rope_head_dim=64,
            fp8_attention=True, supports_quant_query_input=True,
            concat_quant_fp8_op=None, q_scale=1.0,
        )


# ---------------------------------------------------------------------------
# Backend-bound pins (skip on hosts without the compiled ops chain; the
# in-image T26 run executes them with skips prohibited).
# ---------------------------------------------------------------------------
try:
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
        FlashInferMLASparseSM90Backend,
    )
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseBackend,
    )
    from vllm.v1.attention.backends.mla.triton_mla_sparse import (
        TritonMLASparseBackend,
    )

    _BACKENDS_IMPORTABLE = True
    _IMPORT_REASON = "importable"
except Exception as _exc:  # CPU hosts without compiled ops / triton
    _BACKENDS_IMPORTABLE = False
    _IMPORT_REASON = repr(_exc)

pytestmark_backend = pytest.mark.skipif(
    not _BACKENDS_IMPORTABLE, reason=f"backend chain not importable: {_IMPORT_REASON}"
)


@pytest.mark.skipif(not _BACKENDS_IMPORTABLE, reason="backend chain not importable")
def test_flashinfer_sm90_sparse_query_layout_contract() -> None:
    """Proves the FlashInfer SM90 sparse backend serves BOTH layouts: its
    own geometry gate accepts qk_rope_head_dim in (0, 64) — kpe 0 (NoPE)
    needs flashinfer >= 0.6.18 per the module contract. This is the named
    contract test the {NOPE_ONLY, ROPE} declaration cites."""
    import inspect

    from vllm.v1.attention.backends.mla.query_layout import QueryLayout as QL

    declared = FlashInferMLASparseSM90Backend.supported_query_layouts()
    assert declared == frozenset({QL.NOPE_ONLY, QL.ROPE})
    # the module's own supports_mla_geometry names both kpe values
    src = inspect.getsource(FlashInferMLASparseSM90Backend)
    assert "qk_rope_head_dim not in (0, 64)" in src or "(0, 64)" in src


@pytest.mark.skipif(not _BACKENDS_IMPORTABLE, reason="backend chain not importable")
def test_sparse_backend_declarations() -> None:
    from vllm.v1.attention.backends.mla.query_layout import QueryLayout as QL

    assert TritonMLASparseBackend.supported_query_layouts() == frozenset(
        {QL.NOPE_ONLY, QL.ROPE}
    )
    assert FlashMLASparseBackend.supported_query_layouts() == frozenset({QL.ROPE})


@pytest.mark.skipif(not _BACKENDS_IMPORTABLE, reason="backend chain not importable")
def test_layout_mismatch_is_a_structured_rejection() -> None:
    import torch

    from vllm.platforms.interface import DeviceCapability

    reasons = FlashMLASparseBackend.validate_configuration(
        head_size=512,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=64,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(9, 0),  # capability-clean on sm90
        attn_type="decoder",
        qk_rope_head_dim=0,  # NoPE request against a rope-only declaration
    )
    layout_reasons = [r for r in reasons if "query layout" in r]
    assert layout_reasons, reasons
    assert "NOPE_ONLY" in layout_reasons[0]
    assert "ROPE" in layout_reasons[0]
