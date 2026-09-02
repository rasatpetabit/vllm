# SPDX-License-Identifier: Apache-2.0
# FileCopyrightText: Copyright contributors to the vLLM project
"""Step-4 supersession surface pins (design rev 7 4.4, step4 schema).

The 576-only port graft (01ecc8e4) grew a surface that the generalized
HEAD backend (graft tip a818672184) does not carry: duplicated XPU
classes, a private metadata builder, a legacy forward-kernel name, a
local deep-GEMM predicate, and typing/hint imports. Each removal or move
is recorded in ``port/step4-supersessions.json`` on the skynet side with
a resolving superseding symbol; THIS file is the in-tree evidence that
the surface actually changed, pinned by source-text assertions so the
checks run on every host with no imports of the compiled chain (they are
verified in-image at T26 together with the full suite).
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _src(rel: str) -> str:
    path = _REPO_ROOT / rel
    assert path.is_file(), f"expected source file missing: {rel}"
    return path.read_text(encoding="utf-8")


_TRITON = "vllm/v1/attention/backends/mla/triton_mla_sparse.py"
_INDEXER = "vllm/v1/attention/backends/mla/indexer.py"
_FLASHMLA = "vllm/v1/attention/backends/mla/flashmla_sparse.py"
_XPU = "vllm/v1/attention/backends/mla/xpu_mla_sparse.py"
_SELECTOR = "vllm/utils/deep_gemm.py"


def test_xpu_classes_live_in_upstream_module_only() -> None:
    """XPUMLASparse* lives upstream in xpu_mla_sparse.py; the graft's
    copies inside triton_mla_sparse.py are gone."""
    xpu = _src(_XPU)
    for cls in ("XPUMLASparseBackend", "XPUMLASparseImpl", "XPUMLASparseMetadata", "XPUMLASparseMetadataBuilder"):
        assert f"class {cls}" in xpu, cls
    triton = _src(_TRITON)
    assert "class XPUMLASparse" not in triton


def test_builder_surface_moved_to_flashmla_parent() -> None:
    """The generalized Triton sparse backend inherits its builder surface
    from FlashMLASparseBackend; the private builder and its classmethods
    are gone from the child."""
    triton = _src(_TRITON)
    assert "class TritonMLASparseMetadataBuilder" not in triton
    assert "def get_builder_cls" not in triton
    assert "def get_supported_kernel_block_sizes" not in triton
    parent = _src(_FLASHMLA)
    assert "def get_builder_cls" in parent
    assert "def get_supported_kernel_block_sizes" in parent


def test_forward_kernel_and_head_dim_renamed() -> None:
    """_forward_bf16_kv -> _bf16_flash_mla_kernel and _DIM_QK ->
    _INDEXER_HEAD_DIM under the generalized kernel."""
    triton = _src(_TRITON)
    assert "_bf16_flash_mla_kernel" in triton
    # the child DEFINITION is gone (a comment may still name the parent's
    # inherited method; the pin is on the def, not on the substring)
    assert "def _forward_bf16_kv" not in triton
    assert "_INDEXER_HEAD_DIM" in triton
    assert "_DIM_QK" not in triton


def test_indexer_predicate_moved_to_shared_selector() -> None:
    """is_deep_gemm_supported is replaced by the ONE shared selector
    select_indexer_logits_path in utils/deep_gemm (design rev 7 4.4)."""
    indexer = _src(_INDEXER)
    assert "def is_deep_gemm_supported" not in indexer
    selector = _src(_SELECTOR)
    assert "def select_indexer_logits_path" in selector


def test_hint_imports_moved_to_indexer_lane() -> None:
    """AttentionCGSupport and MultipleOf are consumed by the indexer lane
    now; the generalized triton sparse module no longer imports them, and
    its ClassVar typing import went away with the builder surface."""
    triton = _src(_TRITON)
    assert "AttentionCGSupport" not in triton
    assert "MultipleOf" not in triton
    assert "ClassVar" not in triton
    indexer = _src(_INDEXER)
    assert "AttentionCGSupport" in indexer
    assert "MultipleOf" in indexer
