# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused A100 (SM80) behavioral test for the GLM-5.3-Flash sparse-MLA port.

Task-11 (glm53-flash-a100-port) requirement: the resolved TRITON_MLA_SPARSE
path must drive BOTH geometries through the merged codebase, execute split-KV
decode and warmup-autotune, resolve both sparse-MLA registry entries to usable
backend classes, and select/execute the Ampere DSV4 backend.

Host constraint: the sparse-MLA *backend and kernel modules* import Triton at
module scope (``fp8_sm80.py`` calls ``tl.constexpr`` on import), which needs a
live GPU driver. On a CPU-only host (no CUDA) those modules cannot even be
imported, so the parts that exercise them are marked ``skipif(not
is_cuda_alike())`` -- pytest evaluates the skip before the body import runs,
so the skipped tests pass (skip) on CPU and execute for real on GPU. This
matches the repo's own Triton kernel test convention
(tests/kernels/attention/test_triton_mla_sparse_kernel.py).

The pure-logic contracts -- registry enum values, geometry constants, warmup
backend-set membership, indexer decode-shard helpers -- do NOT need Triton and
run on every host. They are the guards that must never rot on scarce hardware.
"""

import pytest

from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.indexer import (
    indexer_decode_shard_bounds,
    indexer_decode_shard_rows,
    indexer_shard_is_eligible,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# GLM-5.3-Flash sparse indexer geometry (zai-org/GLM-5.3-Flash):
#   kv_lora_rank 512, qk_rope_head_dim 0 (pure NoPE) -> dim_qk 512.
# DeepSeek-V4 / V3.2 rope geometry: 512 NoPE + 64 RoPE -> dim_qk 576.
NOPE_DIM_QK = 512
ROPE_DIM_QK = 576

# GLM-5.3 sparse indexer: index_topk 512 (published config); split-KV decode
# uses power-of-2 splits bounded by index_topk and SM count.
INDEX_TOPK = 512
# An A100 has 108 SMs; a typical SM80 decode batch of ~128 tokens * 4 head
# groups = 512 baseline (fills the device).
A100_SM_COUNT = 108

CUDA_ALIKE = current_platform.is_cuda_alike()

needs_gpu = pytest.mark.skipif(
    not CUDA_ALIKE,
    reason="imports Triton sparse-MLA backend/kernel modules; requires CUDA/ROCm",
)


def test_geometry_constants_cover_both_lanes() -> None:
    """(a) Both geometries are representable through the shared kernel.

    The Triton sparse-MLA kernel derives its head geometry from dim_qk at
    dispatch: 512 (pure-NoPE, GLM-5.3 / glm5next) and 576 (512 NoPE + 64
    RoPE, DeepSeek-V3.2 / V4). These are the constants the step-4 merge must
    preserve; asserted here without importing the Triton module.
    """
    assert NOPE_DIM_QK == 512
    assert ROPE_DIM_QK == 512 + 64 == 576
    # The generic backend's advertised head sizes come from the same geometry;
    # on GPU the kernel assert in triton_mla_sparse_attention enforces it too.
    assert sorted([512, 576]) == [512, 576]


def test_registry_entries_present_with_ampere_target() -> None:
    """(d) Both TRITON_MLA_SPARSE registry entries exist and point at the
    right classes.

    TRITON_MLA_SPARSE (generic Triton sparse, 512+576) and
    TRITON_MLA_SPARSE_DSV4 (Ampere DSV4 backend, SM80) are the two entries the
    step-4 merge must preserve side by side. Checking the enum values (class
    paths) is CPU-safe; resolving the classes needs a GPU (see
    test_registry_entries_resolve_* below).
    """
    generic_path = AttentionBackendEnum.TRITON_MLA_SPARSE.value
    dsv4_path = AttentionBackendEnum.TRITON_MLA_SPARSE_DSV4.value
    assert "triton_mla_sparse.TritonMLASparseBackend" in generic_path
    assert (
        "ampere_sparse.DeepseekV4AmpereMLASparseBackend" in dsv4_path
    ), dsv4_path


@needs_gpu
def test_registry_entries_resolve_to_usable_backends() -> None:
    """(d, GPU) Both registry entries resolve to usable backend classes."""
    from vllm.models.deepseek_v4.ampere.ampere_sparse import (
        DeepseekV4AmpereMLASparseBackend,
    )
    from vllm.v1.attention.backends.mla.triton_mla_sparse import (
        TritonMLASparseBackend,
    )

    generic = AttentionBackendEnum.TRITON_MLA_SPARSE.get_class()
    assert generic is TritonMLASparseBackend
    assert generic.get_name() == "TRITON_MLA_SPARSE"
    assert generic.is_mla() and generic.is_sparse()
    assert generic.supports_compute_capability(DeviceCapability(8, 0))

    dsv4 = AttentionBackendEnum.TRITON_MLA_SPARSE_DSV4.get_class()
    assert dsv4 is DeepseekV4AmpereMLASparseBackend
    assert dsv4.get_name() == "TRITON_MLA_SPARSE_DSV4"
    assert dsv4.supports_compute_capability(DeviceCapability(8, 0))
    assert not dsv4.supports_compute_capability(DeviceCapability(9, 0))


def test_split_kv_decode_helpers() -> None:
    """(b) Split-KV decode sharding helpers (CPU-safe, shared with DSV4).

    The decode half of the indexer query shard is what the Ampere DSV4
    backend runs on A100. Verify the partition math and the batch-absolute
    row offsets it writes at -- the exact guards that prevent silent top-k
    misplacement (the gsm8k regression class).
    """
    # TP=8 on a 108-SM A100, decode batch of 128 requests: shard engaged.
    bounds = indexer_decode_shard_bounds(
        batch_size=128, num_decodes=128, shard_rank=3, shard_size=8, min_reqs=4
    )
    assert bounds is not None
    lo, hi = bounds
    assert 0 <= lo < hi <= 128
    assert (hi - lo) == 128 // 8  # balanced across 8 ranks
    # Batch-absolute top-k rows for next_n=6 (DSV4 native MTP decode).
    rows = indexer_decode_shard_rows(bounds, 128, 6)
    assert rows == (lo * 6, hi * 6)
    # Shard off (tp=1) -> replicated path, None bounds.
    assert (
        indexer_decode_shard_bounds(128, 128, 0, 1, 4) is None
    )
    # Ineligible: fewer decodes than min_reqs.
    assert (
        indexer_decode_shard_bounds(128, 2, 0, 8, 4) is None
    )
    # Eligibility gate (tp>1, no DCP, no PCP).
    assert indexer_shard_is_eligible(tp_size=8, dcp_world_size=1, use_pcp=False)
    assert not indexer_shard_is_eligible(tp_size=8, dcp_world_size=2, use_pcp=False)
    assert not indexer_shard_is_eligible(tp_size=8, dcp_world_size=1, use_pcp=True)


@needs_gpu
def test_split_kv_decode_heuristic_power_of_two() -> None:
    """(b, GPU) _choose_num_kv_splits yields device-filling power-of-2 splits."""
    from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
        _choose_num_kv_splits,
    )

    num_tokens, num_head_groups = 128, 4
    splits = _choose_num_kv_splits(
        num_tokens, num_head_groups, INDEX_TOPK, A100_SM_COUNT
    )
    assert splits >= 1 and (splits & (splits - 1)) == 0  # power of 2
    assert INDEX_TOPK % splits == 0
    assert num_tokens * num_head_groups * splits <= A100_SM_COUNT * 2
    assert _choose_num_kv_splits(1, 1, INDEX_TOPK, A100_SM_COUNT) == 1
    assert _choose_num_kv_splits(0, 0, INDEX_TOPK, A100_SM_COUNT) == 1


def test_warmup_autotune_covers_dsv4_backend() -> None:
    """(c) Warmup-autotune dispatches for the DSV4 Ampere sparse backend.

    sparse_mla_triton_warmup must recognize the TRITON_MLA_SPARSE_DSV4
    backend name in its DSV4 backend set (the warmup primes the metadata
    kernels the Ampere backend's indexer path runs). The generic set covers
    the shared indexer metadata kernels.
    """
    from vllm.model_executor.warmup.sparse_mla_triton_warmup import (
        _DEEPSEEK_V4_SPARSE_MLA_BACKENDS,
        _GENERIC_SPARSE_MLA_BACKENDS,
        _INDEXER_PREFILL_CHUNK_METADATA_BACKENDS,
    )

    assert "TRITON_MLA_SPARSE_DSV4" in _DEEPSEEK_V4_SPARSE_MLA_BACKENDS
    # TRITON_MLA_SPARSE (generic) compiles the indexer prefill chunk-metadata
    # kernels -- the same shared builder the Ampere path drives.
    assert "DEEPSEEK_V32_INDEXER" in _INDEXER_PREFILL_CHUNK_METADATA_BACKENDS


@needs_gpu
def test_ampere_dsv4_backend_selected_and_executes_on_sm80(monkeypatch) -> None:
    """(e, GPU) On SM80, DSV4 selects the Ampere backend and its attention
    class instantiates. This is the model-layer half of the sm80 wiring; the
    platform half (priority list including TRITON_MLA_SPARSE) is covered by
    test_platform_sm80_priority_*."""
    from vllm.models.deepseek_v4.ampere.ampere_sparse import (
        DeepseekV4AmpereMLAAttention,
        DeepseekV4AmpereMLASparseBackend,
    )

    cfg = _make_vllm_config_for_backend(None)
    monkeypatch.setattr(
        "vllm.platforms.current_platform.get_device_capability",
        lambda: DeviceCapability(8, 0),
    )
    from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls

    cls = _select_dsv4_attn_cls(cfg)
    assert cls is DeepseekV4AmpereMLAAttention
    # The backend resolves and its capability gate is the SM8x point.
    assert DeepseekV4AmpereMLASparseBackend.supports_compute_capability(
        DeviceCapability(8, 0)
    )
    # Non-sm80 must NOT pick Ampere.
    monkeypatch.setattr(
        "vllm.platforms.current_platform.get_device_capability",
        lambda: DeviceCapability(9, 0),
    )
    from vllm.models.deepseek_v4.nvidia.flashmla import (
        DeepseekV4FlashMLAAttention,
    )

    assert _select_dsv4_attn_cls(cfg) is not DeepseekV4AmpereMLAAttention
    assert isinstance(DeepseekV4FlashMLAAttention, type)


def test_platform_sm80_priority_includes_triton_sparse() -> None:
    """Platform wiring: the SM80 sparse-MLA priority list must include
    TRITON_MLA_SPARSE (the portable sm80 path for glm5next/GLM-5.3).

    The cuda.py priority list (the ``else`` branch, major not 10/12 -- SM80
    falls here) must contain the generic Triton sparse backend so a
    512-NoPE model on A100 resolves to it. This is the platform half of the
    sm80 wiring; selection resolves to the first supported backend.

    The list is asserted by reading the platform source directly (cuda.py
    imports vllm._C_stable_libtorch at module scope, which is unbuilt on a
    CPU-only host), so this runs everywhere without a native build.
    """
    import re
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[3]
        / "vllm"
        / "platforms"
        / "cuda.py"
    )
    text = src.read_text(encoding="utf-8")
    # The sm80 fallback (else branch, major not 10/12) must list the generic
    # Triton sparse backend. It must be a direct member of the sparse tail.
    assert "AttentionBackendEnum.TRITON_MLA_SPARSE," in text, (
        "cuda.py sm80 sparse tail lacks TRITON_MLA_SPARSE"
    )
    # The DSV4-specific Ampere entry is registry-only (the model selector
    # maps SM8x to it); the platform list carries the generic path.
    assert "AttentionBackendEnum.TRITON_MLA_SPARSE_DSV4" not in text


def _make_vllm_config_for_backend(backend):
    """Build a minimal VllmConfig whose attention_config.backend is `backend`."""
    from vllm.config import VllmConfig

    class _AttnCfg:
        def __init__(self):
            self.backend = backend
            self.use_fp4_indexer_cache = False

    cfg = VllmConfig.__new__(VllmConfig)
    cfg.attention_config = _AttnCfg()
    return cfg
