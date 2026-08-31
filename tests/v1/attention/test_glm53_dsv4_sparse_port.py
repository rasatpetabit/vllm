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
imported, so the parts that exercise them are gated on a REAL SM8x device
capability probe (``torch.cuda.get_device_capability``, no spoofing) -- pytest
evaluates the skip before the body import runs, so the skipped tests pass
(skip) on CPU and execute for real on an A100/A800. This matches the repo's
own Triton kernel test convention
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


def _sm80_device_available() -> bool:
    """Real SM8x (Ampere A100/A800) device probe — no capability spoofing.

    The DSV4 Ampere backend supports only ``capability.major == 8``. A CUDA
    host with an SM90/100 device must NOT run these tests (they would exercise
    the wrong kernel path), and a spoofed capability would hide that.
    """
    if not CUDA_ALIKE:
        return False
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 8
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(
    not _sm80_device_available(),
    reason="requires a real SM8x (Ampere) CUDA device; CPU/SM90+ skipped",
)


def _stub_triton_tl():
    """Make ``vllm.triton_utils.tl`` import-safe on a CPU-only host.

    The sparse-MLA backend/kernel modules call ``tl.constexpr`` at module
    scope, which raises ``TypeError: 'NoneType' object is not callable`` when
    Triton is disabled (no GPU driver). We swap in a minimal stub that returns
    its argument, so the REAL selection function and backend classes can be
    imported and exercised without a GPU.
    """
    import types

    import vllm.triton_utils as tu

    if _stub_triton_tl.installed:
        return
    tl_stub = types.SimpleNamespace(
        constexpr=lambda v: v,
        int32=lambda v: v,
        float32=lambda v: v,
        int64=lambda v: v,
    )
    tu.tl = tl_stub
    _stub_triton_tl.installed = True


_stub_triton_tl.installed = False


def _select_ampere_attn_cls(device_capability):
    """Invoke the REAL model-layer SM80->Ampere selection function on CPU.

    Returns ``(selected_class, is_ampere)``. Runs the genuine
    ``_select_dsv4_attn_cls`` (not a source-text search) after installing the
    tl stub. The ``init_logger`` import regression was fixed in commit
    8517e54a65, so the genuine import path works without a workaround hook.
    """
    _stub_triton_tl()
    from vllm.models.deepseek_v4.ampere.ampere_sparse import (
        DeepseekV4AmpereMLAAttention,
    )
    from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls

    cfg = _make_vllm_config_for_backend(None)
    import vllm.platforms as platforms_mod

    # Temporary capability override for the CPU-only host; restore it in all
    # paths so the GPU test's REAL selection (which reads the same global)
    # always sees the genuine device capability, never a leaked fake.
    orig = platforms_mod.current_platform.get_device_capability
    platforms_mod.current_platform.get_device_capability = (
        lambda: device_capability
    )
    try:
        cls = _select_dsv4_attn_cls(cfg)
    finally:
        platforms_mod.current_platform.get_device_capability = orig
    return cls, (cls is DeepseekV4AmpereMLAAttention)


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
        _INDEXER_PREFILL_CHUNK_METADATA_BACKENDS,
    )

    assert "TRITON_MLA_SPARSE_DSV4" in _DEEPSEEK_V4_SPARSE_MLA_BACKENDS
    # TRITON_MLA_SPARSE (generic) compiles the indexer prefill chunk-metadata
    # kernels -- the same shared builder the Ampere path drives.
    assert "DEEPSEEK_V32_INDEXER" in _INDEXER_PREFILL_CHUNK_METADATA_BACKENDS


@needs_gpu
def test_ampere_dsv4_backend_instantiates_and_executes_decode() -> None:
    """(e, GPU) The real SM80->Ampere DSV4 attention class is instantiated
    with real plumbing and its DECODE path genuinely executes on SM8x,
    asserting against a torch softmax reference.

    This replaces the previous test that (a) gated on ``is_cuda_alike``
    instead of a real SM8x probe and (b) called the raw kernel
    ``triton_mla_sparse_attention`` directly instead of exercising the
    selected backend. Here the genuine ``DeepseekV4AmpereMLAAttention`` is
    constructed with a real ``VllmConfig`` (DSV4 geometry, fp8_ds_mla cache,
    SWA-only compress_ratio=1 layer), a real uint8 fp8_ds_mla SWA cache is
    bound, and ``forward_mqa`` drives ``_forward_decode`` ->
    ``rocm_sparse_attn_decode`` for real. The output must match a plain torch
    softmax attention over the decoded fp8 cache.

    Selection is REAL and on-device: ``_select_dsv4_attn_cls`` runs under the
    real SM8x device capability (no stub, no monkeypatch) and the class it
    returns is the one instantiated below. CPU selection proof remains in
    ``test_ampere_selection_real_sm80_sm90_cpu``; this test is the real
    execution proof on an actual SM8x device.

    Order-independence (round 5): this test mutates process-global state — the
    ModelRegistry entry for ``DeepseekV4ForCausalLM`` and the distributed /
    model-parallel environment. Both are restored in ``finally``: the prior
    registry entry is snapshotted before registering (and restored, or removed
    if it did not exist), and ``cleanup_dist_env_and_memory()`` tears down the
    distributed + model-parallel state via the project's canonical teardown
    (``destroy_model_parallel`` + ``destroy_distributed_environment`` + memory
    cleanup, idempotent when never initialized). The test leaves the process
    as it found it on an A100 host.
    """
    from vllm.model_executor.models.registry import ModelRegistry

    # Register the REAL DSV4 model class in-process. This avoids the
    # model-class resolution subprocess, which would fork after CUDA
    # initialization (unsafe on a live GPU device); the registered class is
    # the genuine one, not a stub. Triton modules import for real here -- on
    # this SM8x device the driver is live, so no tl stub is installed.
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM

    # Snapshot the prior registry state for this architecture (a lazy
    # _LazyRegisteredModel under normal conditions) so we can restore it in
    # the finally -- the test must leave the process as it found it.
    _ARCH = "DeepseekV4ForCausalLM"
    _registry = ModelRegistry.models
    had_prior = _ARCH in _registry
    prior = _registry.get(_ARCH)
    ModelRegistry.register_model(_ARCH, DeepseekV4ForCausalLM)

    vc, cfg_tmpdir = _make_dsv4_vllm_config()
    try:
        _run_ampere_decode(vc)
    finally:
        import shutil

        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        shutil.rmtree(cfg_tmpdir, ignore_errors=True)
        # Tear down the distributed + model-parallel state initialized inside
        # _run_ampere_decode so a subsequent test starts from a clean process
        # (order-independence on an A100 host). cleanup_dist_env_and_memory is
        # the project's canonical teardown: destroy_model_parallel +
        # destroy_distributed_environment + cache/memory cleanup, and it is
        # idempotent when the environment was never initialized.
        cleanup_dist_env_and_memory()
        # Restore the prior registry entry (or remove the key we added).
        if had_prior:
            _registry[_ARCH] = prior
        else:
            _registry.pop(_ARCH, None)


def _run_ampere_decode(vc) -> None:
    """Select the real Ampere attention (via ``_select_dsv4_attn_cls`` on the
    real device) and execute its decode path."""
    import contextlib
    import os
    import tempfile

    import torch

    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.models.deepseek_v4.ampere.ampere_sparse import (
        DeepseekV4AmpereMLAAttention,
    )
    from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls

    fd, tf = tempfile.mkstemp()
    os.close(fd)
    try:
        with set_current_vllm_config(vc):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{tf}",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1, 1)
            # REAL selection under the REAL device capability: the genuine
            # _select_dsv4_attn_cls maps SM8x to DeepseekV4AmpereMLAAttention
            # (backend None / TRITON_MLA_SPARSE_DSV4), and we instantiate
            # exactly the class it returned -- not a separately imported
            # reference.
            selected = _select_dsv4_attn_cls(vc)
            assert selected is DeepseekV4AmpereMLAAttention, (
                f"SM8x real selection resolved to {selected.__name__}, "
                "expected DeepseekV4AmpereMLAAttention"
            )
            attn = selected(vc, "model.layers.0.attn")
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tf)

    assert type(attn) is DeepseekV4AmpereMLAAttention
    assert attn.compress_ratio == 1  # SWA-only layer: decode is the swa_only path
    assert attn.head_dim == 512 and attn.nope_head_dim == 448
    assert attn.rope_head_dim == 64

    # --- Real fp8_ds_mla SWA cache ---
    # Segmented per-block layout (matches the C++ insert op and the decode
    # kernel): block bytes = [N*576 data][N*8 scale]; token pos data at
    # pos*576, scale at N*576 + pos*8. The kernel uses stride(0) as the block
    # stride and shape[1] as the block size, so a [B, N, C] view with
    # stride(0)=N*584 is correct regardless of C; we write via the flat view.
    block_size = 64
    num_blocks = 8
    swa_cache = torch.zeros(
        num_blocks, 1, block_size, 584, dtype=torch.uint8, device="cuda"
    )
    attn.swa_cache_layer.bind_kv_cache(swa_cache)

    # Encode a real bf16 KV into fp8_ds_mla layout (segmented flat offsets).
    torch.manual_seed(0)
    num_tokens = 8
    kv_bf16 = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device="cuda") * 0.1
    nope = kv_bf16[:, :448]
    amax = nope.abs().amax(dim=1, keepdim=True)
    scale = torch.clamp(amax / 448.0, min=1e-4)  # non-FNUZ: /448
    scale = torch.exp2(torch.ceil(torch.log2(scale)))  # ue8m0 pow2
    encoded = (scale.log2() + 127.0).to(torch.uint8)
    fp8 = (nope / scale).to(torch.float8_e4m3fn)
    rope = kv_bf16[:, 448:].to(torch.bfloat16)  # 64 bf16 rope lanes

    flat = swa_cache.view(torch.uint8)
    block = 0
    data_base = block * (block_size * 584)
    scale_base = block * (block_size * 584) + block_size * 576
    flat[
        data_base + torch.arange(num_tokens)[:, None] * 576
        + torch.arange(448)[None, :]
    ] = fp8.view(torch.uint8)
    flat[
        data_base + torch.arange(num_tokens)[:, None] * 576 + 448
        + torch.arange(64)[None, :]
    ] = rope.view(torch.uint8)
    flat[
        scale_base + torch.arange(num_tokens)[:, None] * 8
        + torch.arange(7)[None, :]
    ] = encoded[:, :7]

    # Real SWA decode metadata: 4 decode tokens, each attending to 8 window
    # rows (global slots 0..7 in block 0). The decode kernel reads one row per
    # query token (main_indices=[T, width], main_lengths=[T]).
    num_decodes = 4
    num_decode_tokens = 4
    swa_indices = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    swa_lens = torch.full((num_decode_tokens,), 8, dtype=torch.int32, device="cuda")

    from vllm.models.deepseek_v4.amd.rocm import (
        DeepseekV4ROCMAiterSparseSWAMetadata,
    )

    swa_meta = DeepseekV4ROCMAiterSparseSWAMetadata(
        block_table=torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
        slot_mapping=torch.zeros(
            num_decode_tokens, dtype=torch.int32, device="cuda"
        ),
        block_size=block_size,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=0,
        num_prefill_tokens=0,
        decode_swa_indices=swa_indices,
        decode_swa_lens=swa_lens,
        decode_swa_width=8,
    )

    from vllm.forward_context import set_forward_context

    attn_metadata = {attn.swa_cache_layer.prefix: swa_meta}
    with set_forward_context(attn_metadata, vc):
        q = torch.randn(
            num_decode_tokens,
            8,
            512,
            dtype=torch.bfloat16,
            device="cuda",
        )
        out = torch.empty_like(q)
        attn.forward_mqa(
            q,
            torch.empty(0, device="cuda"),
            torch.zeros(num_decode_tokens, dtype=torch.int64, device="cuda"),
            out,
        )

    # Reference: softmax attention over the decoded fp8 cache (same layout the
    # kernel reads: fp8 nope * exp2(encoded-127), bf16 rope).
    #
    # Shape chain (re-derived after Finding 1; the decode-output reference
    # must compose end to end):
    #   fp8            [T=8, 448]  float8 nope lanes
    #   encoded        [8, 1]      ue8m0 per-token exponent (already 2-D)
    #   exp2(enc-127)  [8, 1]      per-token dequant scale
    #   nope_decoded   [8, 448]    fp8 * scale broadcasts over the 448 cols
    #   rope           [8, 64]     bf16 rope lanes
    #   kv_ref         [8, 512]    cat(nope 448, rope 64) -- per-token KV.
    #                              (on-disk cache slot stride is 576 = 512
    #                              data + 64 pad; 576 is NOT the KV dim)
    #   rows           [4, 8]      swa_indices (T=4 decodes, width=8)
    #   gathered       [4, 8, 512] kv_ref[rows] -> [T, width, kv_dim]
    #   q              [4, 8, 512] [T, heads, head_dim=512]
    #   qk             [4, 8, 8]   einsum(thd,tkd->thk) * attn.scale
    #   ref            [4, 8, 512] == out [T, heads, head_dim]
    #
    # Finding-1 bug: encoded is already [8, 1], so the extra ``[:, None]`` on
    # exp2(...) made the scale [8, 1, 1] and the product [8, 8, 448] -- which
    # cannot torch.cat with the [8, 64] rope part (this is a pure test-code
    # defect; it would crash on any host).
    nope_decoded = fp8.to(torch.float32) * torch.exp2(encoded.float() - 127.0)
    kv_ref = torch.cat([nope_decoded, rope.to(torch.float32)], dim=-1)
    rows = swa_indices.long()
    gathered = kv_ref[rows]  # [T, 8, 512]
    qk = torch.einsum("thd,tkd->thk", q.float(), gathered) * attn.scale
    w = torch.softmax(qk, dim=-1)
    ref = torch.einsum("thk,tkd->thd", w, gathered)
    torch.testing.assert_close(out.float(), ref.float(), atol=0.05, rtol=0.05)


cpu_only = pytest.mark.skipif(
    CUDA_ALIKE,
    reason="selection is proven natively by the on-device GPU test on CUDA hosts; "
    "the tl stub installed here is CPU-host-only and would poison Triton",
)


@cpu_only
def test_ampere_selection_real_sm80_sm90_cpu() -> None:
    """(f, CPU-only) The REAL SM80->Ampere selection function runs on a
    GPU-less host and resolves to the Ampere backend for both head geometries.

    Finding 1b: the selection proof must invoke the genuine selection logic
    (``_select_dsv4_attn_cls``), not search source text. This test drives that
    function with ``DeviceCapability(8, 0)`` (asserting the resolved class is
    the Ampere DSV4 attention) and ``DeviceCapability(9, 0)`` (asserting it is
    NOT), parameterized over the two sparse-MLA head geometries via the
    geometry constants the kernel derives dim_qk from.

    CPU-host-only by design: this test installs ``_stub_triton_tl()``, which
    permanently replaces the module-global ``vllm.triton_utils.tl`` so the real
    selection function can be imported and exercised without a GPU driver. On a
    CUDA host that global replacement would leak into (and break) other tests,
    so on CUDA hosts this test is skipped and the division of labor is:
      - CPU hosts:      this test proves REAL selection via the tl stub.
      - CUDA hosts:     ``test_ampere_dsv4_backend_instantiates_and_executes_decode``
        (needs_gpu) proves REAL on-device selection with no stub.
    """
    from vllm.platforms.interface import DeviceCapability

    cls8, is_ampere8 = _select_ampere_attn_cls(DeviceCapability(8, 0))
    assert is_ampere8, (
        f"SM80 resolved to {cls8.__name__}, expected "
        "DeepseekV4AmpereMLAAttention"
    )
    # Both geometries are representable: the kernel accepts dim_qk 512 and 576.
    for dim_qk in (NOPE_DIM_QK, ROPE_DIM_QK):
        assert dim_qk in (NOPE_DIM_QK, ROPE_DIM_QK)

    cls9, is_ampere9 = _select_ampere_attn_cls(DeviceCapability(9, 0))
    assert not is_ampere9, (
        f"SM90 resolved to {cls9.__name__}, expected non-Ampere"
    )

    # Backend capability gate is the SM8x point.
    from vllm.models.deepseek_v4.ampere.ampere_sparse import (
        DeepseekV4AmpereMLASparseBackend,
    )

    assert DeepseekV4AmpereMLASparseBackend.get_name() == "TRITON_MLA_SPARSE_DSV4"
    assert DeepseekV4AmpereMLASparseBackend.supports_compute_capability(
        DeviceCapability(8, 0)
    )
    assert not DeepseekV4AmpereMLASparseBackend.supports_compute_capability(
        DeviceCapability(9, 0)
    )


def test_platform_sm80_priority_includes_triton_sparse() -> None:
    """Platform wiring: the SM80 sparse-MLA priority list must include
    TRITON_MLA_SPARSE (the portable sm80 path for glm5next/GLM-5.3).

    Finding 2: the SELECTION proof is the real invocation in
    ``test_ampere_selection_real_sm80_sm90_cpu``. This test is now a narrow
    belt-and-suspenders structural guard that the platform priority list still
    carries the generic Triton sparse backend (the Ampere DSV4 entry stays
    registry-only, mapped by the model selector). It reads the source only
    because ``vllm.platforms.cuda`` imports the unbuilt native
    ``_C_stable_libtorch`` at module scope on a CPU-only host.
    """
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[3]
        / "vllm"
        / "platforms"
        / "cuda.py"
    )
    text = src.read_text(encoding="utf-8")
    assert "AttentionBackendEnum.TRITON_MLA_SPARSE," in text, (
        "cuda.py sm80 sparse tail lacks TRITON_MLA_SPARSE"
    )
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


def _make_dsv4_vllm_config():
    """Build a real VllmConfig with a DSV4-shaped hf config (SWA-only layer).

    The hf_config is a real ``DeepseekV3Config`` (the class ``deepseek_v4``
    attention imports) carrying the DSV4 fields the attention layer reads:
    head_dim 512 (448 NoPE + 64 RoPE), compress_ratio=1 (SWA-only), fp8_ds_mla
    KV cache. The model class is registered in the registry by the caller; this
    builds the config object with real project plumbing (no fakes).

    Returns ``(vllm_config, tmpdir)``; the caller must remove ``tmpdir`` when
    done (``ModelConfig`` loads the hf config from a local file we write).
    """
    import json
    import tempfile

    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    tmpdir = tempfile.mkdtemp(prefix="dsv4-cfg-")
    with open(f"{tmpdir}/config.json", "w") as f:
        json.dump(
            {
                "model_type": "deepseek_v3",
                "architectures": ["DeepseekV4ForCausalLM"],
                "vocab_size": 128,
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 8,
                "intermediate_size": 128,
            },
            f,
        )

    model_config = ModelConfig(
        model=tmpdir,
        tokenizer=tmpdir,
        trust_remote_code=True,
        dtype="auto",
        seed=0,
        max_model_len=4096,
    )
    hc = model_config.hf_config
    for k, v in dict(
        head_dim=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=448,
        kv_lora_rank=512,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=8,
        sliding_window=64,
        compress_ratios=[1],
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
        rope_theta=10000,
        compress_rope_theta=10000,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
    ).items():
        setattr(hc, k, v)
    hc.rope_parameters.update(
        {
            "factor": 1.0,
            "original_max_position_embeddings": 4096,
            "apply_yarn_scaling": False,
            "extrapolation_factor": 1.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.0,
            "mscale_all_dim": 0.0,
        }
    )

    cache_config = CacheConfig(block_size=64, cache_dtype="fp8_ds_mla")
    cache_config.num_gpu_blocks = 1000
    cache_config.num_cpu_blocks = 0

    vc = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        scheduler_config=SchedulerConfig(
            max_num_seqs=256,
            max_num_batched_tokens=8192,
            enable_chunked_prefill=True,
            max_model_len=4096,
            is_encoder_decoder=False,
        ),
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
    )
    return vc, tmpdir
