# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the sparse MLA backends and utilities."""

import math

# --- CPU bootstrap (fully restoring, platform-gated) ---------------------
# On a CPU-only host the native ``_C_stable_libtorch`` extension and the
# Triton language runtime are unavailable, which would make the module
# below uncollectable at import time (``fp8_sm80`` calls ``tl.constexpr`` at
# module scope). The backend-SELECTION logic (auto / forced / issue-54059
# regression) must run on every host, so on hosts without CUDA we install
# minimal stubs for the three module-scope GPU dependencies -- then RESTORE
# every mutated global byte-for-byte (identity) before the module body
# continues, so no later test in the same pytest process sees the stubs.
#
# The gate is ``current_platform.is_cuda()`` (the vLLM platform), NOT
# ``torch.cuda.is_available()``: the platform is what the production
# selection path actually consults, so a CUDA-platform host with transiently
# unavailable Torch CUDA must still load the real extensions, never the
# stubs. ``test_task12_cpu_bootstrap_restores_globals`` proves the restore
# is byte-identical, and ``test_task12_stub_path_never_runs_on_cuda`` proves
# the stub path is skipped on a CUDA platform.
import sys as _sys
import types as _types
from types import MethodType, SimpleNamespace

import pytest
import torch

# The module-scope GPU dependencies the CPU bootstrap may need to stub.
_STUB_MODULE_NAMES = (
    "vllm._C_stable_libtorch",
    "vllm._moe_C_stable_libtorch",
)
# Sentinel distinguishing "entry/attr absent" from "present-as-None" when
# snapshotting sys.modules / vllm.triton_utils.tl. sys.modules.get() would
# conflate the two, breaking restore for pre-existing None entries.
_ABSENT = object()

_STUB_MODULES_STATE: dict[str, object] = {}
_STUB_TL_STATE: object | None = None

try:
    import vllm.triton_utils as _tu
except Exception:  # pragma: no cover - vllm is importable in all test hosts
    _tu = None


# Collection-time bootstrap. It runs ONCE at import, installs stubs only when
# the vLLM platform reports no CUDA, imports the modules that need the stubs
# to be collectable, then restores every global it touched.
def _install_cpu_stubs() -> bool:
    """Install the minimal stubs. Returns True if stubs were installed.

    Only ever installs on a non-CUDA vLLM platform. On CUDA (or any other
    accelerated platform) this returns False and mutates nothing, so the
    real extensions always load. The state dicts are only written when a
    stub is actually installed, so a CUDA-gated call never leaves residue.
    """
    global _STUB_TL_STATE
    from vllm.platforms import current_platform  # lazy: needs no stub

    if current_platform.is_cuda():
        return False
    for name in _STUB_MODULE_NAMES:
        _STUB_MODULES_STATE[name] = _sys.modules.get(name, _ABSENT)
    _STUB_TL_STATE = (
        _ABSENT if _tu is None else getattr(_tu, "tl", _ABSENT)
    )
    for name in _STUB_MODULE_NAMES:
        if name not in _sys.modules:
            _sys.modules[name] = _types.ModuleType(name)
    if _tu is not None:
        _tu.tl = _types.SimpleNamespace(
            constexpr=lambda v: v,
            int32=lambda v: v,
            float32=lambda v: v,
            int64=lambda v: v,
        )
    return True


def _restore_cpu_stubs() -> None:
    """Restore every global the bootstrap touched, byte-for-byte.

    ``_STUB_MODULES_STATE`` / ``_STUB_TL_STATE`` snapshot the pre-bootstrap
    identity (or absence). Restoring the exact same objects (identity, not
    copy) guarantees later tests see the pristine globals.
    """
    global _STUB_TL_STATE
    if _STUB_TL_STATE is _ABSENT:
        # tl was ABSENT before: delete it back out. Guard on absence (not
        # falsiness) so a pre-existing present-as-None is restored, not
        # deleted.
        if _tu is not None and hasattr(_tu, "tl"):
            delattr(_tu, "tl")
    elif _tu is not None:
        # Present before (object or None): put the exact same value back.
        _tu.tl = _STUB_TL_STATE
    for name, prev in _STUB_MODULES_STATE.items():
        if prev is _ABSENT:
            # Absent before: pop it back out (present-as-None is preserved
            # by the else branch).
            _sys.modules.pop(name, None)
        else:
            _sys.modules[name] = prev
    _STUB_MODULES_STATE.clear()
    _STUB_TL_STATE = None


_STUBS_INSTALLED = _install_cpu_stubs()  # collection-time bootstrap; restores below
try:
    # The two sparse backend modules call ``tl.constexpr`` at module scope;
    # with Triton disabled (no GPU driver) that raises. Import them under the
    # stub so the rest of this module can reference them at collection time.
    import vllm.platforms.cuda  # noqa: E402,F401  (pulls _C_stable_libtorch)
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse  # noqa: E402,F401
    import vllm.v1.attention.backends.mla.flashmla_sparse  # noqa: E402,F401
    import vllm.v1.attention.backends.mla.triton_mla_sparse  # noqa: E402,F401
finally:
    # Restore ONLY when stubs were actually installed. On a CUDA platform
    # _install_cpu_stubs() returns False and never snapshots the globals
    # (_STUB_TL_STATE stays at its module-level None); running the restore
    # unconditionally would then clobber vllm.triton_utils.tl to None and
    # break every later tl.constexpr module import in the same pytest
    # process (incident 2026-08-31 wave-5 G-build, first real-GPU run).
    if _STUBS_INSTALLED:
        _restore_cpu_stubs()


from tests.v1.attention.test_mla_backends import (  # noqa: E402
    BATCH_SPECS,
    BatchSpec,
    MockSparseMLAAttentionLayer,
    create_and_prepopulate_kv_cache,
)
from tests.v1.attention.utils import (  # noqa: E402
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm import _custom_ops as ops  # noqa: E402
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.attention.mla_attention import (  # noqa: E402
    _use_masked_mha,
)
from vllm.model_executor.layers.attention.sparse_mla_attention import (  # noqa: E402
    GLOBAL_TOPK_MASK_MAX_BYTES,
    _masked_mha_workspace_fits,
    _topk_mask_shape,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402

# TODO: Integrate ROCMAiterMLASparseBackend for ROCm.
# The ROCm sparse MLA backend (rocm_aiter_mla_sparse.py) has a compatible
# forward_mqa interface but needs validation on ROCm hardware.
cuda_required = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason=(
        "Sparse MLA backend hardware-execution tests require CUDA "
        "(ROCm not yet integrated)."
    ),
)


def _cuda_capability() -> tuple[int, int]:
    """CUDA compute capability, or (0, 0) on hosts with no usable CUDA.

    Module-scope ``torch.cuda.get_device_capability()`` raises on a CPU-only
    host, so decorators use this CPU-safe helper instead.
    """
    try:
        return tuple(torch.cuda.get_device_capability())  # type: ignore[return-value]
    except Exception:
        return (0, 0)

import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as flashinfer_sparse_mod  # noqa: E402
from vllm.utils.math_utils import cdiv  # noqa: E402
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (  # noqa: E402
    FlashInferMLASparseImpl,
    FlashInferMLASparseTRTLLMBackend,
)  # noqa: E402
from vllm.v1.attention.backends.mla.flashmla_sparse import (  # noqa: E402
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
    FlashMLASparseMetadataBuilder,
    triton_convert_req_index_to_global_index,
)  # noqa: E402
from vllm.v1.attention.backends.mla.indexer import (  # noqa: E402
    split_indexer_prefill_chunks,
)
from vllm.v1.attention.backends.utils import (  # noqa: E402
    split_decodes_and_prefills,
    split_prefill_chunks,
)  # noqa: E402
from vllm.v1.attention.ops import flashmla  # noqa: E402

SPARSE_BACKEND_BATCH_SPECS = {
    name: BATCH_SPECS[name]
    for name in [
        "mixed_small",
        "mixed_medium",
        "small_prefill",
        "medium_prefill",
        "single_prefill",
    ]
}

SPARSE_BACKEND_BATCH_SPECS["large_q_prefill"] = BatchSpec(
    seq_lens=[1024] * 2, query_lens=[256] * 2
)
SPARSE_BACKEND_BATCH_SPECS["large_q_pure_prefill"] = BatchSpec(
    seq_lens=[256] * 2, query_lens=[256] * 2
)

DEVICE_TYPE = current_platform.device_type


@cuda_required
def test_nope_flashinfer_sparse_mla_uses_model_scale(monkeypatch):
    """Weight absorption must not change the model's attention temperature."""
    model_scale = 256**-0.5
    kv_lora_rank = 512
    topk = torch.zeros((1, 1), dtype=torch.int32)
    metadata = SimpleNamespace(
        req_id_per_token=torch.zeros(1, dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        block_size=1,
    )
    recorded_scale = None

    impl = object.__new__(FlashInferMLASparseImpl)
    impl.scale = model_scale
    impl.qk_nope_head_dim = 256
    impl.kv_lora_rank = kv_lora_rank
    impl.qk_rope_head_dim = 0
    impl.kv_cache_dtype = "auto"
    impl.topk_indices_buffer = topk
    impl.dcp_world_size = 1
    impl._workspace_buffer = torch.empty(1)
    impl.bmm1_scale = None
    impl.bmm2_scale = None
    impl.is_nope_mla = True
    impl.need_to_return_lse_for_decode = False
    monkeypatch.setattr(
        flashinfer_sparse_mod,
        "triton_convert_req_index_to_global_index",
        lambda *args, **kwargs: (topk, torch.ones(1, dtype=torch.int32)),
    )

    import flashinfer.decode

    def fake_flashinfer(**kwargs):
        nonlocal recorded_scale
        recorded_scale = kwargs["bmm1_scale"]
        return torch.zeros((1, 1, 1, kv_lora_rank))

    monkeypatch.setattr(
        flashinfer.decode,
        "trtllm_batch_decode_with_kv_cache_mla",
        fake_flashinfer,
    )
    impl.forward_mqa(
        torch.zeros(1, 1, kv_lora_rank),
        torch.zeros(1, kv_lora_rank),
        metadata,
        SimpleNamespace(),
    )

    assert recorded_scale == model_scale
    assert recorded_scale != kv_lora_rank**-0.5


def _float_to_e8m0_truncate(f: float) -> float:
    """Simulate SM100's float -> e8m0 -> bf16 scale conversion.
    e8m0 format only stores the exponent (power of 2).
    cudaRoundZero truncates toward zero, meaning we round down to the
    nearest power of 2.
    """
    if f <= 0:
        return 0.0
    # e8m0 = floor(log2(f)), then 2^(e8m0)
    # This is equivalent to truncating to the nearest power of 2 below f
    exp = math.floor(math.log2(f))
    return 2.0**exp


def _dequantize_fp8_ds_mla_entry(
    cache_slice: torch.Tensor,
    kv_lora_rank: int,
    rope_dim: int,
    dtype: torch.dtype,
    simulate_sm100_e8m0_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize a single fp8_ds_mla cache entry back to latent + rope.

    Args:
        simulate_sm100_e8m0_scales: If True, simulate the SM100 kernel's
            float -> e8m0 -> bf16 scale conversion path.
    """

    # The first kv_lora_rank bytes store FP8 latent values with one scale per
    # 128 element tile written as float32 right after the latent payload.
    scales = cache_slice.view(torch.float32)[kv_lora_rank // 4 : kv_lora_rank // 4 + 4]
    latent = torch.empty(kv_lora_rank, dtype=torch.float16, device=cache_slice.device)
    for tile_idx in range(4):
        tile_start = tile_idx * 128
        tile_end = tile_start + 128
        scale_val = float(scales[tile_idx].item())
        if simulate_sm100_e8m0_scales:
            # Simulate the lossy float -> e8m0 -> bf16 conversion
            scale_val = _float_to_e8m0_truncate(scale_val)
        ops.convert_fp8(
            latent[tile_start:tile_end],
            cache_slice[tile_start:tile_end],
            scale_val,
            kv_dtype="fp8",
        )
    latent = latent.to(dtype)

    rope_offset = kv_lora_rank // 2 + 8
    rope_vals = cache_slice.view(dtype)[rope_offset : rope_offset + rope_dim]
    return latent, rope_vals.clone()


def _quantize_dequantize_fp8_ds_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    block_size: int,
    scale: torch.Tensor,
    simulate_sm100_e8m0_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-trip kv_c/k_pe though the fp8_ds_mla cache layout.

    Args:
        simulate_sm100_e8m0_scales: If True, simulate the SM100 kernel's
            float -> e8m0 -> bf16 scale conversion in dequantization.
    """

    if kv_c.numel() == 0:
        return kv_c.clone(), k_pe.clone()

    kv_lora_rank = kv_c.shape[-1]
    rope_dim = k_pe.shape[-1]
    num_tokens = kv_c.shape[0]
    num_blocks = max(1, math.ceil(num_tokens / block_size))
    entry_size = kv_lora_rank + 4 * 4 + 2 * rope_dim

    tmp_cache = torch.zeros(
        num_blocks, block_size, entry_size, dtype=torch.uint8, device=kv_c.device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=kv_c.device)

    ops.concat_and_cache_mla(
        kv_c, k_pe, tmp_cache, slot_mapping, kv_cache_dtype="fp8_ds_mla", scale=scale
    )

    dequant_kv_c = torch.empty_like(kv_c)
    dequant_k_pe = torch.empty_like(k_pe)

    for token_idx in range(num_tokens):
        slot = slot_mapping[token_idx].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        cache_slice = tmp_cache[block_idx, block_offset]
        latent, rope_vals = _dequantize_fp8_ds_mla_entry(
            cache_slice,
            kv_lora_rank,
            rope_dim,
            kv_c.dtype,
            simulate_sm100_e8m0_scales=simulate_sm100_e8m0_scales,
        )
        dequant_kv_c[token_idx] = latent
        dequant_k_pe[token_idx] = rope_vals

    return dequant_kv_c, dequant_k_pe


@pytest.mark.parametrize(
    "backend_cls",
    [FlashMLASparseBackend, FlashInferMLASparseTRTLLMBackend],
    ids=["FlashMLA", "FlashInferTRTLLM"],
)
@pytest.mark.parametrize("batch_name", list(SPARSE_BACKEND_BATCH_SPECS.keys()))
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8", "fp8_ds_mla"])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize(("q_scale", "k_scale"), [(1.0, 1.0), (2.0, 3.0)])
@cuda_required
def test_sparse_backend_decode_correctness(
    default_vllm_config,
    dist_init,
    backend_cls,
    batch_name,
    kv_cache_dtype,
    tensor_parallel_size,
    block_size,
    workspace_init,
    q_scale: float,
    k_scale: float,
):
    if kv_cache_dtype not in backend_cls.supported_kv_cache_dtypes:
        pytest.skip(f"{backend_cls.get_name()} does not support {kv_cache_dtype}")

    if (
        backend_cls == FlashMLASparseBackend
        and kv_cache_dtype.startswith("fp8")
        and kv_cache_dtype != "fp8_ds_mla"
    ):
        pytest.skip(
            "FlashMLA Sparse Attention backend fp8 only supports "
            "fp8_ds_mla kv-cache dtype"
        )

    supported_block_sizes = backend_cls.get_supported_kernel_block_sizes()
    if block_size not in supported_block_sizes:
        pytest.skip(
            f"{backend_cls.get_name()} does not support block_size={block_size}"
        )

    if backend_cls == FlashMLASparseBackend:
        ok, reason = flashmla.is_flashmla_sparse_supported()
        if not ok:
            pytest.skip(reason)
    elif backend_cls == FlashInferMLASparseTRTLLMBackend:
        device_capability = current_platform.get_device_capability()
        if device_capability is None or not backend_cls.supports_compute_capability(
            device_capability
        ):
            pytest.skip("FlashInferMLASparseTRTLLMBackend requires SM 10.x capability")

    batch_spec = SPARSE_BACKEND_BATCH_SPECS[batch_name]
    use_fp8_ds_mla_quantization = kv_cache_dtype == "fp8_ds_mla"

    device = torch.device(DEVICE_TYPE)
    dtype = torch.bfloat16

    # Model hyper-parameters (kept intentionally small for the unit test)
    total_num_heads = 128
    # Compute per-rank heads for simulated TP
    num_heads = max(1, total_num_heads // tensor_parallel_size)

    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    head_size = kv_lora_rank + qk_rope_head_dim
    topk_tokens = 128

    max_seqlen = max(batch_spec.seq_lens)
    total_cache_tokens = sum(batch_spec.seq_lens)

    # Note: We use TP=1 to avoid multi-GPU requirements in CI.
    # The test simulates head partitioning via mocked methods below.
    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        tensor_parallel_size=1,
        max_model_len=max_seqlen,
        num_gpu_blocks=max(2048, cdiv(total_cache_tokens, block_size) + 1),
        block_size=block_size,
        hf_config_override={
            "index_topk": topk_tokens,
            "attn_module_list_cfg": [{"topk_tokens": topk_tokens}],
        },
    )
    model_config = vllm_config.model_config
    model_config.hf_text_config = SimpleNamespace(
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        model_type="deepseek_v2",
    )
    model_config.dtype = dtype
    model_config.get_num_attention_heads = MethodType(
        lambda self, parallel_config: num_heads,
        model_config,
    )
    model_config.get_num_kv_heads = MethodType(
        lambda self, parallel_config: 1, model_config
    )
    model_config.get_head_size = MethodType(lambda self: head_size, model_config)
    model_config.get_sliding_window = MethodType(lambda self: None, model_config)

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    torch.manual_seed(0)

    scale = 1.0 / math.sqrt(head_size)

    # Shared MLA projection weights to keep reference and backend in sync
    W_UK = torch.rand(
        kv_lora_rank, num_heads, qk_nope_head_dim, dtype=dtype, device=device
    )
    W_UV = torch.rand(kv_lora_rank, num_heads, v_head_dim, dtype=dtype, device=device)

    # Build synthetic decode-only workload
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens

    # Pre-compute positions and sparse indices for all tokens.
    # We need these BEFORE computing the reference to use sparse attention masks.
    total_query_tokens = sum(query_lens)
    positions = []
    for i in range(batch_spec.batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        ctx_len = s_len - q_len
        for q_idx in range(q_len):
            positions.append(ctx_len + q_idx)

    # Create sparse indices with UNIQUE per-token offsets to catch bugs where
    # the kernel uses wrong indices for some tokens (e.g., due to incorrect
    # tensor shapes like [1, num_tokens, ...] instead of [num_tokens, 1, ...]).
    # Also include -1 masked indices to verify the kernel handles them correctly.
    sparse_indices = torch.empty(
        total_query_tokens, topk_tokens, dtype=torch.int32, device=device
    )
    for tok_idx in range(total_query_tokens):
        max_valid_idx = positions[tok_idx]
        offset = tok_idx * 7  # Prime number for varied offsets
        # Use only half the topk indices as valid, mask the rest with -1
        # This tests that the kernel correctly ignores -1 indices
        num_valid = min(topk_tokens // 2, max_valid_idx + 1)
        if num_valid > 0:
            valid_range = torch.arange(num_valid, device=device, dtype=torch.int32)
            tok_indices = (valid_range + offset) % (max_valid_idx + 1)
            # Pad with -1 for the remaining positions
            tok_indices = torch.cat(
                [
                    tok_indices,
                    torch.full(
                        (topk_tokens - num_valid,), -1, device=device, dtype=torch.int32
                    ),
                ]
            )
        else:
            tok_indices = torch.full(
                (topk_tokens,), -1, device=device, dtype=torch.int32
            )
            tok_indices[0] = 0  # At least one valid index
        sparse_indices[tok_idx] = tok_indices

    all_q_vllm, all_kv_c_vllm, all_k_pe_vllm = [], [], []
    kv_c_contexts, k_pe_contexts = [], []
    reference_outputs = []

    kv_cache_scale = torch.tensor(k_scale, dtype=torch.float32, device=device)
    global_token_idx = 0

    for i in range(batch_spec.batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        ctx_len = s_len - q_len

        q_c = torch.rand(
            q_len,
            num_heads,
            qk_nope_head_dim + qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )
        kv_c_full = torch.rand(s_len, kv_lora_rank, dtype=dtype, device=device)
        k_pe_full = torch.rand(s_len, 1, qk_rope_head_dim, dtype=dtype, device=device)

        if use_fp8_ds_mla_quantization:
            is_sm100 = torch.cuda.get_device_capability()[0] >= 10
            kv_c_full, k_pe_squeezed = _quantize_dequantize_fp8_ds_mla(
                kv_c_full,
                k_pe_full.squeeze(1),
                block_size=block_size,
                scale=kv_cache_scale,
                simulate_sm100_e8m0_scales=is_sm100,
            )
            k_pe_full = k_pe_squeezed.unsqueeze(1)

        q_nope, q_pe = q_c.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
        ql_nope = torch.einsum("qnh,lnh->qnl", q_nope, W_UK)
        q_mqa = torch.cat([ql_nope, q_pe], dim=-1)

        k_mqa = torch.cat([kv_c_full, k_pe_full.squeeze(1)], dim=-1)
        v_mqa = kv_c_full

        # Compute sparse SDPA reference per query token using its sparse indices
        for q_idx in range(q_len):
            tok_sparse_idx = sparse_indices[global_token_idx]
            valid_mask = tok_sparse_idx >= 0
            valid_indices = tok_sparse_idx[valid_mask].long()

            q_tok = q_mqa[q_idx : q_idx + 1]  # [1, num_heads, head_dim]
            k_sparse = k_mqa[valid_indices]  # [num_valid, head_dim]
            v_sparse = v_mqa[valid_indices]  # [num_valid, kv_lora_rank]

            k_sparse = k_sparse.unsqueeze(1).expand(-1, num_heads, -1)
            v_sparse = v_sparse.unsqueeze(1).expand(-1, num_heads, -1)

            # SDPA: [1, num_heads, 1, head_dim] x [1, num_heads, num_valid, head_dim]
            q_sdpa_in = q_tok.unsqueeze(0).transpose(1, 2)
            k_sdpa_in = k_sparse.unsqueeze(0).transpose(1, 2)
            v_sdpa_in = v_sparse.unsqueeze(0).transpose(1, 2)

            sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa_in, k_sdpa_in, v_sdpa_in, scale=scale
            )
            sdpa_out = sdpa_out.transpose(1, 2).squeeze(
                0
            )  # [1, num_heads, kv_lora_rank]

            sdpa_out = torch.einsum("qnl,lnv->qnv", sdpa_out, W_UV)
            reference_outputs.append(sdpa_out.flatten(start_dim=-2))

            global_token_idx += 1

        all_q_vllm.append(q_c)
        all_kv_c_vllm.append(kv_c_full[ctx_len:])
        all_k_pe_vllm.append(k_pe_full[ctx_len:])
        kv_c_contexts.append(kv_c_full[: ctx_len + 1])
        k_pe_contexts.append(k_pe_full[: ctx_len + 1])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    kv_c_vllm = torch.cat(all_kv_c_vllm, dim=0)
    k_pe_vllm = torch.cat(all_k_pe_vllm, dim=0)
    sdpa_reference = torch.cat(reference_outputs, dim=0)

    vllm_config.cache_config.cache_dtype = kv_cache_dtype
    vllm_config.model_config.hf_config.index_topk = topk_tokens

    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        vllm_config.cache_config.block_size,
        device,
        arange_block_indices=True,
    )

    kv_cache = create_and_prepopulate_kv_cache(
        kv_c_contexts=kv_c_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=vllm_config.cache_config.block_size,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=vllm_config.cache_config.num_gpu_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
        kv_cache_dtype=kv_cache_dtype,
        scale=kv_cache_scale,
    )

    # The sparse builder clones the layer's dense-MHA prefill backend from
    # static_forward_context; register a mock layer carrying one.
    from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend

    prefill_backend = get_mla_prefill_backend(vllm_config)(
        num_heads=num_heads,
        scale=scale,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        vllm_config=vllm_config,
    )
    vllm_config.compilation_config.static_forward_context["placeholder"] = (
        SimpleNamespace(prefill_backend=prefill_backend)
    )

    builder_cls = backend_cls.get_builder_cls()
    builder = builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)
    metadata = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )

    # Use the pre-computed sparse_indices for the mock indexer
    mock_indexer = SimpleNamespace(topk_indices_buffer=sparse_indices)

    kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1)
    kv_b_proj_weight = kv_b_proj_weight.view(
        kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)
    )

    mock_kv_b_proj = ColumnParallelLinear(
        input_size=kv_lora_rank,
        output_size=num_heads * (qk_nope_head_dim + v_head_dim),
        bias=False,
    ).to(device=device, dtype=dtype)
    mock_kv_b_proj.weight = torch.nn.Parameter(kv_b_proj_weight.T.contiguous())

    impl_cls = backend_cls.get_impl_cls()
    with set_current_vllm_config(vllm_config):
        impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=None,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_b_proj=mock_kv_b_proj,
            indexer=mock_indexer,
        )

        impl.process_weights_after_loading(dtype)

        # Create mock sparse MLA layer with weight matrices
        mock_layer = MockSparseMLAAttentionLayer(
            impl=impl,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            device=device,
            W_UK=W_UK,
            W_UV=W_UV,
            q_scale=q_scale,
            k_scale=k_scale,
        )

    out_buffer = torch.empty(
        metadata.num_actual_tokens, num_heads * v_head_dim, dtype=dtype, device=device
    )

    with torch.inference_mode():
        backend_output = mock_layer.forward_impl(
            query_vllm,
            kv_c_vllm,
            k_pe_vllm,
            kv_cache,
            metadata,
            out_buffer,
        )

    assert backend_output.shape == sdpa_reference.shape
    assert backend_output.dtype == sdpa_reference.dtype
    assert torch.isfinite(backend_output).all()

    # FP8 quantization introduces some error, but should be within reasonable bounds
    # BF16 (auto) should be very accurate, FP8 allows slightly more tolerance
    if kv_cache_dtype.startswith("fp8"):
        torch.testing.assert_close(
            backend_output, sdpa_reference, rtol=0.065, atol=0.05
        )
    else:
        torch.testing.assert_close(backend_output, sdpa_reference, rtol=0.01, atol=0.01)


def _triton_convert_reference_impl(
    req_ids: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
    num_topk_tokens: int,
    HAS_PREFILL_WORKSPACE: bool = False,
    prefill_workspace_request_ids: torch.Tensor | None = None,
    prefill_workspace_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation for triton_convert_req_index_to_global_index."""
    num_tokens = req_ids.shape[0]
    max_blocks_per_req = block_table.shape[1]
    result = torch.empty(
        num_tokens, num_topk_tokens, dtype=torch.int32, device=req_ids.device
    )

    for token_id in range(num_tokens):
        req_id = req_ids[token_id].item()

        # Determine if this token uses workspace or paged cache
        use_prefill_workspace = False
        workspace_start = 0
        if HAS_PREFILL_WORKSPACE and prefill_workspace_request_ids is not None:
            assert prefill_workspace_starts is not None
            prefill_req_id = prefill_workspace_request_ids[token_id].item()
            if prefill_req_id >= 0:
                use_prefill_workspace = True
                workspace_start = prefill_workspace_starts[prefill_req_id].item()

        for idx_id in range(num_topk_tokens):
            token_idx = token_indices[token_id, idx_id].item()

            if token_idx == -1:
                result[token_id, idx_id] = -1
            elif use_prefill_workspace:
                # Prefill + using prefill workspace: map to workspace offset
                result[token_id, idx_id] = workspace_start + token_idx
            else:
                # Decode: map to paged cache
                block_id = token_idx // block_size
                if block_id >= max_blocks_per_req:
                    result[token_id, idx_id] = -1
                else:
                    block_num = block_table[req_id, block_id].item()
                    offset = token_idx % block_size
                    result[token_id, idx_id] = block_num * block_size + offset

    return result


@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize("num_topk_tokens", [128, 256, 512])
@pytest.mark.skipif(
    _cuda_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
def test_triton_convert_req_index_to_global_index_decode_only(
    block_size, num_topk_tokens
):
    device = torch.device(DEVICE_TYPE)
    num_tokens = 8
    num_requests = 4
    max_blocks_per_req = 10

    req_id = torch.randint(
        0, num_requests, (num_tokens,), dtype=torch.int32, device=device
    )
    block_table = torch.randint(
        0, 100, (num_requests, max_blocks_per_req), dtype=torch.int32, device=device
    )

    token_indices = torch.randint(
        0,
        block_size * max_blocks_per_req,
        (num_tokens, num_topk_tokens),
        dtype=torch.int32,
        device=device,
    )

    # Set some to -1 to test masking
    token_indices[0, :10] = -1
    token_indices[3, 50:60] = -1

    # Set some to out of bounds
    token_indices[2, 100:110] = max_blocks_per_req * block_size
    token_indices[6, 150:160] = max_blocks_per_req * block_size

    result = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
    )

    reference_result = _triton_convert_reference_impl(
        req_id,
        block_table,
        token_indices,
        block_size,
        num_topk_tokens,
    )

    torch.testing.assert_close(result, reference_result, rtol=0, atol=0)


@pytest.mark.parametrize("block_size", [16])
@pytest.mark.skipif(
    _cuda_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
def test_triton_convert_req_index_to_global_index_with_prefill_workspace(block_size):
    device = torch.device(DEVICE_TYPE)
    num_requests = 4
    max_blocks_per_req = 8
    num_topk_tokens = 128

    # First 6 tokens are decode (reqs 0, 1), last 6 are prefill (reqs 2, 3)
    req_id = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.int32, device=device
    )
    prefill_workspace_request_ids = torch.tensor(
        [-1, -1, -1, -1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device
    )

    # Workspace starts for the 2 prefill reqs: req 2 starts at 0, req 3 starts at 100
    prefill_workspace_starts = torch.tensor([0, 100], dtype=torch.int32, device=device)

    block_table = torch.randint(
        0, 50, (num_requests, max_blocks_per_req), dtype=torch.int32, device=device
    )
    token_indices = torch.randint(
        0,
        block_size * max_blocks_per_req,
        (req_id.shape[0], num_topk_tokens),
        dtype=torch.int32,
        device=device,
    )

    # Set some to -1 to test masking
    token_indices[0, :10] = -1
    token_indices[3, 50:60] = -1

    # Set some to out of bounds
    token_indices[2, 100:110] = max_blocks_per_req * block_size
    token_indices[6, 150:160] = max_blocks_per_req * block_size

    result = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
        HAS_PREFILL_WORKSPACE=True,
        prefill_workspace_request_ids=prefill_workspace_request_ids,
        prefill_workspace_starts=prefill_workspace_starts,
    )

    reference_result = _triton_convert_reference_impl(
        req_id,
        block_table,
        token_indices,
        block_size,
        num_topk_tokens,
        HAS_PREFILL_WORKSPACE=True,
        prefill_workspace_request_ids=prefill_workspace_request_ids,
        prefill_workspace_starts=prefill_workspace_starts,
    )

    torch.testing.assert_close(result, reference_result, rtol=0, atol=0)


@pytest.mark.skipif(
    _cuda_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
def test_triton_convert_rejects_req_id_longer_than_token_indices():
    """Guard against the #47327 regression: the kernel grid is sized by
    req_id but the output is allocated like token_indices, so a full-batch
    req_id combined with an MQA-subset token_indices wrote past the end of
    the output buffer. The wrapper must reject the length mismatch instead
    of corrupting memory."""
    device = torch.device(DEVICE_TYPE)
    num_topk_tokens = 128
    block_size = 64
    block_table = torch.arange(40, dtype=torch.int32, device=device).view(4, 10)

    # Full batch: 2 decode tokens + 10 prefill tokens
    req_id_full = torch.tensor(
        [0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], dtype=torch.int32, device=device
    )
    num_mqa_tokens = 2
    token_indices = torch.randint(
        0,
        block_size * 10,
        (num_mqa_tokens, num_topk_tokens),
        dtype=torch.int32,
        device=device,
    )

    with pytest.raises(AssertionError, match="must cover the same tokens"):
        triton_convert_req_index_to_global_index(
            req_id_full,
            block_table,
            token_indices,
            BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=num_topk_tokens,
        )

    # The sliced call is the intended usage and must match the reference.
    result = triton_convert_req_index_to_global_index(
        req_id_full[:num_mqa_tokens],
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
    )
    reference = _triton_convert_reference_impl(
        req_id_full[:num_mqa_tokens],
        block_table,
        token_indices,
        block_size,
        num_topk_tokens,
    )
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.skipif(
    _cuda_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
def test_flashmla_forward_bf16_kv_slices_req_id_to_mqa_tokens():
    """Guard against the #47327 regression: when the dense-MHA prefill split
    is active, forward_mqa only receives the leading decode tokens, but
    _forward_bf16_kv passed the full-batch req_id_per_token to the index
    conversion, making it write past the end of its output buffer. The call
    site must slice req_id_per_token to the MQA tokens."""
    device = torch.device(DEVICE_TYPE)
    num_topk_tokens = 128
    block_size = 64
    num_batch_tokens = 12
    num_mqa_tokens = 2

    attn_metadata = SimpleNamespace(
        req_id_per_token=torch.tensor(
            [0, 1] + [2] * 5 + [3] * 5, dtype=torch.int32, device=device
        ),
        block_table=torch.arange(40, dtype=torch.int32, device=device).view(4, 10),
        block_size=block_size,
    )
    assert attn_metadata.req_id_per_token.shape[0] == num_batch_tokens

    q = torch.zeros(num_mqa_tokens, 4, 576, dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros(40, block_size, 576, dtype=torch.bfloat16, device=device)
    topk_indices = torch.randint(
        0,
        block_size * 10,
        (num_mqa_tokens, num_topk_tokens),
        dtype=torch.int32,
        device=device,
    )

    captured = {}

    def _stub_kernel(q, kv, indices, lengths, actual_num_heads):
        captured["indices"] = indices
        captured["actual_num_heads"] = actual_num_heads
        return torch.zeros(q.shape[0], q.shape[1], 512, dtype=q.dtype, device=q.device)

    stub_impl = SimpleNamespace(_bf16_flash_mla_kernel=_stub_kernel)

    out = FlashMLASparseImpl._forward_bf16_kv(
        stub_impl, q, kv_cache, topk_indices, attn_metadata, q.shape[1]
    )

    assert out.shape[0] == num_mqa_tokens
    assert captured["indices"].shape[0] == num_mqa_tokens
    assert captured["actual_num_heads"] == q.shape[1]
    reference = _triton_convert_reference_impl(
        attn_metadata.req_id_per_token[:num_mqa_tokens],
        attn_metadata.block_table,
        topk_indices,
        block_size,
        num_topk_tokens,
    )
    torch.testing.assert_close(captured["indices"], reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "seq_lens,max_buf,expected",
    [
        # Basic split: totals per chunk ≤ max_buf
        (torch.tensor([2, 3, 4, 2]), 5, [(0, 2), (2, 3), (3, 4)]),
        # Exact fits should split between items when adding the next would overflow
        (torch.tensor([5, 5, 5]), 5, [(0, 1), (1, 2), (2, 3)]),
        # All requests fit in a single chunk
        (torch.tensor([1, 1, 1]), 10, [(0, 3)]),
        # Large buffer
        (torch.tensor([4, 4, 4]), 100, [(0, 3)]),
    ],
)
def test_split_prefill_chunks(seq_lens, max_buf, expected):
    out = split_prefill_chunks(seq_lens, max_buf)
    assert out == expected


@pytest.mark.parametrize(
    ("max_query_len", "expected"),
    [(32768, True), (33024, False)],
)
def test_masked_mha_workspace_fits_single_request_boundary(max_query_len, expected):
    """A 32K prefill needs the default workspace exactly; shrinking it would
    push a supported request onto MQA."""
    assert (
        _masked_mha_workspace_fits(
            batch_size=1,
            max_query_len=max_query_len,
            max_context_chunk_seq_len=0,
            workspace_numel=GLOBAL_TOPK_MASK_MAX_BYTES // torch.int32.itemsize,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("backend_name", "tensor_parallel_size", "query_len"),
    [
        ("FLASHMLA_SPARSE", 4, 48 * 1024),
        ("FLASHMLA_SPARSE", 8, 112 * 1024),
        ("FLASHINFER_MLA_SPARSE", 4, 36 * 1024),
        ("FLASHINFER_MLA_SPARSE", 8, 64 * 1024),
    ],
)
def test_masked_mha_workspace_guards_long_routing_policy(
    backend_name, tensor_parallel_size, query_len
):
    assert _use_masked_mha(
        backend_name=backend_name,
        tensor_parallel_size=tensor_parallel_size,
        qk_head_dim=256,
        v_head_dim=256,
        query_len=query_len,
        seq_len=query_len,
        has_context=False,
    )
    assert not _masked_mha_workspace_fits(
        batch_size=1,
        max_query_len=query_len,
        max_context_chunk_seq_len=0,
        workspace_numel=GLOBAL_TOPK_MASK_MAX_BYTES // torch.int32.itemsize,
    )


def test_masked_mha_workspace_fits_accounts_for_batch_and_context():
    """Request count and context chunk length are independent multipliers."""
    base = dict(batch_size=2, max_query_len=2048, max_context_chunk_seq_len=2048)
    exact = math.prod(_topk_mask_shape(2, 2048, 2048))

    assert _masked_mha_workspace_fits(**base, workspace_numel=exact)
    assert not _masked_mha_workspace_fits(
        **{**base, "batch_size": 3}, workspace_numel=exact
    )
    assert not _masked_mha_workspace_fits(
        **{**base, "max_context_chunk_seq_len": 4096}, workspace_numel=exact
    )


PREFILL_BATCH_SPECS = {
    "short_dense_mha": BatchSpec(seq_lens=[64, 128], query_lens=[64, 128]),
    "short_context_dense_mha": BatchSpec(seq_lens=[128, 160], query_lens=[64, 32]),
    "masked_mha": BatchSpec(seq_lens=[256], query_lens=[256]),
    "masked_mha_chunked_context": BatchSpec(seq_lens=[448, 384], query_lens=[256, 256]),
}


@pytest.mark.skipif(
    _cuda_capability()[0] < 10,
    reason="Sparse MLA forward_mha requires FA4 (SM100+)",
)
@pytest.mark.parametrize("batch_name", list(PREFILL_BATCH_SPECS.keys()))
@pytest.mark.parametrize("kv_cache_dtype", ["auto"])
@pytest.mark.parametrize(
    ("num_heads", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"),
    [
        pytest.param(128, 128, 64, 128, id="deepseek_hd192_v128"),
        pytest.param(64, 192, 64, 256, id="glm5_hd256_v256"),
    ],
)
def test_sparse_backend_prefill_correctness(
    default_vllm_config,
    dist_init,
    batch_name,
    kv_cache_dtype,
    num_heads,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    workspace_init,
):
    """Test dense and masked MHA across supported sparse MLA dimensions."""
    backend_cls = FlashMLASparseBackend
    batch_spec = PREFILL_BATCH_SPECS[batch_name]

    device = torch.device("cuda")
    dtype = torch.bfloat16
    block_size = 64

    kv_lora_rank = 512
    head_size = kv_lora_rank + qk_rope_head_dim
    masked_mha = batch_name.startswith("masked_mha")
    topk_tokens = 200 if masked_mha else 512

    max_seqlen = max(batch_spec.seq_lens)
    total_cache_tokens = sum(batch_spec.seq_lens)

    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        tensor_parallel_size=1,
        max_model_len=max_seqlen,
        num_gpu_blocks=max(2048, cdiv(total_cache_tokens, block_size) + 1),
        block_size=block_size,
        hf_config_override={
            "index_topk": topk_tokens,
            "attn_module_list_cfg": [{"topk_tokens": topk_tokens}],
        },
    )
    model_config = vllm_config.model_config
    model_config.hf_text_config = SimpleNamespace(
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        model_type="deepseek_v2",
    )
    model_config.dtype = dtype
    model_config.model_arch_config.total_num_attention_heads = num_heads
    model_config.get_num_attention_heads = MethodType(
        lambda self, parallel_config: num_heads, model_config
    )
    model_config.get_num_kv_heads = MethodType(
        lambda self, parallel_config: 1, model_config
    )
    model_config.get_head_size = MethodType(lambda self: head_size, model_config)
    model_config.get_sliding_window = MethodType(lambda self: None, model_config)

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)

    torch.manual_seed(42)

    W_UK = torch.rand(
        kv_lora_rank, num_heads, qk_nope_head_dim, dtype=dtype, device=device
    )
    W_UV = torch.rand(kv_lora_rank, num_heads, v_head_dim, dtype=dtype, device=device)

    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens

    # Compute dense reference outputs.
    total_query_tokens = sum(query_lens)
    sparse_indices = torch.full(
        (total_query_tokens, topk_tokens), -1, dtype=torch.int32, device=device
    )

    all_q, all_kv_c_new, all_k_pe_new = [], [], []
    kv_c_contexts, k_pe_contexts = [], []
    reference_outputs = []
    global_token_idx = 0

    for i in range(batch_spec.batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        ctx_len = s_len - q_len

        q_mha = torch.rand(
            q_len,
            num_heads,
            qk_nope_head_dim + qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )
        kv_c_full = torch.rand(s_len, kv_lora_rank, dtype=dtype, device=device)
        k_pe_full = torch.rand(s_len, 1, qk_rope_head_dim, dtype=dtype, device=device)

        # Decompress all KV for reference
        kv_b_weight = torch.cat([W_UK, W_UV], dim=-1).view(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)
        )
        kv_decompressed = (kv_c_full @ kv_b_weight).view(
            s_len, num_heads, qk_nope_head_dim + v_head_dim
        )
        k_nope_all, v_all = kv_decompressed.split(
            [qk_nope_head_dim, v_head_dim], dim=-1
        )
        k_pe_expanded = k_pe_full.expand(-1, num_heads, -1)
        k_all = torch.cat([k_nope_all, k_pe_expanded], dim=-1)

        for j in range(q_len):
            attend_end = ctx_len + j + 1
            q_tok = q_mha[j : j + 1]  # (1, H, D_qk)
            if masked_mha:
                actual_topk = min(topk_tokens, attend_end)
                attend_indices = torch.randperm(attend_end, device=device)[:actual_topk]
                sparse_indices[global_token_idx, :actual_topk] = attend_indices
                k_attend = k_all[attend_indices]
                v_attend = v_all[attend_indices]
            else:
                k_attend = k_all[:attend_end]  # (N, H, D_qk)
                v_attend = v_all[:attend_end]  # (N, H, D_v)

            q_sdpa = q_tok.unsqueeze(0).transpose(1, 2).float()
            k_sdpa = k_attend.unsqueeze(0).transpose(1, 2).float()
            v_sdpa = v_attend.unsqueeze(0).transpose(1, 2).float()

            out = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, scale=scale
            )
            out = out.transpose(1, 2).squeeze(0)  # (1, H, D_v)
            reference_outputs.append(out.to(dtype).flatten(start_dim=-2))
            global_token_idx += 1

        all_q.append(q_mha)
        all_kv_c_new.append(kv_c_full[ctx_len:])
        all_k_pe_new.append(k_pe_full[ctx_len:])
        kv_c_contexts.append(kv_c_full)
        k_pe_contexts.append(k_pe_full)

    query_cat = torch.cat(all_q, dim=0)
    kv_c_cat = torch.cat(all_kv_c_new, dim=0)
    k_pe_cat = torch.cat(all_k_pe_new, dim=0)
    ref_output = torch.cat(reference_outputs, dim=0)

    vllm_config.cache_config.cache_dtype = kv_cache_dtype
    vllm_config.model_config.hf_config.index_topk = topk_tokens

    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        vllm_config.cache_config.block_size,
        device,
        arange_block_indices=True,
    )

    kv_cache = create_and_prepopulate_kv_cache(
        kv_c_contexts=kv_c_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=block_size,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=vllm_config.cache_config.num_gpu_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
        kv_cache_dtype=kv_cache_dtype,
    )

    # The sparse builder clones the layer's dense-MHA prefill backend from
    # static_forward_context; register a mock layer carrying one.
    from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend

    prefill_backend = get_mla_prefill_backend(vllm_config)(
        num_heads=num_heads,
        scale=scale,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        vllm_config=vllm_config,
    )
    vllm_config.compilation_config.static_forward_context["placeholder"] = (
        SimpleNamespace(prefill_backend=prefill_backend)
    )

    builder_cls = backend_cls.get_builder_cls()
    builder = builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)
    if batch_name == "masked_mha_chunked_context":
        builder.chunked_prefill_workspace_size = block_size * batch_spec.batch_size
        builder.chunked_prefill_workspace = torch.empty(
            (builder.chunked_prefill_workspace_size, head_size),
            dtype=dtype,
            device=device,
        )
    # Drive the queries through the dense-MHA prefill path directly (the routing
    # threshold would otherwise classify these short queries as MQA decodes).
    builder.reorder_batch_threshold = 1
    metadata = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )

    mock_indexer = SimpleNamespace(topk_indices_buffer=sparse_indices)

    kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1).view(
        kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)
    )

    mock_kv_b_proj = ColumnParallelLinear(
        input_size=kv_lora_rank,
        output_size=num_heads * (qk_nope_head_dim + v_head_dim),
        bias=False,
    ).to(device=device, dtype=dtype)
    mock_kv_b_proj.weight = torch.nn.Parameter(kv_b_proj_weight.T.contiguous())

    impl_cls = backend_cls.get_impl_cls()
    with set_current_vllm_config(vllm_config):
        impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=None,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_b_proj=mock_kv_b_proj,
            indexer=mock_indexer,
        )
        impl.process_weights_after_loading(dtype)

    out_buffer = torch.empty(
        total_query_tokens, num_heads * v_head_dim, dtype=dtype, device=device
    )

    with torch.inference_mode():
        impl.forward_mha(
            q=query_cat,
            kv_c_normed=kv_c_cat,
            k_pe=k_pe_cat,
            # Impls see the bind-time-squeezed [B, N, C] cache; mirror bind_kv_cache.
            kv_c_and_k_pe_cache=kv_cache.squeeze(1),
            attn_metadata=metadata,
            k_scale=torch.tensor(1.0, device=device),
            output=out_buffer,
        )

    assert out_buffer.shape == ref_output.shape
    assert torch.isfinite(out_buffer).all(), "Non-finite values in output"
    torch.testing.assert_close(out_buffer, ref_output, rtol=0.01, atol=0.01)


@pytest.mark.parametrize(
    "seq_lens,query_lens,workspace_size,max_logits_bytes,expected",
    [
        (
            torch.tensor([0]),
            torch.tensor([0]),
            100,
            1000,
            [],
        ),
        # Logits constraint triggers split (M*N exceeds budget)
        # req0: M=10, N=100 -> 1000 elems (4000 bytes) - fits in 5000
        # req1: adding M=10, N=100 -> new_M=20, new_N=200 -> 4000 elems > 1250
        (
            torch.tensor([100, 100, 100]),
            torch.tensor([10, 10, 10]),
            1000,  # workspace allows all
            5000,  # 1250 float32 elems -> forces split
            [
                (slice(0, 1), slice(0, 10)),
                (slice(1, 2), slice(0, 10)),
                (slice(2, 3), slice(0, 10)),
            ],
        ),
        # Both constraints satisfied - all fit in one chunk
        (
            torch.tensor([10, 10, 10]),
            torch.tensor([5, 5, 5]),
            100,
            10000,  # 2500 elems, M*N = 15*30 = 450 < 2500
            [(slice(0, 3), slice(0, 15))],
        ),
        # Workspace constraint triggers first
        (
            torch.tensor([50, 50, 50]),
            torch.tensor([1, 1, 1]),
            50,  # workspace only fits one at a time
            1000000,  # logits budget is huge
            [
                (slice(0, 1), slice(0, 1)),
                (slice(1, 2), slice(0, 1)),
                (slice(2, 3), slice(0, 1)),
            ],
        ),
        # Greedy filling: first two fit, third doesn't
        # req0: M=5, N=10 -> 50 elems
        # req0+1: M=10, N=20 -> 200 elems <= 250
        # req0+1+2: M=15, N=30 -> 450 elems > 250
        (
            torch.tensor([10, 10, 10]),
            torch.tensor([5, 5, 5]),
            100,
            1000,  # 250 elems
            [(slice(0, 2), slice(0, 10)), (slice(2, 3), slice(0, 5))],
        ),
    ],
)
def test_split_indexer_prefill_chunks(
    seq_lens, query_lens, workspace_size, max_logits_bytes, expected
):
    out = split_indexer_prefill_chunks(
        seq_lens,
        query_lens,
        workspace_size,
        max_logits_bytes,
    )
    assert out == expected


def test_split_indexer_prefill_chunks_single_request_overflow():
    """Test that single request exceeding budget is sub-chunked on query dim."""
    seq_lens = torch.tensor([1000, 50])
    query_lens = torch.tensor([100, 5])

    out = split_indexer_prefill_chunks(seq_lens, query_lens, 2000, 1000)
    # max_logits_elems = 250, N=1000 -> max_q = 1 -> 100 query sub-chunks
    expected = [(slice(0, 1), slice(i, i + 1)) for i in range(100)]
    # req1: M=5, N=50 -> 250 elems fits budget
    expected.append((slice(1, 2), slice(0, 5)))
    assert out == expected


# 384 is not a power of two, so it counts via the tiled atomic accumulation
# rather than the single-tile path 128 takes.
@pytest.mark.parametrize("num_topk_tokens", [128, 384])
@cuda_required
def test_triton_convert_returns_valid_counts(num_topk_tokens: int):
    """Test that return_valid_counts correctly counts non-negative indices."""
    device = torch.device(DEVICE_TYPE)
    num_tokens = 8
    num_requests = 2
    max_blocks_per_req = 10
    block_size = 64

    req_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device)
    block_table = torch.arange(
        num_requests * max_blocks_per_req, dtype=torch.int32, device=device
    ).view(num_requests, max_blocks_per_req)

    # Create token indices with varying numbers of valid entries: half the row,
    # a quarter of it, the whole row, then a single valid entry -- twice over.
    token_indices = torch.full(
        (num_tokens, num_topk_tokens), -1, dtype=torch.int32, device=device
    )
    valid_counts_per_token = [
        num_topk_tokens // 2,
        num_topk_tokens // 4,
        num_topk_tokens,
        1,
    ] * 2
    expected_valid = []
    for i in range(num_tokens):
        num_valid = valid_counts_per_token[i]
        token_indices[i, :num_valid] = torch.arange(
            num_valid, dtype=torch.int32, device=device
        ) % (block_size * max_blocks_per_req)
        expected_valid.append(num_valid)

    expected_valid_tensor = torch.tensor(
        expected_valid, dtype=torch.int32, device=device
    )

    # Test with return_valid_counts=True
    result, valid_counts = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
        return_valid_counts=True,
    )

    torch.testing.assert_close(valid_counts, expected_valid_tensor, rtol=0, atol=0)

    # Test that return_valid_counts=False returns only the indices
    result_only = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
        return_valid_counts=False,
    )
    assert isinstance(result_only, torch.Tensor)
    for row, num_valid in enumerate(expected_valid):
        compact_valid = result[row, :num_valid].sort().values
        original_valid = result_only[row][result_only[row] >= 0].sort().values
        torch.testing.assert_close(compact_valid, original_valid, rtol=0, atol=0)
        assert torch.all(result[row, num_valid:] == -1)


@cuda_required
def test_flashmla_cache_dtype_aliases_use_ds_layout():
    from vllm.model_executor.layers.attention.mla_attention import (
        _canonicalize_sparse_mla_kv_cache_dtype,
    )

    # kv-cache dtype aliases are canonicalized to fp8_ds_mla before the layer
    # stores kv_cache_dtype, so they cannot bypass the gate.
    for alias in ("fp8", "fp8_e4m3"):
        assert (
            _canonicalize_sparse_mla_kv_cache_dtype(FlashMLASparseBackend, alias)
            == "fp8_ds_mla"
        )


@cuda_required
def test_flashmla_fp8_metadata_reuses_common_batch_split():
    builder = SimpleNamespace(
        device=torch.device(DEVICE_TYPE),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=8)),
    )
    common_metadata = SimpleNamespace(
        num_actual_tokens=1,
        seq_lens_cpu_upper_bound=torch.tensor([1]),
        query_start_loc_cpu=torch.tensor([0, 1]),
        block_table_tensor=torch.zeros(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
    )
    metadata = FlashMLASparseMetadata(
        num_reqs=1,
        max_query_len=1,
        max_seq_len=1,
        num_actual_tokens=1,
        query_start_loc=torch.tensor([0, 1], device=DEVICE_TYPE),
        slot_mapping=torch.tensor([0], device=DEVICE_TYPE),
        block_table=torch.zeros(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
        req_id_per_token=torch.zeros(1, dtype=torch.int32, device=DEVICE_TYPE),
        num_decodes=0,
        num_prefills=1,
        num_decode_tokens=0,
    )

    fp8_metadata = FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode(
        builder, common_metadata, metadata
    )

    assert fp8_metadata.num_decodes == 0
    assert fp8_metadata.num_prefills == 1
    assert fp8_metadata.num_decode_tokens == 0
    assert fp8_metadata.num_prefill_tokens == 1


@cuda_required
def test_flashmla_common_metadata_requires_uniform_decodes():
    common_metadata = SimpleNamespace(
        max_query_len=3,
        num_reqs=3,
        num_actual_tokens=6,
        query_start_loc_cpu=torch.tensor([0, 1, 3, 6]),
        is_prefilling=None,
    )

    split = split_decodes_and_prefills(
        common_metadata,
        decode_threshold=128,
        require_uniform=FlashMLASparseMetadataBuilder.require_uniform_decodes,
    )

    assert split == (1, 2, 1, 5)


@cuda_required
def test_flashmla_fp8_metadata_excludes_zero_token_decode_padding(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashmla_sparse.get_mla_metadata",
        lambda: (object(), None),
    )
    builder = SimpleNamespace(
        device=torch.device(DEVICE_TYPE),
        dummy_block_table=torch.zeros(7, 1, device=DEVICE_TYPE),
        max_model_len_tensor=torch.zeros(7, device=DEVICE_TYPE),
    )
    query_start_loc_cpu = torch.tensor([0, 110, 220, 330, 440, 550, 660, 660])
    common_metadata = SimpleNamespace(
        num_actual_tokens=660,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=torch.arange(7, device=DEVICE_TYPE),
    )
    metadata = FlashMLASparseMetadata(
        num_reqs=7,
        max_query_len=110,
        max_seq_len=110,
        num_actual_tokens=660,
        query_start_loc=query_start_loc_cpu.to(DEVICE_TYPE),
        slot_mapping=torch.arange(660, device=DEVICE_TYPE),
        block_table=torch.zeros(7, 1, dtype=torch.int32, device=DEVICE_TYPE),
        req_id_per_token=torch.zeros(660, dtype=torch.int32, device=DEVICE_TYPE),
        num_decodes=7,
        num_prefills=0,
        num_decode_tokens=660,
    )

    fp8_metadata = FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode(
        builder, common_metadata, metadata
    )

    assert fp8_metadata.num_decodes == 6
    assert fp8_metadata.num_decode_tokens == 660
    assert fp8_metadata.decode is not None
    assert fp8_metadata.decode.decode_query_len == 110
    torch.testing.assert_close(
        fp8_metadata.decode.seq_lens, torch.arange(6, device=DEVICE_TYPE)
    )


@pytest.mark.parametrize("use_mixed_batch", [False, True])
@cuda_required
def test_flashmla_fp8_paths_accept_decode_subset(monkeypatch, use_mixed_batch: bool):
    num_decode_tokens = 2
    num_batch_tokens = 5
    q = torch.empty(num_decode_tokens, 2, 3, device=DEVICE_TYPE)
    topk_indices = torch.empty(num_decode_tokens, 4, device=DEVICE_TYPE)
    kernel_q_shapes = []

    def convert_indices(*args, **kwargs):  # noqa: ARG001
        assert not kwargs.get("HAS_PREFILL_WORKSPACE", False)
        if not kwargs.get("return_valid_counts", False):
            return topk_indices
        valid_counts = torch.full(
            (num_decode_tokens,), 4, dtype=torch.int32, device=DEVICE_TYPE
        )
        return topk_indices, valid_counts

    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashmla_sparse."
        "triton_convert_req_index_to_global_index",
        convert_indices,
    )

    def run_kernel(**kwargs):
        kernel_q_shapes.append(kwargs["q"].shape)
        return kwargs["q"][..., :1], None

    if use_mixed_batch:
        fp8_metadata = FlashMLASparseMetadata.FP8KernelMetadata(
            scheduler_metadata=object(),  # type: ignore[arg-type]
            dummy_block_table=torch.empty(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
            cache_lens=torch.empty(1, dtype=torch.int32, device=DEVICE_TYPE),
        )
    else:
        FP8Meta = FlashMLASparseMetadata.FP8SeparatePrefillDecode
        fp8_metadata = FP8Meta(
            num_decodes=1,
            num_prefills=1,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_batch_tokens - num_decode_tokens,
            decode=FP8Meta.Decode(
                seq_lens=torch.empty(1, dtype=torch.int32, device=DEVICE_TYPE),
                kernel_metadata=object(),  # type: ignore[arg-type]
                decode_query_len=num_decode_tokens,
            ),
            prefill=FP8Meta.Prefill(
                request_ids=torch.empty(
                    num_batch_tokens, dtype=torch.int32, device=DEVICE_TYPE
                ),
                workspace_starts=torch.empty(1, dtype=torch.int32, device=DEVICE_TYPE),
                chunks=[],
            ),
        )
    metadata = SimpleNamespace(
        fp8_extra_metadata=fp8_metadata,
        fp8_use_mixed_batch=use_mixed_batch,
        num_actual_tokens=num_batch_tokens,
        req_id_per_token=torch.empty(
            num_batch_tokens, dtype=torch.int32, device=DEVICE_TYPE
        ),
        block_table=torch.empty(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
        block_size=64,
    )
    impl = SimpleNamespace(
        kv_cache_dtype="fp8_ds_mla",
        topk_indices_buffer=topk_indices,
        num_heads=2,
        kv_lora_rank=1,
        dcp_world_size=1,
        need_to_return_lse_for_decode=False,
        _fp8_flash_mla_kernel=run_kernel,
    )
    impl._forward_fp8_kv_mixed_batch = MethodType(
        FlashMLASparseImpl._forward_fp8_kv_mixed_batch, impl
    )
    impl._forward_fp8_kv_separate_prefill_decode = MethodType(
        FlashMLASparseImpl._forward_fp8_kv_separate_prefill_decode, impl
    )

    output, lse = FlashMLASparseImpl.forward_mqa(
        impl,
        q,
        torch.empty(0, device=DEVICE_TYPE),
        metadata,
        None,
    )

    assert kernel_q_shapes == [(1, num_decode_tokens, 2, 3)]
    assert output.shape == (num_decode_tokens, 2, 1)
    assert lse is None


def _build_sparse_dcp_vllm_config(
    local_heads: int,
    dcp_world_size: int,
    comm_backend: str = "ag_rs",
):
    """Minimal sparse-MLA VllmConfig for the FlashMLASparse DCP head-envelope
    guard. TP is simulated by mocking ``get_num_attention_heads`` to return the
    per-rank head count, as the decode-correctness test above does.
    """
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    head_size = kv_lora_rank + qk_rope_head_dim
    topk_tokens = 128

    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        tensor_parallel_size=1,
        max_model_len=4096,
        block_size=64,
        hf_config_override={
            "index_topk": topk_tokens,
            "attn_module_list_cfg": [{"topk_tokens": topk_tokens}],
        },
    )
    model_config = vllm_config.model_config
    model_config.dtype = torch.bfloat16
    model_config.hf_text_config = SimpleNamespace(
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        model_type="deepseek_v2",
    )
    model_config.get_num_attention_heads = MethodType(
        lambda self, parallel_config: local_heads, model_config
    )
    model_config.get_num_kv_heads = MethodType(
        lambda self, parallel_config: 1, model_config
    )
    model_config.get_head_size = MethodType(lambda self: head_size, model_config)
    model_config.get_sliding_window = MethodType(lambda self: None, model_config)

    vllm_config.cache_config.cache_dtype = "fp8_ds_mla"
    vllm_config.parallel_config.decode_context_parallel_size = dcp_world_size
    vllm_config.parallel_config.dcp_comm_backend = comm_backend
    # The base builder clones the layer's dense-MHA prefill backend from
    # static_forward_context; the guard tests never run prefill.
    vllm_config.compilation_config.static_forward_context["placeholder"] = (
        SimpleNamespace(prefill_backend=None)
    )
    return vllm_config


@pytest.mark.skipif(
    _cuda_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
@pytest.mark.parametrize(
    "local_heads,dcp_world_size,should_raise",
    [
        (16, 8, True),
        (24, 4, True),
        (16, 4, False),
        (16, 1, False),
    ],
)
def test_fp8_dcp_head_envelope_guard(local_heads, dcp_world_size, should_raise):
    """The fp8 decode envelope (head padding + tile-scheduler metadata) is
    sized from the local head count while the kernel runs on the DCP-gathered
    heads, so the builder must reject configs where the two pad differently.
    """
    device = torch.device(DEVICE_TYPE)
    vllm_config = _build_sparse_dcp_vllm_config(local_heads, dcp_world_size)
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    builder_cls = FlashMLASparseBackend.get_builder_cls()

    if should_raise:
        with pytest.raises(NotImplementedError, match="envelope"):
            builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)
    else:
        builder = builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)
        gathered_heads = local_heads * dcp_world_size
        local_pad = 64 if local_heads <= 64 else 128
        gathered_pad = 64 if gathered_heads <= 64 else 128
        assert builder.fp8_decode_padded_heads == local_pad
        assert local_pad == gathered_pad


@cuda_required
def test_fp8_mixed_batch_dcp_neutralizes_empty_rows(monkeypatch):
    """A decode row whose top-k shard holds no local candidates (all -1) has
    undefined kernel out/lse; it must come back as (0, -inf), the identity of
    the cross-rank LSE merge, or a NaN would survive the merge even at zero
    weight (0 * NaN = NaN)."""
    num_tokens, num_heads, head_dim = 3, 2, 3
    q = torch.empty(num_tokens, num_heads, head_dim, device=DEVICE_TYPE)
    local_indices = torch.tensor(
        [[0, 1, -1, -1], [-1, -1, -1, -1], [2, -1, 3, -1]],
        dtype=torch.int32,
        device=DEVICE_TYPE,
    )

    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashmla_sparse."
        "triton_filter_and_convert_dcp_index",
        lambda *args, **kwargs: local_indices,
    )

    def run_kernel(**kwargs):
        out = torch.full(
            (1, num_tokens, num_heads, 1), float("nan"), device=DEVICE_TYPE
        )
        lse = torch.full((1, num_heads, num_tokens), float("nan"), device=DEVICE_TYPE)
        for token_id in (0, 2):  # rows with local candidates get real values
            out[0, token_id] = float(token_id + 1)
            lse[0, :, token_id] = float(token_id + 1)
        return out, lse

    metadata = SimpleNamespace(
        fp8_extra_metadata=FlashMLASparseMetadata.FP8KernelMetadata(
            scheduler_metadata=object(),  # type: ignore[arg-type]
            dummy_block_table=torch.empty(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
            cache_lens=torch.empty(1, dtype=torch.int32, device=DEVICE_TYPE),
        ),
        req_id_per_token=torch.empty(num_tokens, dtype=torch.int32, device=DEVICE_TYPE),
        block_table=torch.empty(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
        block_size=64,
        cp_kv_cache_interleave_size=1,
    )
    impl = SimpleNamespace(
        dcp_world_size=2,
        dcp_rank=0,
        need_to_return_lse_for_decode=True,
        _fp8_flash_mla_kernel=run_kernel,
    )

    out, lse = FlashMLASparseImpl._forward_fp8_kv_mixed_batch(
        impl, q, torch.empty(0, device=DEVICE_TYPE), local_indices, metadata
    )

    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.isneginf(lse[1]).all()
    for token_id in (0, 2):
        assert torch.equal(out[token_id], torch.full_like(out[token_id], token_id + 1))
        assert torch.equal(lse[token_id], torch.full_like(lse[token_id], token_id + 1))
    assert out.is_contiguous()
    assert not out.isnan().any()
    assert not lse.isnan().any()


# ---------------------------------------------------------------------------
# Task 12 — sm80 backend selection: glm5next (head_size 512) vs DSV4 (576)
# ---------------------------------------------------------------------------
# These tests exercise the REAL selection machinery on any host (CPU included):
# ``CudaPlatformBase.get_valid_backends`` runs each candidate's
# ``validate_configuration`` and returns (valid list, structured invalid
# reasons). The only CPU-unavailable pieces are the engine-level
# ``get_attn_backend_cls`` (it reads the *host* device capability) and the
# kernel execution paths, both of which are GPU-gated above.

from vllm.platforms.cuda import CudaPlatformBase  # noqa: E402
from vllm.platforms.interface import DeviceCapability  # noqa: E402
from vllm.v1.attention.backends.registry import (  # noqa: E402
    AttentionBackendEnum,
)
from vllm.v1.attention.selector import AttentionSelectorConfig  # noqa: E402


def _sparse_selection(
    capability: DeviceCapability,
    head_size: int,
    kv_cache_dtype: str = "auto",
    **kw,
):
    """Run the REAL selection machinery for a capability/head-size/kv dtype.

    ``kv_cache_dtype`` is an explicit parameter (default ``"auto"``) rather
    than a ``**kw`` passthrough, so the issue-54059 regression matrix can
    vary it without the duplicate-key ``TypeError`` that ``**kw`` would raise
    (``AttentionSelectorConfig`` already names the field).
    """
    cfg = AttentionSelectorConfig(
        head_size=head_size,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        block_size=64,
        use_mla=True,
        use_sparse=True,
        **kw,
    )
    valid, invalid = CudaPlatformBase.get_valid_backends(
        capability, cfg, num_heads=32
    )
    return [b.backend for b in valid], invalid, cfg


# NOTE: every caller of _sparse_selection must take the
# ``selection_vllm_config`` fixture. On hosts where flashinfer is importable
# (e.g. the campaign image), get_valid_backends reaches
# FlashInferMLABackend.supports_combination, which calls
# get_current_vllm_config() and raises AssertionError outside a
# set_current_vllm_config context (incident 2026-08-31 wave-5 G-build: on the
# CPU dev host flashinfer is absent, so the path was never reached and the
# bare tests passed).
@pytest.fixture()
def selection_vllm_config():
    """VllmConfig context for selection probes on accelerated hosts.

    On CUDA platforms it yields inside set_current_vllm_config so
    backend.validate_configuration/supports_combination can read a config.
    On CPU-only hosts VllmConfig() itself raises RuntimeError (device type
    inference fails without an accelerator), and no context is needed there
    anyway because flashinfer is unimportable and the config-dependent path
    is never reached."""
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        yield None
        return
    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    with set_current_vllm_config(config):
        yield config


def _selection_invalid_reason(
    backend: AttentionBackendEnum,
    invalid: dict,
    substring: str,
) -> bool:
    if backend not in invalid:
        return False
    return any(substring in r for r in invalid[backend][1])


def test_task12_auto_selection_sm80_head512_resolves_triton_sparse(selection_vllm_config):
    """glm5next (head_size 512) on sm80 auto-selects TRITON_MLA_SPARSE."""
    valid, invalid, _ = _sparse_selection(DeviceCapability(8, 0), 512)
    assert valid == [AttentionBackendEnum.TRITON_MLA_SPARSE], valid
    # The higher-priority sparse candidates are rejected with STRUCTURED
    # reasons (the issue-54059 failure class is a crash; these are clean
    # rejections, never exceptions).
    assert _selection_invalid_reason(
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
        invalid,
        "compute capability",
    ), invalid
    assert _selection_invalid_reason(
        AttentionBackendEnum.FLASHMLA_SPARSE, invalid, "head_size"
    ), invalid
    assert _selection_invalid_reason(
        AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,
        invalid,
        "compute capability",
    ), invalid


def test_task12_auto_selection_sm80_head576_preserves_dsv4(selection_vllm_config):
    """DSV4 (head_size 576) on sm80 still resolves to TRITON_MLA_SPARSE."""
    valid, invalid, _ = _sparse_selection(DeviceCapability(8, 0), 576)
    assert valid == [AttentionBackendEnum.TRITON_MLA_SPARSE], valid
    # FLASHMLA_SPARSE supports 576 but not sm80 -> clean capability rejection,
    # never a crash.
    assert _selection_invalid_reason(
        AttentionBackendEnum.FLASHMLA_SPARSE, invalid, "compute capability"
    ), invalid


def test_task12_backend_priority_order_512_vs_576():
    """SM90 sparse is preferred FIRST for 512 (glm5next NoPE) and LAST for
    576 (DSV4), so DSV4's existing sparse preference order is preserved."""
    from vllm.platforms.cuda import _get_backend_priorities

    p512 = [
        b.name
        for b in _get_backend_priorities(
            use_mla=True,
            device_capability=DeviceCapability(8, 0),
            num_heads=32,
            kv_cache_dtype="auto",
            use_non_causal=False,
            head_size=512,
        )
    ]
    p576 = [
        b.name
        for b in _get_backend_priorities(
            use_mla=True,
            device_capability=DeviceCapability(8, 0),
            num_heads=32,
            kv_cache_dtype="auto",
            use_non_causal=False,
            head_size=576,
        )
    ]
    # 512: SM90 sparse before FlashAttn/FlashMLA/Triton sparse (preferred).
    assert p512.index("FLASHINFER_MLA_SPARSE_SM90") < p512.index(
        "TRITON_MLA_SPARSE"
    )
    # 576: SM90 sparse AFTER the reference DSV4 tail (fallback, not a
    # preference change). Both keep the dense head identical.
    assert p576[-1] == "FLASHINFER_MLA_SPARSE_SM90"
    dense_head = [
        "FLASH_ATTN_MLA",
        "FLASHMLA",
        "FLASHINFER_MLA",
        "TRITON_MLA",
    ]
    assert p512[:4] == dense_head
    assert p576[:4] == dense_head


def test_task12_forced_triton_sparse_accepted_on_sm80(selection_vllm_config):
    """Forcing TRITON_MLA_SPARSE on sm80 (512) is accepted (no reasons)."""
    _, _, cfg = _sparse_selection(DeviceCapability(8, 0), 512)
    reasons = AttentionBackendEnum.TRITON_MLA_SPARSE.get_class().validate_configuration(
        device_capability=DeviceCapability(8, 0), **cfg._asdict()
    )
    assert reasons == [], reasons


def test_task12_forced_sm90_sparse_cleanly_rejected_on_sm80(selection_vllm_config):
    """Forcing FLASHINFER_MLA_SPARSE_SM90 on sm80 is a CLEAN rejection
    (structured reasons), never an exception — the issue-54059 failure class."""
    _, _, cfg = _sparse_selection(DeviceCapability(8, 0), 512)
    reasons = (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90.get_class().validate_configuration(
            device_capability=DeviceCapability(8, 0), **cfg._asdict()
        )
    )
    assert any("compute capability" in r for r in reasons), reasons


def test_task12_forced_flashmla_sparse_rejected_for_head512(selection_vllm_config):
    """Forcing FLASHMLA_SPARSE with head_size 512 is rejected on head_size
    (the native kernel only serves 576), regardless of capability."""
    _, _, cfg = _sparse_selection(DeviceCapability(9, 0), 512)
    reasons = (
        AttentionBackendEnum.FLASHMLA_SPARSE.get_class().validate_configuration(
            device_capability=DeviceCapability(9, 0), **cfg._asdict()
        )
    )
    assert any("head_size" in r for r in reasons), reasons


def test_task12_issue54059_regression_never_raises(selection_vllm_config):
    """Regression for vllm-project/vllm#54059 (sm8x head_size=512 sparse-MLA
    backend-selection failure).

    Full capability x head_size x kv_cache_dtype matrix. For every combo,
    selection must NEVER raise: it either resolves a usable backend or
    returns structured invalid reasons. The resolved backend, when present,
    must be the portable TRITON_MLA_SPARSE (the sm80 path for
    glm5next/GLM-5.3), and the SM90/FlashMLA sparse entries must be rejected
    as invalid reasons, not crash.

    kv_cache_dtype is varied explicitly (auto / bfloat16 / fp8 / fp8_ds_mla).
    fp8 and fp8_ds_mla are rejected by TRITON_MLA_SPARSE (which only serves
    [auto, bfloat16]): that rejection is BY DESIGN -- the matrix asserts a
    structured invalid-reason entry, never an exception.
    """
    kv_cache_dtypes = ["auto", "bfloat16", "fp8", "fp8_ds_mla"]
    for cap in (
        DeviceCapability(8, 0),
        DeviceCapability(8, 6),
        DeviceCapability(8, 9),
    ):
        for hs in (512, 576):
            for kv in kv_cache_dtypes:
                valid, invalid, _ = _sparse_selection(cap, hs, kv)
                # Never raises is the contract; reaching here proves it.
                if valid:
                    assert valid == [AttentionBackendEnum.TRITON_MLA_SPARSE], (
                        f"cap={cap} hs={hs} kv={kv} resolved {valid}"
                    )
                else:
                    # No usable backend: must be a structured invalid-reason
                    # dictionary, and TRITON_MLA_SPARSE must be present with a
                    # concrete reason (so a future capability/head-size
                    # regression that silently drops the sm80 path is caught).
                    assert AttentionBackendEnum.TRITON_MLA_SPARSE in invalid, (
                        f"cap={cap} hs={hs} kv={kv}: Triton sparse dropped"
                    )
                    assert _selection_invalid_reason(
                        AttentionBackendEnum.TRITON_MLA_SPARSE,
                        invalid,
                        "kv_cache_dtype",
                    ), invalid


def test_task12_cpu_bootstrap_restores_globals():
    """The CPU-only bootstrap must restore every global it touched.

    This runs in the SAME pytest process as the rest of the module, so it
    directly proves the restore contract: after the collection-time bootstrap
    ran (and any later re-invocation below), ``vllm.triton_utils.tl`` and the
    two stub module entries are byte-identical (by identity) to their
    pristine pre-bootstrap values. The snapshots read the ACTUAL globals
    (``sys.modules`` / ``getattr(tu, 'tl')``) rather than the bootstrap's
    bookkeeping dicts.
    """
    import sys

    import vllm.triton_utils as tu
    from vllm.platforms import current_platform

    # Snapshot pristine ACTUAL state (distinguishing absent from None).
    orig_tl = getattr(tu, "tl", _ABSENT)
    orig_mods = {
        name: sys.modules.get(name, _ABSENT) for name in _STUB_MODULE_NAMES
    }

    # Re-run the bootstrap exactly as at collection time (including the
    # install-guarded restore), then assert restoration.
    installed = _install_cpu_stubs()
    try:
        # Exercise the full import surface the bootstrap protects.
        import vllm.v1.attention.backends.mla.flashinfer_mla_sparse  # noqa: F401
        import vllm.v1.attention.backends.mla.flashmla_sparse  # noqa: F401
    finally:
        # Guarded exactly like the module bootstrap: on a CUDA platform
        # nothing was installed, so nothing may be restored (an unguarded
        # restore would clobber vllm.triton_utils.tl to the uncaptured
        # _STUB_TL_STATE).
        if installed:
            _restore_cpu_stubs()

    # Identity assertions on the ACTUAL globals: the exact same objects (or
    # the same absence) must be back in place.
    assert getattr(tu, "tl", _ABSENT) is orig_tl, (
        "vllm.triton_utils.tl was not restored"
    )
    for name, prev in orig_mods.items():
        assert sys.modules.get(name, _ABSENT) is prev, (
            f"sys.modules[{name!r}] was not restored to its pre-bootstrap object"
        )

    # And the stub path must not have run on this host's platform in a way
    # that left any residue: if we're on CUDA it must never have installed.
    if current_platform.is_cuda():
        assert installed is False, "stubs must never install on a CUDA platform"


def _snapshot_actual_globals():
    """Snapshot the REAL globals the bootstrap touches, as (value, absent?) pairs.

    Returns ``(mods, tl)`` where ``mods`` maps each stub module name to
    ``(value, was_absent)`` and ``tl`` is ``(value, was_absent)``. "absent"
    is tracked separately because ``sys.modules`` may legitimately contain a
    ``None`` entry and ``tl`` may be present-as-``None``.
    """
    import sys

    mods = {
        name: (sys.modules.get(name, _ABSENT), name not in sys.modules)
        for name in _STUB_MODULE_NAMES
    }
    if _tu is None:
        tl_value = _ABSENT
        tl_absent = True
    else:
        tl_value = getattr(_tu, "tl", _ABSENT)
        tl_absent = not hasattr(_tu, "tl")
    return mods, (tl_value, tl_absent)


def _restore_actual_globals(pristine):
    """Restore the ACTUAL globals to a captured pristine snapshot.

    Puts back exact identities (or absence) so a test is fully
    order-independent regardless of what prior globals were.
    """
    import sys

    mods, (tl_val, tl_absent) = pristine
    for name in _STUB_MODULE_NAMES:
        (val, was_absent) = mods[name]
        if was_absent:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = val
    if _tu is not None:
        if tl_absent:
            if hasattr(_tu, "tl"):
                delattr(_tu, "tl")
        else:
            _tu.tl = tl_val


def _assert_globals_unchanged(before, after):
    """Assert the ACTUAL globals after == before, identity-preserving."""
    mods_before, (tl_before, tl_absent_before) = before
    mods_after, (tl_after, tl_absent_after) = after

    # sys.modules entries: identity (or the same absence) must hold.
    for name in _STUB_MODULE_NAMES:
        (v_before, absent_before) = mods_before[name]
        (v_after, absent_after) = mods_after[name]
        assert absent_after is absent_before, (
            f"sys.modules[{name!r}] presence flipped"
        )
        assert v_after is v_before, (
            f"sys.modules[{name!r}] value identity changed"
        )

    # vllm.triton_utils.tl: identity (or the same absence) must hold.
    assert tl_absent_after is tl_absent_before, (
        "vllm.triton_utils.tl presence flipped"
    )
    assert tl_after is tl_before, (
        "vllm.triton_utils.tl value identity changed"
    )


def test_task12_cuda_gated_bootstrap_leaves_actual_globals_unchanged():
    """A CUDA-gated ``_install_cpu_stubs()`` must not mutate the real globals.

    Covers the three cases for each global explicitly: (a) absent before,
    (b) present-as-``None`` before, (c) present-with-object before (identity
    preserved). Each case asserts the ACTUAL ``sys.modules`` entry and
    ``vllm.triton_utils.tl`` attribute, not the bootstrap's bookkeeping.
    """
    from unittest import mock

    from vllm.platforms import current_platform

    import sys  # isort:skip (local helper needs stdlib before vllm imports)

    def _case(name, mutate, assert_setup):
        """Run one sub-case: set up a prior global state, then verify a
        CUDA-gated ``_install_cpu_stubs()`` leaves the ACTUAL globals exactly
        as they were.

        Each case snapshots the pristine baseline, applies its own ``mutate``,
        proves the setup took effect (``assert_setup``), calls the CUDA-gated
        install, asserts the real globals are unchanged, then restores the
        pristine baseline so the suite stays order-independent.
        """
        pristine = _snapshot_actual_globals()
        try:
            mutate()
            assert_setup()  # prove this case's prior state is genuinely in place
            before = _snapshot_actual_globals()
            with mock.patch.object(
                current_platform, "is_cuda", return_value=True
            ):
                assert _install_cpu_stubs() is False
            after = _snapshot_actual_globals()
            _assert_globals_unchanged(before, after)
        finally:
            _restore_actual_globals(pristine)

    # (a) Absent before: every stub module entry and tl are absent.
    def _absent():
        for name in _STUB_MODULE_NAMES:
            sys.modules.pop(name, None)
        if _tu is not None and hasattr(_tu, "tl"):
            delattr(_tu, "tl")

    def _assert_absent():
        for name in _STUB_MODULE_NAMES:
            assert name not in sys.modules, f"{name} should be absent"
        assert not hasattr(_tu, "tl"), "tl should be absent"

    _case("absent", _absent, _assert_absent)

    # (b) Present-as-None before: entries exist with value None.
    def _none():
        for name in _STUB_MODULE_NAMES:
            sys.modules[name] = None
        if _tu is not None:
            _tu.tl = None

    def _assert_none():
        for name in _STUB_MODULE_NAMES:
            assert name in sys.modules, f"{name} should be present"
            assert sys.modules[name] is None, f"{name} should be None"
        assert hasattr(_tu, "tl") and _tu.tl is None, "tl should be present-None"

    _case("present-none", _none, _assert_none)

    # (c) Present-with-object before (identity must be preserved).
    obj = object()
    mod_obj = {name: object() for name in _STUB_MODULE_NAMES}

    def _objects():
        for name in _STUB_MODULE_NAMES:
            sys.modules[name] = mod_obj[name]
        if _tu is not None:
            _tu.tl = obj

    def _assert_objects():
        for name in _STUB_MODULE_NAMES:
            assert sys.modules[name] is mod_obj[name], (
                f"{name} identity setup broken"
            )
        assert _tu.tl is obj, "tl identity setup broken"

    _case("present-object", _objects, _assert_objects)

    # Post-collection residue must match the pre-collection actual state
    # (the collection-time bootstrap restored byte-for-byte). This is a
    # direct check: running the full install+restore cycle leaves the real
    # globals identical to the pristine baseline.
    baseline = _snapshot_actual_globals()
    _install_cpu_stubs()
    _restore_cpu_stubs()
    _assert_globals_unchanged(baseline, _snapshot_actual_globals())


def test_task12_stub_path_never_runs_on_cuda():
    """On a CUDA platform the stub-install path must never execute.

    The bootstrap is gated on ``current_platform.is_cuda()`` (the vLLM
    platform), NOT ``torch.cuda.is_available()``. This proves that even a
    CUDA-platform host with transiently unavailable Torch CUDA would load the
    real extensions: ``_install_cpu_stubs`` returns False and mutates nothing
    when the platform reports CUDA.
    """
    from unittest import mock

    from vllm.platforms import current_platform

    import sys  # isort:skip (local helper needs stdlib before vllm imports)

    if not current_platform.is_cuda():
        # CPU host: directly assert the gate by simulating a CUDA platform,
        # and assert the ACTUAL globals (sys.modules entries and
        # vllm.triton_utils.tl) are unchanged.
        before = _snapshot_actual_globals()
        with mock.patch.object(
            current_platform, "is_cuda", return_value=True
        ):
            assert _install_cpu_stubs() is False
        _assert_globals_unchanged(before, _snapshot_actual_globals())
    else:
        # Real CUDA host: the stub path must simply never have installed, so
        # the real compiled extensions (or absence) must occupy those names.
        # The bootstrap stub is a bare types.ModuleType with no __file__; a
        # real extension loads from a .so and always has one. Identity-by-name
        # is NOT a stub signal on CUDA: the real modules legitimately sit in
        # sys.modules under their own names.
        assert _install_cpu_stubs() is False
        for name in _STUB_MODULE_NAMES:
            entry = sys.modules.get(name, _ABSENT)
            if entry is _ABSENT or not isinstance(entry, _types.ModuleType):
                continue
            assert getattr(entry, "__file__", None), (
                f"bootstrap stub installed on CUDA: {name!r} has no __file__"
            )
        if _tu is not None:
            assert not hasattr(_tu, "tl") or not isinstance(
                _tu.tl, _types.SimpleNamespace
            ), "stub tl must never be installed on CUDA"


def test_task12_cuda_restore_never_clobbers_tl():
    """Regression: the CUDA-gated bootstrap pair must leave the real globals
    byte-identical.

    Incident 2026-08-31 (wave-5 G-build, first real-GPU run): on a CUDA
    platform ``_install_cpu_stubs()`` returns False WITHOUT snapshotting the
    globals, and the collection-time ``finally: _restore_cpu_stubs()`` then
    wrote the module-level initial ``_STUB_TL_STATE = None`` into
    ``vllm.triton_utils.tl``. Every later module-scope ``tl.constexpr``
    import in the same pytest process failed with
    ``AttributeError: 'NoneType' object has no attribute 'constexpr'``. The
    bootstrap now gates the restore on an actual install
    (``_STUBS_INSTALLED``); this test re-runs the gated pair with a mocked
    CUDA platform and asserts nothing changed.
    """
    from unittest import mock

    from vllm.platforms import current_platform

    before = _snapshot_actual_globals()
    with mock.patch.object(current_platform, "is_cuda", return_value=True):
        installed = _install_cpu_stubs()
    assert installed is False
    # The bootstrap's guarded shape: restore runs only when stubs were
    # installed. Emulate it exactly; with no install it must be skipped.
    if installed:
        _restore_cpu_stubs()
    _assert_globals_unchanged(before, _snapshot_actual_globals())

    # And document the hazard the guard prevents: on a CUDA-gated run the
    # snapshot globals were never captured, so an UNGUARDED restore would
    # clobber. _STUB_TL_STATE must still be the uncaptured initial (None),
    # never the _ABSENT sentinel -- the guard key is the installed flag, not
    # the state value.
    assert _STUB_TL_STATE is None
    assert _STUB_TL_STATE is not _ABSENT
