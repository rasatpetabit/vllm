# SPDX-License-Identifier: Apache-2.0
# FileCopyrightText: Copyright contributors to the vLLM project
"""kpool indexer MQA-logits Triton fallback: GPU numerics (design rev 7 4.4).

Pinned seeds, shapes, head-counts, and dtypes; the Triton
``fp8_mqa_logits_triton`` (the sm80 lane — no DeepGEMM below Hopper) is
compared against a dequantized float32 reference:

- logits: relative error with an absolute floor near zero,
- top-k indices: exact agreement with the stated tie rule (descending
  value, ties broken by the lower row index — i.e. a stable descending
  argsort).

GPU-gated (skips without CUDA; an unavailable GPU in the W1 in-image run
is a GATE FAILURE there, not a skip — the T26 no-skip gate enforces it).
Collected on every host (the ops import is deferred to call time).
"""

from __future__ import annotations

import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="kpool MQA-logits numerics need a GPU"
)


def _ops():
    try:
        from vllm.v1.attention.ops.mqa_logits_triton import fp8_mqa_logits_triton
    except Exception as exc:  # no compiled chain on this host
        pytest.skip(f"mqa_logits chain not importable: {exc!r}")
    return fp8_mqa_logits_triton


def _make_case(seed: int, m: int, n: int, num_heads: int, head_dim: int):
    """Build the EXACT fp8 tensors the kernel consumes. The reference below
    dequantizes THESE tensors — never the pre-quantization f32 draws — so
    the comparison measures the kernel against its own inputs."""
    g = torch.Generator().manual_seed(seed)
    q_f32 = (torch.randn(m, num_heads, head_dim, generator=g) * 8).clamp(-448, 448)
    k_f32 = (torch.randn(n, head_dim, generator=g) * 8).clamp(-448, 448)
    q = q_f32.to(torch.float8_e4m3fn).cuda()
    k = k_f32.to(torch.float8_e4m3fn).cuda()
    scales = torch.rand(n, generator=g).cuda() * 0.5 + 0.5
    weights = torch.randn(m, num_heads, generator=g).cuda()
    # windowed rows: ks in [0, n-1], ke in (ks, n]
    ks = torch.sort(torch.randint(0, max(n - 1, 1), (m,), generator=g))[0].int().cuda()
    ke = (ks + torch.randint(1, max(n - ks.max().item(), 1), (m,), generator=g)).clamp(max=n).int().cuda()
    return q, k, scales, weights, ks, ke


def _reference_logits(q, k_fp8, scales, weights, ks, ke):
    """Float32 reference over each row's [ks, ke) window, built from the
    DEQUANTIZED ACTUAL fp8 inputs (q, k_fp8) — the same bytes the kernel
    consumes — scaled exactly as the kernel scales them."""
    q_deq = q.float()
    k_deq = k_fp8.float()
    m, num_heads, head_dim = q_deq.shape
    n = k_deq.shape[0]
    out = torch.full((m, n), float("-inf"), dtype=torch.float32, device=q.device)
    k_scaled = k_deq * scales[:, None]
    for i in range(m):
        ks_i = int(ks[i].item())
        ke_i = int(ke[i].item())
        window = k_scaled[ks_i:ke_i]
        # [H, D] x [D, W] -> [H, W]
        dots = q_deq[i] @ window.t()
        out[i, ks_i:ke_i] = (dots * weights[i][:, None]).sum(dim=0)
    return out


@requires_cuda
@pytest.mark.parametrize(
    ("seed", "m", "n", "num_heads", "head_dim", "k"),
    [
        (20260901, 8, 512, 1, 576, 64),
        (20260902, 16, 1024, 4, 128, 128),
        (20260903, 3, 384, 2, 64, 16),
    ],
)
def test_kpool_mqa_logits_matches_reference(seed, m, n, num_heads, head_dim, k) -> None:
    fp8_mqa_logits_triton = _ops()
    q, k_fp8, scales, weights, ks, ke = _make_case(seed, m, n, num_heads, head_dim)
    got = fp8_mqa_logits_triton(
        q,
        (k_fp8, scales),
        weights,
        ks,
        ke,
        clean_logits=True,
    ).float()
    ref = _reference_logits(q, k_fp8, scales, weights, ks, ke)

    assert got.shape == ref.shape == (m, n)
    finite = torch.isfinite(ref)
    assert finite.any(), "reference window is empty"
    diff = (got - ref).abs()
    rel = diff / ref.abs().clamp_min(1e-3)
    # relative error with an absolute floor near zero: exact-equal windows
    # (both -inf outside [ks, ke)) and tiny logits never fail the rel gate
    assert torch.equal(torch.isfinite(got), finite), "window mask mismatch"
    in_window = finite
    max_rel = (rel[in_window]).max().item()
    assert max_rel <= 2e-2, f"max relative error {max_rel} over window"

    # top-k index agreement with the stated tie rule: descending value,
    # ties to the lower index (stable descending argsort)
    for i in range(m):
        row_got = got[i]
        row_ref = ref[i]
        top_got = torch.argsort(row_got, descending=True, stable=True)[:k]
        top_ref = torch.argsort(row_ref, descending=True, stable=True)[:k]
        # indices outside the window are all -inf; ordering among them is
        # irrelevant — compare the window-prefix
        window_len = (ke[i] - ks[i]).item()
        overlap = min(k, window_len)
        assert torch.equal(top_got[:overlap], top_ref[:overlap]), (
            f"row {i}: top-{overlap} mismatch\n got={top_got[:overlap].tolist()}\n ref={top_ref[:overlap].tolist()}"
        )


@requires_cuda
def test_kpool_clean_logits_is_parity_only_output_fully_determined() -> None:
    """The implementation writes EVERY row element (in-window logits,
    explicit -inf outside [ks, ke)); ``clean_logits`` is accepted for
    DeepGEMM signature parity only. Contract pinned here: both flag values
    produce the SAME fully-determined output — -inf outside every window,
    bit-identical inside."""
    fp8_mqa_logits_triton = _ops()
    q, k_fp8, scales, weights, ks, ke = _make_case(20260904, 4, 256, 1, 128, 32)
    got_clean = fp8_mqa_logits_triton(q, (k_fp8, scales), weights, ks, ke, clean_logits=True).float()
    got_parity = fp8_mqa_logits_triton(q, (k_fp8, scales), weights, ks, ke, clean_logits=False).float()
    assert torch.equal(got_clean, got_parity), "clean_logits must not change the output"
    for i in range(got_clean.shape[0]):
        row = got_clean[i]
        w = slice(int(ks[i]), int(ke[i]))
        outside = torch.ones(row.shape[0], dtype=torch.bool, device=row.device)
        outside[w] = False
        assert torch.all(row[outside] == float("-inf")), (
            f"row {i}: outside-window elements must be -inf (fully-determined output)"
        )
