# SPDX-License-Identifier: Apache-2.0
# FileCopyrightText: Copyright contributors to the vLLM project
"""sm80 fp8 e4m3fn encoder: bitwise CPU verification (design rev 7 4.4).

Checks in the vector generator + reference definitions and asserts bitwise
equality of the ALU encode path (``_f32_to_e4m3fn_u8`` — the sm80 lane,
where Triton refuses native fp8 converts below SM89) against the torch
``float8_e4m3fn`` reference over:

- the full 256-code lattice (decode -> re-encode round trip),
- every adjacent-pair tie midpoint (RNE: ties round to even),
- uniform and log-uniform random sweeps,
- subnormals, +-0, +-448, overflow (satfinite), inf, NaN (compared
  sign-insensitively: the kernel preserves the NaN sign, torch
  canonicalizes it).

Executed on CPU via the Triton interpreter (TRITON_INTERPRET=1); collected
on every host (the ops import is deferred to call time), executed in-image
at T26 where the compiled chain is importable.
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

_E4M3_MAX = 448.0

# Bound at launch time from the ops module: Triton resolves jit-function
# free names through this module's globals, so the kernel below calls the
# ALU encoder without importing the compiled chain at collection time.
_f32_to_e4m3fn_u8 = None


def _encoder():
    """Deferred import so collection succeeds on hosts without the chain."""
    try:
        import vllm.v1.attention.ops.fp8_sm80 as f8
    except Exception as exc:  # no compiled chain on this host
        pytest.skip(f"fp8_sm80 chain not importable: {exc!r}")
    return f8


def _reference_encode(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn).view(torch.uint8)


def _reference_decode(u: torch.Tensor) -> torch.Tensor:
    return u.view(torch.float8_e4m3fn).to(torch.float32)


def _run_encoder(f8, x: torch.Tensor) -> torch.Tensor:
    """Launch the ALU encoder over a CPU buffer through the interpreter."""
    import triton
    import triton.language as tl

    global _f32_to_e4m3fn_u8
    _f32_to_e4m3fn_u8 = f8._f32_to_e4m3fn_u8

    @triton.jit
    def _encode_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        xv = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = _f32_to_e4m3fn_u8(xv)
        tl.store(out_ptr + offs, y, mask=mask)

    x = x.contiguous().to(torch.float32)
    out = torch.empty(x.numel(), dtype=torch.uint8)
    BLOCK = 1024
    n = x.numel()
    _encode_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK)
    return out


def _assert_bitwise(f8, x: torch.Tensor) -> None:
    got = _run_encoder(f8, x)
    ref = _reference_encode(x)
    # NaN compared sign-insensitively (kernel preserves NaN sign; the torch
    # reference canonicalizes it)
    ref_nan = torch.isnan(ref.view(torch.float8_e4m3fn).to(torch.float32))
    got_nan = torch.isnan(got.view(torch.float8_e4m3fn).to(torch.float32))
    assert torch.equal(got_nan, ref_nan), "NaN classification mismatch"
    got_cmp = torch.where(ref_nan, got & 0x7F, got)
    ref_cmp = torch.where(ref_nan, ref & 0x7F, ref)
    mismatch = torch.nonzero(got_cmp != ref_cmp)[:8].flatten().tolist()
    assert torch.equal(got_cmp, ref_cmp), f"bitwise mismatch at indices {mismatch}"


def test_full_256_code_lattice_roundtrip() -> None:
    """Every representable byte decodes to f32 and re-encodes to itself."""
    f8 = _encoder()
    codes = torch.arange(256, dtype=torch.uint8)
    values = _reference_decode(codes)
    got = _run_encoder(f8, values)
    # NaN codes (0x7E is max finite; 0x7F/0xFF are NaN) re-encode to a NaN
    # code; compare sign-insensitively there, exactly elsewhere
    nan_mask = torch.isnan(values)
    got_cmp = torch.where(nan_mask, got & 0x7F, got)
    codes_cmp = torch.where(nan_mask, codes & 0x7F, codes)
    assert torch.equal(got_cmp, codes_cmp)


def test_adjacent_pair_tie_midpoints_round_to_even() -> None:
    """The midpoint of every adjacent representable pair rounds RNE: the
    even code wins the tie, matching the torch reference bit for bit."""
    f8 = _encoder()
    codes = torch.arange(256, dtype=torch.uint8)
    vals = _reference_decode(codes)
    finite = vals[~torch.isnan(vals)]
    finite, _ = torch.sort(finite)
    midpoints = (finite[:-1] + finite[1:]) / 2
    # exact fp32 midpoints (representable sums are exact for e4m3 pairs)
    _assert_bitwise(f8, midpoints)


def test_uniform_and_log_uniform_sweeps() -> None:
    f8 = _encoder()
    g = torch.Generator().manual_seed(20260901)
    uni = (torch.rand(65536, generator=g) * 2 * _E4M3_MAX) - _E4M3_MAX
    log = torch.exp(
        torch.rand(65536, generator=g) * (torch.log(torch.tensor(_E4M3_MAX * 2)) - torch.log(torch.tensor(1e-6))) + torch.log(torch.tensor(1e-6))
    ) * torch.where(torch.rand(65536, generator=g) < 0.5, -1.0, 1.0)
    _assert_bitwise(f8, uni)
    _assert_bitwise(f8, log)


def test_subnormals_zeros_max_overflow_inf_nan() -> None:
    f8 = _encoder()
    sub = torch.tensor(
        [2**-9, 3 * 2**-9, 2**-8, 2**-7, 2**-6 - 2**-9], dtype=torch.float32
    )
    special = torch.tensor(
        [
            0.0, -0.0, _E4M3_MAX, -_E4M3_MAX,
            _E4M3_MAX + 1.0, -_E4M3_MAX - 1.0,  # satfinite
            float("inf"), float("-inf"),
            float("nan"), -float("nan"),
            2**-133, -(2**-133),  # f32-subnormal region -> rounds to zero
        ],
        dtype=torch.float32,
    )
    _assert_bitwise(f8, sub)
    _assert_bitwise(f8, special)
