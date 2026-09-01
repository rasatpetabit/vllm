# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-selection capability matrix (adversarial review slice-4).

Pins the compute-capability declarations the base validator now enforces
(commit de345ac04): on sm80 the Hopper/Blackwell-only sparse-MLA backends
must declare themselves invalid so auto selection falls through to
TRITON_MLA_SPARSE, and the base default stays True for backends without an
override. The whole module skips on hosts where the backend modules cannot
be imported (no compiled ops / disabled triton).
"""

import pytest

from vllm.platforms.interface import DeviceCapability

try:
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (
        FlashAttnMLASparseBackend,
    )
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
        FlashInferMLASparseSM90Backend,
    )
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseBackend,
    )
    from vllm.v1.attention.backends.mla.triton_mla_sparse import (
        TritonMLASparseBackend,
    )
except Exception as exc:  # CPU hosts without compiled ops / triton
    pytest.skip(
        f"backend capability matrix requires importable backends: {exc}",
        allow_module_level=True,
    )

SM80 = DeviceCapability(8, 0)
SM90 = DeviceCapability(9, 0)
SM100 = DeviceCapability(10, 0)


@pytest.mark.parametrize(
    ("backend", "capability", "expected"),
    [
        # Hopper/Blackwell-only sparse MLA backends must reject sm80.
        (FlashMLASparseBackend, SM80, False),
        (FlashMLASparseBackend, SM90, True),
        (FlashMLASparseBackend, SM100, True),
        (FlashAttnMLASparseBackend, SM80, False),
        (FlashAttnMLASparseBackend, SM90, True),
        (FlashAttnMLASparseBackend, SM100, False),
        (FlashInferMLASparseSM90Backend, SM80, False),
        (FlashInferMLASparseSM90Backend, SM90, True),
        (FlashInferMLASparseSM90Backend, SM100, False),
        # The Triton fallback is the sm80 lane: it must stay valid everywhere.
        (TritonMLASparseBackend, SM80, True),
        (TritonMLASparseBackend, SM90, True),
        (TritonMLASparseBackend, SM100, True),
        # Non-sparse sanity: FA needs sm80+.
        (FlashAttentionBackend, SM80, True),
        (FlashAttentionBackend, DeviceCapability(7, 0), False),
    ],
)
def test_capability_declaration(backend, capability, expected) -> None:
    assert backend.supports_compute_capability(capability) is expected


def test_base_default_stays_true() -> None:
    # Backends without an override keep the historical accept-all default;
    # enforcing the check in validate_configuration must not reject them.
    assert AttentionBackend.supports_compute_capability(SM80) is True
    assert AttentionBackend.supports_compute_capability(SM90) is True
    assert AttentionBackend.supports_compute_capability(SM100) is True
