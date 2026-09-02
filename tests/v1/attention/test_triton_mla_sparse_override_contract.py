# SPDX-License-Identifier: Apache-2.0
# FileCopyrightText: Copyright contributors to the vLLM project
"""TritonMLASparseImpl override contract (design rev 7 section 4.4).

An inspect-based CPU test asserting the override binds every keyword the
parent call site passes: FlashMLASparseImpl calls
``self._bf16_flash_mla_kernel(q, kv_rows, topk_indices, topk_length,
actual_num_heads)`` positionally; TritonMLASparseImpl overrides it with
optional trailing parameters. If either side drifts (a new parent
argument, a renamed override parameter), binding FAILS here in CI rather
than exploding at boot. Runs where the backend chain imports; skips on
hosts without it (executes in-image at T26).
"""

from __future__ import annotations

import inspect

import pytest

try:
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseImpl,
    )
    from vllm.v1.attention.backends.mla.triton_mla_sparse import (
        TritonMLASparseImpl,
    )

    _IMPORTABLE = True
    _REASON = "importable"
except Exception as _exc:  # CPU hosts without compiled ops / triton
    _IMPORTABLE = False
    _REASON = repr(_exc)

pytestmark = pytest.mark.skipif(not _IMPORTABLE, reason=f"backend chain not importable: {_REASON}")


def test_triton_override_binds_every_parent_call_argument() -> None:
    """The parent's positional call site must bind against the child's
    signature: parent drift or override drift fails here, not at boot."""
    child_sig = inspect.signature(TritonMLASparseImpl._bf16_flash_mla_kernel)
    # the parent call site (FlashMLASparseImpl._forward_bf16_kv) passes
    # exactly these five positional arguments (self excluded)
    bound = child_sig.bind(
        None,  # self (unbound for signature purposes)
        "q",
        "kv_rows",
        "topk_indices",
        "topk_length",
        "actual_num_heads",
    )
    assert set(bound.arguments) - {"self"} == {
        "q",
        "kv_c_and_k_pe_cache",
        "topk_indices",
        "topk_length",
        "actual_num_heads",
    }
    # the prefill chunk call site passes four (no gathered-heads argument);
    # the override's None default absorbs it
    bound4 = child_sig.bind(None, "q", "kv", "topk", "topk_length")
    bound4.apply_defaults()
    assert bound4.arguments["actual_num_heads"] is None


def test_parent_call_site_argument_count_is_pinned() -> None:
    """The parent source passes exactly five positional arguments; if the
    parent grows or shrinks the call, this pin (and the bind test above)
    force the override contract to be re-examined."""
    src = inspect.getsource(FlashMLASparseImpl._forward_bf16_kv)
    call = "self._bf16_flash_mla_kernel("
    assert call in src
    # the two known call sites (decode + prefill chunk) each pass 5 args
    import re

    calls = re.findall(r"_bf16_flash_mla_kernel\(([^)]*)\)", src)
    assert calls, "parent call site disappeared"
    for c in calls:
        # drop the trailing comma (the call sites use it); count
        # top-level commas (no nested calls with commas in the current
        # call sites; a nested-call future changes this pin loudly)
        c = c.strip().rstrip(",").strip()
        depth = 0
        args = 1
        for ch in c:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif ch == "," and depth == 0:
                args += 1
        assert args == 5, f"parent call site now passes {args} args: {c!r}"


def test_override_signature_documented_contract() -> None:
    """The override's optional parameters carry the HEAD-parent default
    contract (DCP-gathered heads): topk_length and actual_num_heads default
    to None, never silently clamped."""
    child_sig = inspect.signature(TritonMLASparseImpl._bf16_flash_mla_kernel)
    assert child_sig.parameters["topk_length"].default is None
    assert child_sig.parameters["actual_num_heads"].default is None
