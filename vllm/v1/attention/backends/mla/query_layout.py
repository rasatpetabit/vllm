# SPDX-License-Identifier: Apache-2.0
# FileCopyrightText: Copyright contributors to the vLLM project
"""Query-layout capability for MLA backends (design rev 7 section 4.4).

Pure logic, importable on CPU hosts without the compiled ops chain:

- ``QueryLayout`` — the decode-query geometry an MLA layer requests, derived
  from ``qk_rope_head_dim`` (0 -> NoPE-only; otherwise rope).
- ``UnsupportedQueryLayout`` — the stable exception for layout-contract
  violations (never a bare ``assert``; ``python -O`` removes asserts).
- ``supported_query_layouts`` — the declaration surface every MLA backend
  must override; the base raises for undeclared MLA backends (fail closed:
  an unproven layout is omitted, a missing declaration refuses selection).
- ``sort_sparse_tail`` — the declaration-derived total ordering that
  replaces the identity ``head_size == 512`` branch in the platform
  priorities: a backend specialised to exactly the requested layout
  precedes general-purpose ones; the platform priority index is the next
  key; the backend name is the deterministic final tie-break.
"""

from __future__ import annotations

from enum import Enum


class QueryLayout(str, Enum):
    """The MLA decode-query geometry a layer requests."""

    NOPE_ONLY = "NOPE_ONLY"
    ROPE = "ROPE"

    def __str__(self) -> str:  # pragma: no cover - repr convenience
        return self.value


class UnsupportedQueryLayout(NotImplementedError):
    """A layout-contract violation: stable type so callers can catch it
    without string matching (design rev 7 section 4.4)."""


def query_layout_for(qk_rope_head_dim: int) -> QueryLayout:
    """Derive the requested layout from the configured rope dimension."""
    if qk_rope_head_dim == 0:
        return QueryLayout.NOPE_ONLY
    return QueryLayout.ROPE


def supported_query_layouts() -> frozenset[QueryLayout]:
    """Base declaration: an MLA backend that has not declared its layouts
    must NOT silently pass selection. Overridden by every MLA backend with
    the layouts a named kernel-contract test proves."""
    raise UnsupportedQueryLayout(
        "MLA backends must declare supported_query_layouts(); missing "
        "declaration refuses selection (fail closed)"
    )


def sort_sparse_tail(
    backends: list,
    requested: QueryLayout,
    declared: "dict",
    platform_priority_index: "dict",
) -> list:
    """Total, deterministic ordering of the sparse-MLA tail.

    Key: ``(0 if declared == {requested} else 1, platform_priority_index,
    backend_name)``. ``declared`` maps backend key -> frozenset[QueryLayout];
    an undeclared backend sorts last within its key group (never first).
    """
    def sort_key(backend):
        layouts = declared.get(backend)
        specialist = 0 if layouts is not None and layouts == frozenset({requested}) else 1
        return (
            specialist,
            platform_priority_index.get(backend, len(platform_priority_index)),
            str(getattr(backend, "name", getattr(backend, "value", backend))),
        )

    return sorted(backends, key=sort_key)
