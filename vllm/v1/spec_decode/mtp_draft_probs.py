# SPDX-License-Identifier: Apache-2.0
"""Probabilistic MTP draft-probability coverage.

Uncovered requests in a mixed prefill+decode batch drop to zero draft
tokens instead of collapsing the whole batch to None (which crashes
EngineCore). Remaining drafts without rows still fail closed. Keep this
module free of vLLM runtime imports so unit tests can load it without
the full package.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

MISSING_DRAFT_PROBS_MSG = (
    "MTP probabilistic draft sampling requires draft probability rows "
    "for exact rejection sampling. Missing draft_probs would "
    "silently fall back to an invalid no-draft-probability "
    "acceptance path and can corrupt output quality."
)


def collect_draft_prob_rows(
    req_ids: Sequence[str],
    num_draft_tokens: list[int],
    cached_req_ids: Sequence[str] | None,
    row_at: Callable[[int, int], Any] | None,
) -> tuple[list[Any], list[str]]:
    """Collect cached probability rows; zero uncovered drafts in place.

    ``num_draft_tokens`` is mutated: requests without a cached row become 0.
    Returns (rows for remaining drafts, skipped request ids).
    """
    row_by_req_id = (
        {req_id: idx for idx, req_id in enumerate(cached_req_ids)}
        if cached_req_ids is not None
        else {}
    )
    rows: list[Any] = []
    skipped: list[str] = []
    for i, (req_id, num_draft) in enumerate(zip(req_ids, num_draft_tokens)):
        if num_draft == 0:
            continue
        row_idx = row_by_req_id.get(req_id)
        if row_idx is None or row_at is None:
            skipped.append(req_id)
            num_draft_tokens[i] = 0
            continue
        rows.append(row_at(row_idx, num_draft))
    return rows, skipped


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        data = value.tolist()
        return data if isinstance(data, list) else [data]
    return list(value)


def _restore(original: Any, values: list[Any]) -> Any:
    if original is None:
        return values
    if isinstance(original, list):
        original[:] = list(values)
        return original
    if hasattr(original, "new_tensor"):
        return original.new_tensor(values)
    return list(values)


def compact_uncovered_drafts(
    metadata: Any,
    req_ids: Sequence[str],
    cached_req_ids: Sequence[str] | None,
    *,
    rows_available: bool = True,
) -> list[bool]:
    """Drop uncovered requests' draft tokens from sampler metadata.

    Zeroing ``num_draft_tokens`` alone leaves ``cu_num_draft_tokens`` and
    ``draft_token_ids`` pointing at the skipped request's drafts. Compact
    those tensors (and ``target_logits_indices``) to the remaining tokens.
    Inclusive ``np.cumsum`` layout is ``[n0, n0+n1, ...]`` (batch-length, no
    leading zero). ``rows_available=False`` uncovers every request that still
    has draft tokens, matching collect_draft_prob_rows when the probability
    tensor is missing. Zero-draft requests stay covered: there is nothing to
    compact.
    """
    if len(req_ids) != len(metadata.num_draft_tokens):
        raise ValueError("req_ids and num_draft_tokens length mismatch")
    old_num = list(metadata.num_draft_tokens)
    cached = set(cached_req_ids or ())
    keep: list[bool] = []
    new_num: list[int] = []
    for req_id, num_draft in zip(req_ids, old_num):
        covered = num_draft == 0 or (rows_available and req_id in cached)
        keep.append(covered)
        new_num.append(num_draft if covered else 0)

    ids = _as_list(metadata.draft_token_ids) or []
    tli = _as_list(getattr(metadata, "target_logits_indices", None))
    new_ids: list[Any] = []
    new_tli: list[Any] = []
    if sum(old_num) > len(ids):
        raise ValueError("draft_token_ids shorter than declared num_draft_tokens")
    offset = 0
    for num_draft, covered in zip(old_num, keep):
        chunk = ids[offset : offset + num_draft]
        if covered:
            new_ids.extend(chunk)
            if tli is not None:
                new_tli.extend(tli[offset : offset + num_draft])
        offset += num_draft

    cu: list[int] = []
    acc = 0
    for num_draft in new_num:
        acc += num_draft
        cu.append(acc)

    if isinstance(metadata.num_draft_tokens, list):
        metadata.num_draft_tokens[:] = new_num
    else:
        metadata.num_draft_tokens = new_num
    metadata.cu_num_draft_tokens = _restore(metadata.cu_num_draft_tokens, cu)
    metadata.draft_token_ids = _restore(metadata.draft_token_ids, new_ids)
    if tli is not None:
        metadata.target_logits_indices = _restore(
            metadata.target_logits_indices, new_tli
        )
    metadata.max_spec_len = max(new_num) if new_num else 0
    return keep


def require_mtp_draft_probs(
    speculative_config: Any,
    sampling_metadata: Any,
    draft_probs: Any,
    num_draft_tokens: Sequence[int] | None = None,
) -> None:
    if speculative_config is None:
        return
    if getattr(speculative_config, "method", None) != "mtp":
        return
    if getattr(speculative_config, "draft_sample_method", None) != "probabilistic":
        return
    if getattr(sampling_metadata, "all_greedy", False):
        return
    remaining = 0
    if num_draft_tokens is not None:
        remaining = sum(int(n) for n in num_draft_tokens)
        if remaining == 0:
            return
    if draft_probs is None:
        raise RuntimeError(MISSING_DRAFT_PROBS_MSG)
