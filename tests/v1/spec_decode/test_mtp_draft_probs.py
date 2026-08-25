# SPDX-License-Identifier: Apache-2.0
"""Mixed-batch skip for missing probabilistic MTP draft_probs.

Loads mtp_draft_probs.py by path so collection does not import the full
vLLM package (cbor2 / CUDA). Uncovered requests drop to zero draft tokens
instead of collapsing the whole batch to None and crashing EngineCore.
Remaining drafts without rows still fail closed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_HELPER = (
    Path(__file__).resolve().parents[3]
    / "vllm"
    / "v1"
    / "spec_decode"
    / "mtp_draft_probs.py"
)
_SPEC = importlib.util.spec_from_file_location("mtp_draft_probs", _HELPER)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
require_mtp_draft_probs = _MOD.require_mtp_draft_probs
collect_draft_prob_rows = _MOD.collect_draft_prob_rows
MISSING_MSG = "MTP probabilistic draft sampling requires draft probability rows"


def _get_spec_decode_draft_probs(runner, spec_decode_metadata):
    rows, _skipped = collect_draft_prob_rows(
        runner.input_batch.req_ids,
        spec_decode_metadata.num_draft_tokens,
        runner._draft_prob_req_ids,
        (lambda idx, n: object()) if runner._draft_probs is not None else None,
    )
    return rows or None


def test_mixed_prefill_skips_uncovered_request_instead_of_collapsing_batch():
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["resume-decode", "new-prefill"]),
        _draft_probs=object(),
        _draft_prob_req_ids=["resume-decode"],
    )
    metadata = SimpleNamespace(num_draft_tokens=[2, 2])
    draft_probs = _get_spec_decode_draft_probs(runner, metadata)
    assert metadata.num_draft_tokens == [2, 0]
    assert draft_probs is not None and len(draft_probs) == 1
    speculative = SimpleNamespace(method="mtp", draft_sample_method="probabilistic")
    sampling = SimpleNamespace(all_greedy=False)
    require_mtp_draft_probs(
        speculative, sampling, draft_probs, metadata.num_draft_tokens
    )


def test_empty_cache_skips_all_drafts_instead_of_raising():
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["tp-rank0-req"]),
        _draft_probs=None,
        _draft_prob_req_ids=None,
    )
    metadata = SimpleNamespace(num_draft_tokens=[1])
    assert _get_spec_decode_draft_probs(runner, metadata) is None
    assert metadata.num_draft_tokens == [0]
    speculative = SimpleNamespace(method="mtp", draft_sample_method="probabilistic")
    sampling = SimpleNamespace(all_greedy=False)
    require_mtp_draft_probs(speculative, sampling, None, metadata.num_draft_tokens)


def test_greedy_may_omit_draft_probs():
    speculative = SimpleNamespace(method="mtp", draft_sample_method="probabilistic")
    sampling = SimpleNamespace(all_greedy=True)
    require_mtp_draft_probs(speculative, sampling, None, [2])


def test_tp_rank_req_id_misalignment_skips_uncovered_ranks():
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["global-req-0", "global-req-1"]),
        _draft_probs=object(),
        _draft_prob_req_ids=["rank0-local-0"],
    )
    metadata = SimpleNamespace(num_draft_tokens=[1, 1])
    assert _get_spec_decode_draft_probs(runner, metadata) is None
    assert metadata.num_draft_tokens == [0, 0]
    speculative = SimpleNamespace(method="mtp", draft_sample_method="probabilistic")
    sampling = SimpleNamespace(all_greedy=False)
    require_mtp_draft_probs(speculative, sampling, None, metadata.num_draft_tokens)


def test_inclusive_cumsum_has_batch_length_not_a_leading_zero():
    compact = _MOD.compact_uncovered_drafts
    metadata = SimpleNamespace(
        num_draft_tokens=[3, 0, 2],
        cu_num_draft_tokens=[3, 3, 5],
        draft_token_ids=[1, 2, 3, 4, 5],
        target_logits_indices=[0, 1, 2, 5, 6],
        max_spec_len=3,
    )
    compact(metadata, ["a", "b", "c"], ["a", "c"])
    assert metadata.num_draft_tokens == [3, 0, 2]
    assert metadata.cu_num_draft_tokens == [3, 3, 5]
    assert len(metadata.cu_num_draft_tokens) == len(metadata.num_draft_tokens)
    assert metadata.cu_num_draft_tokens[0] != 0 or metadata.num_draft_tokens[0] == 0


def test_missing_probability_tensor_uncovers_cached_ids_too():
    compact = _MOD.compact_uncovered_drafts
    metadata = SimpleNamespace(
        num_draft_tokens=[2, 2],
        cu_num_draft_tokens=[2, 4],
        draft_token_ids=[10, 11, 20, 21],
        target_logits_indices=[0, 1, 5, 6],
        max_spec_len=2,
    )
    compact(metadata, ["resume-decode", "new-prefill"], ["resume-decode"], rows_available=False)
    assert metadata.num_draft_tokens == [0, 0]
    assert metadata.cu_num_draft_tokens == [0, 0]
    assert metadata.draft_token_ids == []
    assert metadata.target_logits_indices == []


def test_skip_compacts_cu_and_draft_token_ids_not_just_counts():
    compact = _MOD.compact_uncovered_drafts
    metadata = SimpleNamespace(
        num_draft_tokens=[2, 2],
        cu_num_draft_tokens=[2, 4],
        draft_token_ids=[10, 11, 20, 21],
        target_logits_indices=[0, 1, 5, 6],
        max_spec_len=2,
    )
    keep = compact(
        metadata,
        ["resume-decode", "new-prefill"],
        ["resume-decode"],
    )
    assert metadata.num_draft_tokens == [2, 0]
    assert metadata.cu_num_draft_tokens == [2, 2]
    assert metadata.draft_token_ids == [10, 11]
    assert metadata.target_logits_indices == [0, 1]
    assert metadata.max_spec_len == 2
    assert keep == [True, False]


def test_fixed_logit_row_count_matches_remaining_drafts():
    num_draft = [2, 2, 0]
    rows, skipped = collect_draft_prob_rows(
        ["a", "b", "c"],
        num_draft,
        ["a"],
        lambda idx, n: (idx, n),
    )
    assert skipped == ["b"]
    assert num_draft == [2, 0, 0]
    assert rows == [(0, 2)]


def test_remaining_drafts_without_rows_still_fail_closed():
    speculative = SimpleNamespace(method="mtp", draft_sample_method="probabilistic")
    sampling = SimpleNamespace(all_greedy=False)
    with pytest.raises(RuntimeError, match=MISSING_MSG):
        require_mtp_draft_probs(speculative, sampling, None, [2, 0])


def test_prefill_only_zero_draft_tokens_does_not_need_a_row():
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["resume-decode", "new-prefill"]),
        _draft_probs=object(),
        _draft_prob_req_ids=["resume-decode"],
    )
    metadata = SimpleNamespace(num_draft_tokens=[2, 0])
    assert _get_spec_decode_draft_probs(runner, metadata) is not None
