# SPDX-License-Identifier: Apache-2.0
"""Fail-closed regression for missing probabilistic MTP draft_probs.

Loads mtp_draft_probs.py by path so collection does not import the full
vLLM package (cbor2 / CUDA). GPUModelRunner._sample must call the same
guard; this file reconstructs the cache-miss that produced None.
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
MISSING_MSG = "MTP probabilistic draft sampling requires draft probability rows"


def _get_spec_decode_draft_probs(runner, spec_decode_metadata):
    """Port of GPUModelRunner._get_spec_decode_draft_probs (fork + 1Cat)."""
    if runner._draft_probs is None or runner._draft_prob_req_ids is None:
        return None
    row_by_req_id = {
        req_id: idx for idx, req_id in enumerate(runner._draft_prob_req_ids)
    }
    rows = []
    for req_id, num_draft in zip(
        runner.input_batch.req_ids, spec_decode_metadata.num_draft_tokens
    ):
        if num_draft == 0:
            continue
        row_idx = row_by_req_id.get(req_id)
        if row_idx is None:
            return None
        rows.append(object())
    if not rows:
        return None
    return rows


def test_mixed_prefill_and_resumed_decode_missing_draft_probs_is_fail_closed():
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["resume-decode", "new-prefill"]),
        _draft_probs=object(),
        _draft_prob_req_ids=["resume-decode"],
    )
    metadata = SimpleNamespace(num_draft_tokens=[2, 2])
    draft_probs = _get_spec_decode_draft_probs(runner, metadata)
    assert draft_probs is None
    speculative = SimpleNamespace(method="mtp", draft_sample_method="probabilistic")
    sampling = SimpleNamespace(all_greedy=False)
    with pytest.raises(RuntimeError, match=MISSING_MSG):
        require_mtp_draft_probs(speculative, sampling, draft_probs)


def test_async_tp_aggregation_skips_proposal_without_probability_rows():
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["tp-rank0-req"]),
        _draft_probs=None,
        _draft_prob_req_ids=None,
    )
    metadata = SimpleNamespace(num_draft_tokens=[1])
    assert _get_spec_decode_draft_probs(runner, metadata) is None
    speculative = SimpleNamespace(method="mtp", draft_sample_method="probabilistic")
    sampling = SimpleNamespace(all_greedy=False)
    with pytest.raises(RuntimeError, match=MISSING_MSG):
        require_mtp_draft_probs(speculative, sampling, None)


def test_greedy_may_omit_draft_probs():
    speculative = SimpleNamespace(method="mtp", draft_sample_method="probabilistic")
    sampling = SimpleNamespace(all_greedy=True)
    require_mtp_draft_probs(speculative, sampling, None)


def test_tp_rank_req_id_misalignment_is_fail_closed():
    """TP aggregation must not sample when rank-local cache ids diverge."""
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["global-req-0", "global-req-1"]),
        _draft_probs=object(),
        _draft_prob_req_ids=["rank0-local-0"],
    )
    metadata = SimpleNamespace(num_draft_tokens=[1, 1])
    assert _get_spec_decode_draft_probs(runner, metadata) is None
    speculative = SimpleNamespace(method="mtp", draft_sample_method="probabilistic")
    sampling = SimpleNamespace(all_greedy=False)
    with pytest.raises(RuntimeError, match=MISSING_MSG):
        require_mtp_draft_probs(speculative, sampling, None)


def test_prefill_only_zero_draft_tokens_does_not_need_a_row():
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["resume-decode", "new-prefill"]),
        _draft_probs=object(),
        _draft_prob_req_ids=["resume-decode"],
    )
    metadata = SimpleNamespace(num_draft_tokens=[2, 0])
    assert _get_spec_decode_draft_probs(runner, metadata) is not None
