# SPDX-License-Identifier: Apache-2.0
"""Red reproducer for missing probabilistic MTP draft_probs.

Deployed 1Cat 1.2.2 (`gpu_model_runner.py` around the rejection-sampler
call) raises:

    MTP probabilistic draft sampling requires draft probability rows
    for exact rejection sampling.

The fork currently returns None from `_get_spec_decode_draft_probs` when a
mixed new-prefill + resumed non-greedy decode batch has cached rows that
do not cover every request with `num_draft_tokens > 0` (the resumed decode
row is the missing one). Prefill-only requests with zero draft tokens are
skipped and do not require a probability row. Async scheduling and TP
aggregation are the live trigger conditions, not extra requirements of
this unit reconstruction. This module is intentionally red: it
reconstructs that cache miss and re-raises the deployed error so Wave 8
verify (`pytest` nonzero + message grep) records the defect. Do not wrap
in pytest.raises here. Wave 9 repairs the source and turns this into a
passing regression.
"""

from __future__ import annotations

from types import SimpleNamespace

MISSING_MSG = (
    "MTP probabilistic draft sampling requires draft probability rows "
    "for exact rejection sampling. Missing draft_probs would "
    "silently fall back to an invalid no-draft-probability "
    "acceptance path and can corrupt output quality."
)


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


def require_mtp_draft_probs(speculative_config, sampling_metadata, draft_probs):
    if speculative_config is None:
        return
    if getattr(speculative_config, "method", None) != "mtp":
        return
    if getattr(speculative_config, "draft_sample_method", None) != "probabilistic":
        return
    if getattr(sampling_metadata, "all_greedy", False):
        return
    if draft_probs is None:
        raise RuntimeError(MISSING_MSG)


def test_mixed_prefill_and_resumed_decode_missing_draft_probs_is_fail_closed():
    """New prefill + resumed decode: cached rows do not cover the new request."""
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
    # Uncaught: Wave 8 verify requires pytest rc != 0 and this exact message.
    require_mtp_draft_probs(speculative, sampling, draft_probs)


def test_async_tp_aggregation_skips_proposal_without_probability_rows():
    """Async scheduling + TP aggregation: empty cache is not a silent fallback."""
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["tp-rank0-req"]),
        _draft_probs=None,
        _draft_prob_req_ids=None,
    )
    metadata = SimpleNamespace(num_draft_tokens=[1])
    assert _get_spec_decode_draft_probs(runner, metadata) is None
    speculative = SimpleNamespace(method="mtp", draft_sample_method="probabilistic")
    sampling = SimpleNamespace(all_greedy=False)
    require_mtp_draft_probs(speculative, sampling, None)
