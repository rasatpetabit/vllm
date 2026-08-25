# SPDX-License-Identifier: Apache-2.0
"""Fail-closed probabilistic MTP draft-probability guard.

Deployed 1Cat 1.2.2 raises this error from GPUModelRunner._sample when
method=mtp, draft_sample_method=probabilistic, sampling is non-greedy,
and cached draft_probs cannot cover every request with num_draft_tokens>0.
The fork previously passed draft_probs=None into rejection sampling, which
is an invalid silent-accept path. Keep this module free of vLLM runtime
imports so unit tests can load it without the full package.
"""

from __future__ import annotations

from typing import Any

MISSING_DRAFT_PROBS_MSG = (
    "MTP probabilistic draft sampling requires draft probability rows "
    "for exact rejection sampling. Missing draft_probs would "
    "silently fall back to an invalid no-draft-probability "
    "acceptance path and can corrupt output quality."
)


def require_mtp_draft_probs(
    speculative_config: Any,
    sampling_metadata: Any,
    draft_probs: Any,
) -> None:
    if speculative_config is None:
        return
    if getattr(speculative_config, "method", None) != "mtp":
        return
    if getattr(speculative_config, "draft_sample_method", None) != "probabilistic":
        return
    if getattr(sampling_metadata, "all_greedy", False):
        return
    if draft_probs is None:
        raise RuntimeError(MISSING_DRAFT_PROBS_MSG)
