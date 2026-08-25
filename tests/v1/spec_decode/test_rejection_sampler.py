# SPDX-License-Identifier: Apache-2.0
"""CPU-only guard: probabilistic MTP must not reach rejection sampling without rows.

GPU distribution tests live in test_rejection_sampler_utils.py and need CUDA.
This file pins the fail-closed contract that Wave 9 wired into _sample:
missing draft_probs is an error, not a silent no-probability accept.
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

RUNNER = (
    Path(__file__).resolve().parents[3] / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
)


def test_missing_probs_never_reach_rejection_sampler_fallback():
    speculative = SimpleNamespace(method="mtp", draft_sample_method="probabilistic")
    sampling = SimpleNamespace(all_greedy=False)
    with pytest.raises(RuntimeError, match="draft probability rows"):
        require_mtp_draft_probs(speculative, sampling, None, [2])


def test_sample_path_calls_require_mtp_draft_probs():
    text = RUNNER.read_text(encoding="utf-8")
    assert "collect_draft_prob_rows" in text
    assert "require_mtp_draft_probs(" in text
    assert "self.speculative_config" in text
