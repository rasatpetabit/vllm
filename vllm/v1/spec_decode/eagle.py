# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import VllmConfig
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


class EagleProposer(SpecDecodeBaseProposer):
    # Probabilistic draft_probs cache coverage is enforced in
    # GPUModelRunner._sample via require_mtp_draft_probs. Mixed
    # prefill+decode skips live in llm_base_proposer / the runner
    # cache key, not in this thin subclass.

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
