# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression tests for the DeepSeek-V4 compressed-MLA KV spec
translation onto the post-refactor layout system.

Pins the glm53-a100-port decision: dsv4's compressed fp8_ds_mla page is
expressed HEAD-natively with ``tokens_per_state=compress_ratio`` and
``state_content_bytes=584`` (448B NoPE + 128B RoPE + 8B fp8 scale), NOT via
the 01ecc8e4 ``compress_ratio``/``storage_block_size`` contract. The numbers
replicate the production geometry observed in the G-dsv4 wave (block 256,
ratio 4, BLHNC layout, fixed block stride 1534464) where the old contract
tripped ``compute_layout_strides``'s page-padding divisibility assert.
"""

import torch

from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    compute_layer_kv_cache_shape_bytes,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout


def _dsv4_fp8_spec(block_size: int = 256, compress_ratio: int = 4):
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        tokens_per_state=compress_ratio,
        state_content_bytes=584,
        cache_dtype_str="fp8_ds_mla",
        alignment=576,
        model_version="deepseek_v4",
    )


def test_dsv4_fp8_spec_content_and_shape():
    spec = _dsv4_fp8_spec()
    assert spec.state_content_size_bytes == 584
    assert compute_layer_kv_cache_shape_bytes(spec, num_blocks=256) == (
        256,
        1,
        64,
        584,
    )


def test_dsv4_fp8_page_sizing():
    spec = _dsv4_fp8_spec()
    # 64 stored states x 584B; padded to 576B alignment.
    assert spec.unpadded_page_size_bytes == 37376
    assert spec.page_size_padded == 37440
    assert spec.page_size_bytes == 37440


def test_dsv4_fp8_layout_strides_divisibility():
    # The exact call shape that asserted under the old contract.
    spec = _dsv4_fp8_spec()
    strides = compute_layout_strides(
        spec,
        num_blocks=256,
        num_layers=21,
        layout=KVCacheLayout.BLHNC,
        fixed_strides=(None, 1534464, None, None, None),
    )
    assert strides[1] == 1534464  # fixed block stride preserved
    assert strides[0] % spec.page_size_bytes == 0


def test_generic_compressed_sizing_without_fp8_override():
    # Non-fp8_ds_mla compressed MLA keeps the generic formula:
    # block_size / ratio x head_size x dtype_size.
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        tokens_per_state=4,
        alignment=512,
    )
    assert spec.unpadded_page_size_bytes == (256 // 4) * 512 * 2


def test_plain_and_state_content_specs_unchanged():
    plain = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
    )
    assert plain.tokens_per_state == 1
    assert plain.unpadded_page_size_bytes == 64 * 576 * 2

    kimi_like = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        state_content_bytes=656,
    )
    assert kimi_like.unpadded_page_size_bytes == 64 * 656


def test_merge_preserves_translation_fields():
    spec = _dsv4_fp8_spec()
    merged = MLAAttentionSpec.merge([spec, spec])
    assert merged.tokens_per_state == 4
    assert merged.state_content_bytes == 584
    assert merged.page_size_bytes == spec.page_size_bytes
