# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn as nn

from tensorrt_llm._torch.attention_backend.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding

if TYPE_CHECKING:
    from .deepseek_v4 import DeepseekV4TrtllmAttentionMetadata


class KVCacheDtype(IntEnum):
    """KV cache dtype/layout preset (values match C++ cache_scale_type parameter).

    NONE = 0, FP8_PERTENSOR = 1, FP8_BLOCKWISE = 2, MXFP4_BLOCKWISE = 3
    """

    NONE = 0
    FP8_PERTENSOR = 1
    FP8_BLOCKWISE = 2
    MXFP4_BLOCKWISE = 3


_KV_CACHE_DTYPE_MAP = {
    "default": KVCacheDtype.NONE,
    "bf16": KVCacheDtype.NONE,
    "fp8_pertensor": KVCacheDtype.FP8_PERTENSOR,
    "fp8_blockwise": KVCacheDtype.FP8_BLOCKWISE,
    "mxfp4": KVCacheDtype.MXFP4_BLOCKWISE,
}


def resolve_kv_cache_dtype(kv_cache_dtype: Union[str, KVCacheDtype]) -> KVCacheDtype:
    if isinstance(kv_cache_dtype, str):
        return _KV_CACHE_DTYPE_MAP[kv_cache_dtype]
    return kv_cache_dtype


class Compressor(nn.Module):
    """KV compressor — thin Python wrapper over C++ compressor_forward op.

    Only holds weights for checkpoint loading; all compute (F.linear, 3 kernel
    launches, buffer allocations, scatter) is done in C++.
    """

    def __init__(
        self,
        mla_params: MLAParams,
        layer_idx: int,
        compress_ratio: int,
        norm_eps: float,
        skip_create_weights_in_init: bool,
        pos_embd_params: PositionalEmbeddingParams,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        kv_cache_dtype: Union[str, KVCacheDtype] = KVCacheDtype.NONE,
        is_indexer: bool = False,
        rotate_activation: bool = False,
    ):
        super().__init__()
        self.dim = mla_params.hidden_size
        self.head_dim = mla_params.qk_rope_head_dim + mla_params.qk_nope_head_dim
        self.rope_head_dim = mla_params.qk_rope_head_dim
        self.nope_head_dim = mla_params.qk_nope_head_dim

        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.state_dim = 2 * self.head_dim if self.overlap else self.head_dim

        self.layer_idx = layer_idx
        self.kv_cache_dtype: KVCacheDtype = resolve_kv_cache_dtype(kv_cache_dtype)
        self.is_indexer = is_indexer
        self.rotate_activation = rotate_activation

        self.wkv_gate = Linear(
            self.dim,
            self.state_dim * 2,
            bias=False,
            dtype=dtype,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )
        self.norm = RMSNorm(hidden_size=self.head_dim, eps=norm_eps, dtype=dtype)
        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_head_dim,
            is_neox=pos_embd_params.is_neox,
        )
        self.ape = nn.Parameter(torch.empty(compress_ratio, self.state_dim, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        metadata: "DeepseekV4TrtllmAttentionMetadata",
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        from .deepseek_v4 import DeepseekV4AttentionType

        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        bsz = num_contexts + num_generations

        if self.is_indexer:
            compress_type = DeepseekV4AttentionType.INDEXER_COMPRESS
            kv_type = DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV
            score_type = DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE
        else:
            compress_type = DeepseekV4AttentionType.COMPRESS
            kv_type = DeepseekV4AttentionType.COMPRESSOR_KV
            score_type = DeepseekV4AttentionType.COMPRESSOR_SCORE

        kv_cache = metadata.kv_cache_manager.get_buffers(self.layer_idx, compress_type)
        paged_kv_state = metadata.kv_cache_manager.get_buffers(self.layer_idx, kv_type)
        paged_score_state = metadata.kv_cache_manager.get_buffers(self.layer_idx, score_type)

        local_layer_idx = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
        if self.is_indexer:
            block_table = metadata.indexer_k_cache_block_offsets
        else:
            block_table = metadata.compress_block_tables[self.compress_ratio]
        block_table_kv_state = metadata.sliding_block_tables[local_layer_idx, kv_type.value]
        block_table_score_state = metadata.sliding_block_tables[local_layer_idx, score_type.value]

        state_tokens_per_block = metadata.kv_cache_manager.tokens_per_block
        compress_tokens_per_block = metadata.kv_cache_manager.compressed_block_sizes[self.layer_idx]

        cu_new_comp_kv = metadata.cu_new_comp_kv_cuda[self.compress_ratio]
        kv_lens = metadata.kv_lens_cuda_runtime
        total_num_comp_tokens = metadata.num_total_compressed_tokens[self.compress_ratio]
        num_comp_tokens = metadata.new_comp_kv_lens_cuda[self.compress_ratio][:bsz]
        max_ctx_comp_kv_lens = metadata.max_ctx_compressed_tokens[self.compress_ratio]

        position_ids = metadata.compressed_position_ids_cuda[self.compress_ratio][:total_num_comp_tokens]
        compressed_mask = metadata.compressed_mask_cuda[self.compress_ratio][:total_num_comp_tokens]

        next_n = metadata.num_gen_tokens_per_seq if num_generations > 0 else 0

        primary_output, scale_output = torch.ops.trtllm.compressor_forward(
            x,
            self.wkv_gate.weight,
            self.norm.weight,
            float(self.norm.variance_epsilon),
            self.ape,
            self.rotary_emb.rotary_cos_sin,
            paged_kv_state,
            paged_score_state,
            block_table_kv_state,
            block_table_score_state,
            block_table,
            kv_cache,
            kv_lens,
            metadata.cu_seq_lens_cuda,
            cu_new_comp_kv,
            metadata.cached_token_lens_cuda,
            metadata.past_kv_lens_cuda[self.compress_ratio][:bsz],
            num_comp_tokens,
            position_ids,
            compressed_mask,
            num_contexts,
            num_generations,
            self.head_dim,
            self.state_dim,
            self.compress_ratio,
            state_tokens_per_block,
            state_tokens_per_block,
            compress_tokens_per_block,
            max_ctx_comp_kv_lens,
            total_num_comp_tokens,
            int(self.kv_cache_dtype),
            self.rotate_activation,
            self.is_indexer,
            self.nope_head_dim,
            self.rope_head_dim,
            next_n,
        )

        if scale_output is not None:
            if self.kv_cache_dtype == KVCacheDtype.MXFP4_BLOCKWISE:
                return primary_output.view(torch.float4_e2m1fn_x2), scale_output
            return primary_output.view(torch.float8_e4m3fn), scale_output
        if primary_output is not None:
            return primary_output, None
        return None, None
