# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""flashinfer — Vulkan-backed drop-in replacement for flashinfer-ai/flashinfer.

Provides the same public API as the real flashinfer package but dispatches
all compute to Vulkan shaders + numpy instead of CUDA kernels.

Usage:
    import flashinfer
    probs = flashinfer.softmax(logits)
    token = flashinfer.sampling_from_probs(probs)
"""

import sys

# --- Enums (module-level constants matching flashinfer) ---
from .tllm_enums import (
    RoutingMethodType,
    ActivationType,
    SfLayout,
    WeightLayout,
    GatedActType,
    Fp8QuantizationType,
    is_gated_activation,
)

# Flat constants
Default = 0
Renormalize = 1
DeepSeekV3 = 2
Llama4 = 3
RenormalizeNaive = 4
MiniMax2 = 5
SigmoidRenorm = 6
Unspecified = 7
Swiglu = 0
Relu2 = 1
Silu = 2

# --- Submodule imports ---
from . import sampling
from . import activation
from . import rope
from . import norm
from . import page
from . import gemm
from . import topk
from . import prefill
from . import decode
from . import mla
from . import fused_moe
from . import tllm_enums

# --- Sampling re-exports ---
from .sampling import (
    softmax,
    sampling_from_logits,
    sampling_from_probs,
    top_p_sampling_from_probs,
    top_k_sampling_from_probs,
    top_k_top_p_sampling_from_probs,
    top_k_top_p_sampling_from_logits,
    min_p_sampling_from_probs,
    top_p_renorm_probs,
    top_k_renorm_probs,
    top_k_mask_logits,
    chain_speculative_sampling,
)

# --- Page re-exports ---
from .page import (
    append_paged_kv_cache,
    append_paged_mla_kv_cache,
    get_batch_indices_positions,
    get_seq_lens,
)

# --- Norm re-exports ---
from .norm import (
    rmsnorm,
    rmsnorm_quant,
    fused_add_rmsnorm,
    fused_add_rmsnorm_quant,
    gemma_rmsnorm,
    gemma_fused_add_rmsnorm,
    layernorm,
    layernorm_quant,
    fused_rmsnorm_silu,
)

# --- Activation re-exports ---
from .activation import (
    silu_and_mul,
    gelu_and_mul,
    gelu_tanh_and_mul,
)

# --- RoPE re-exports ---
from .rope import (
    apply_rope,
    apply_rope_inplace,
    apply_rope_pos_ids,
    apply_rope_pos_ids_inplace,
    apply_llama31_rope,
    apply_llama31_rope_inplace,
    apply_llama31_rope_pos_ids,
    apply_llama31_rope_pos_ids_inplace,
    apply_rope_with_cos_sin_cache,
    apply_rope_with_cos_sin_cache_inplace,
)

# --- GEMM re-exports ---
from .gemm import (
    bmm_bf16,
    bmm_fp8,
    mm_bf16,
    mm_fp8,
    mm_fp4,
    SegmentGEMMWrapper,
)

# --- Top-K re-exports ---
from .topk import top_k, TopKTieBreak

# --- Prefill re-exports ---
from .prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    single_prefill_with_kv_cache,
    single_prefill_with_kv_cache_return_lse,
)

# --- Decode re-exports ---
from .decode import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchDecodeMlaWithPagedKVCacheWrapper,
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    single_decode_with_kv_cache,
)

# --- MLA re-exports ---
from .mla import BatchMLAPagedAttentionWrapper

# --- MoE re-exports ---
from .fused_moe import B12xMoEWrapper

# --- Version ---
__version__ = "1.0.0-vulkan"


def __getattr__(name):
    """Fallback for any attribute not explicitly defined."""
    raise AttributeError(f"module 'flashinfer' has no attribute '{name}'")
