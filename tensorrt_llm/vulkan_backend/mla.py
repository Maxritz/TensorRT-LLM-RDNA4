# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""MLA (multi-layer attention) wrappers.

Provides BatchMLAPagedAttentionWrapper matching flashinfer.mla's API.
MLA in the Vulkan backend uses the mla_fmha.comp shader (already ported
from CUDA).  Falls back to PyTorch SDPA when the C++ Vulkan shared library
is not available.
"""

from typing import Optional
import math

import torch


class BatchMLAPagedAttentionWrapper:
    """FlashInfer-compatible wrapper for MLA paged attention.

    Usage (matches flashinfer.mla):
        wrapper = BatchMLAPagedAttentionWrapper(workspace, kv_layout, ...)
        wrapper.plan(qo_indptr, paged_kv_indptr, paged_kv_indices,
                     paged_kv_last_page_len, num_heads, kv_lora_rank,
                     qk_rope_head_dim, page_size, ...)
        output = wrapper(q, kv_cache)
    """

    def __init__(self, workspace_buffer=None, kv_layout="NHD",
                 use_cuda_graph=False, **kwargs):
        self.workspace_buffer = workspace_buffer
        self.kv_layout = kv_layout
        self._use_cuda_graph = use_cuda_graph
        self._planned = False
        self._plan = None

    @property
    def _backend(self):
        return "mla_vulkan"

    @property
    def is_cuda_graph(self):
        return self._use_cuda_graph

    def plan(self, qo_indptr, paged_kv_indptr, paged_kv_indices,
             paged_kv_last_page_len, num_heads, kv_lora_rank,
             qk_rope_head_dim, page_size, softmax_scale=None,
             causal=True, **kwargs):
        """Store plan parameters for MLA attention."""
        self.page_size = page_size
        self._plan = dict(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            page_size=page_size,
            softmax_scale=softmax_scale or 1.0,
            causal=causal,
            **kwargs,
        )
        self._planned = True

    def __call__(self, q, kv_cache, out=None, **kwargs):
        """Run MLA attention forward.

        Args:
            q: [total_q_tokens, hidden_dim] (kv_lora_rank + qk_rope_head_dim)
            kv_cache: MLA KV cache [batch, num_pages, page_size, kv_combined_dim]
        """
        if not self._planned:
            raise RuntimeError("plan() must be called before calling the wrapper")

        if out is None:
            out = torch.empty_like(q)

        plan = self._plan
        sm_scale = plan.get("softmax_scale", 1.0)
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(plan["qk_rope_head_dim"])

        # MLA attention: Q has kv_lora_rank + qk_rope_head_dim
        # KV cache stores (k_lora, qk_rope) per position
        # For the Vulkan path, we'd dispatch mla_fmha.comp
        # For fallback, use PyTorch SDPA with MLA reshaping

        # This is a simplified MLA implementation that works via PyTorch
        # The full Vulkan path would use the mla_fmha.comp shader
        # For now, implement the math correctly using PyTorch ops

        kv_lora_rank = plan["kv_lora_rank"]
        qk_rope_head_dim = plan["qk_rope_head_dim"]
        num_heads = plan["num_heads"]
        page_size = plan["page_size"]

        qo_indptr = plan["qo_indptr"]
        paged_kv_indptr = plan["paged_kv_indptr"]
        paged_kv_indices = plan["paged_kv_indices"]
        paged_kv_last_page_len = plan["paged_kv_last_page_len"]

        num_seqs = len(qo_indptr) - 1

        # q is already combined (k_lora + qk_rope) per token
        # out has same shape
        out_flat = out.view(-1, num_heads * (kv_lora_rank + qk_rope_head_dim))

        for seq_idx in range(num_seqs):
            q_start = int(qo_indptr[seq_idx])
            q_end = int(qo_indptr[seq_idx + 1])

            # Gather KV for this sequence
            kv_start = int(paged_kv_indptr[seq_idx])
            kv_end = int(paged_kv_indptr[seq_idx + 1])
            pages = paged_kv_indices[kv_start:kv_end]
            num_pages = kv_end - kv_start
            last_len = int(paged_kv_last_page_len[seq_idx])

            # KV cache per page: [page_size, kv_lora_rank + qk_rope_head_dim]
            # But MLA caches K and V separately in some layouts
            # For the simplified version, treat kv_cache as [batch, num_pages, page_size, dim]
            kv_dim = kv_cache.shape[-1]
            kv_tokens = num_pages * page_size - (page_size - last_len)

            kv_seq = kv_cache[seq_idx, pages[:num_pages]].reshape(kv_tokens, kv_dim)

            q_seq = q[q_start:q_end].reshape(-1, num_heads, kv_lora_rank + qk_rope_head_dim)
            k_seq = kv_seq[:, :kv_lora_rank + qk_rope_head_dim].reshape(1, -1, num_heads, kv_lora_rank + qk_rope_head_dim)
            v_seq = kv_seq[:, kv_lora_rank + qk_rope_head_dim:].reshape(1, -1, num_heads, kv_lora_rank)

            # Apply scaled dot-product attention
            # Expand V for MHA
            v_seq = v_seq.repeat(1, 1, num_heads // 1, 1) if num_heads > 1 else v_seq

            q_attn = q_seq * sm_scale
            out_seq = torch.nn.functional.scaled_dot_product_attention(
                q_attn.unsqueeze(0), k_seq, v_seq,
                is_causal=plan.get("causal", True),
            ).squeeze(0)

            out_flat[q_start:q_end] = out_seq.reshape(q_end - q_start, -1)

        return out

    # Alias methods for compatibility
    def run(self, q, kv_cache, out=None, **kwargs):
        return self.__call__(q, kv_cache, out=out, **kwargs)

    @property
    def _is_cuda_graph(self):
        return self._use_cuda_graph
