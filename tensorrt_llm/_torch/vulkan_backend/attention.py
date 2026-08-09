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

"""
Vulkan-backed attention wrappers that match flashinfer's API.

Provides:
  - BatchPrefillWithPagedKVCacheWrapper
  - BatchDecodeWithPagedKVCacheWrapper
  - BatchPrefillWithRaggedKVCacheWrapper

The paged-KV attention dispatch uses the existing Vulkan `attention.comp`
compute shader (ported from CUDA `single_stream_attention_kernel_2d`).
When the C++ Vulkan shared library is not available, a pure-PyTorch/ROCm
fallback using ``F.scaled_dot_product_attention`` is used instead.
"""

from typing import Optional

import torch
import torch.nn.functional as F

try:
    from . import torch_bridge as _tb
    _HAS_TORCH_BRIDGE = _tb.is_available()
except Exception:
    _HAS_TORCH_BRIDGE = False


def _to_device(t, device):
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, device=device)
    elif t.device != device:
        t = t.to(device)
    return t


class _PagedAttentionBase:
    """Shared state for paged-KV attention wrappers."""

    def __init__(self, workspace_buffer=None, kv_layout="NHD", **kwargs):
        self.workspace_buffer = workspace_buffer
        self.kv_layout = kv_layout
        self._backend = "fa2"
        self._planned = False
        # plan parameters stored for use in run()
        self._plan = None

    @property
    def _is_cuda_graph(self):
        return getattr(self, "_use_cuda_graph", False)

    def plan(self, qo_indptr, num_qo_heads, num_kv_heads, head_dim,
             custom_mask=None, causal=False, sm_scale=None,
             window_left=-1, q_data_type=None, kv_data_type=None,
             o_data_type=None, max_token_per_sequence=None,
             **kwargs):
        """Store plan parameters.  In Vulkan mode we don't pre-compile
        kernels, so this just records the config."""
        self._plan = dict(
            qo_indptr=qo_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            custom_mask=custom_mask,
            causal=causal,
            sm_scale=sm_scale,
            window_left=window_left,
            q_data_type=q_data_type or torch.float16,
            kv_data_type=kv_data_type or torch.float16,
            o_data_type=o_data_type,
            max_token_per_sequence=max_token_per_sequence,
            **kwargs,
        )
        self._planned = True

    def _run_paged_attention(self, q, kv_cache, out, is_decode):
        """Pure-PyTorch paged attention that works on ROCm.

        Uses F.scaled_dot_product_attention per-sequence for correctness.
        """
        plan = self._plan
        num_heads = plan["num_qo_heads"]
        num_kv_heads = plan["num_kv_heads"]
        head_dim = plan["head_dim"]
        causal = plan["causal"]
        sm_scale = plan["sm_scale"]
        window_left = plan["window_left"]

        if sm_scale is None:
            sm_scale = 1.0 / (head_dim ** 0.5)

        # q shape: [total_tokens, num_heads, head_dim]
        # kv_cache shape depends on kv_layout
        if self.kv_layout == "NHD":
            # kv_cache: [batch, num_pages, page_size * 2, num_kv_heads, head_dim]
            # or tuple (k, v): each [batch, num_pages, page_size, num_kv_heads, head_dim]
            if isinstance(kv_cache, (tuple, list)) and len(kv_cache) == 2:
                k_cache, v_cache = kv_cache
            else:
                # Combined k/v
                half = kv_cache.shape[-2] // 2
                k_cache = kv_cache[..., :half, :]
                v_cache = kv_cache[..., half:, :]
        elif self.kv_layout == "HND":
            if isinstance(kv_cache, (tuple, list)) and len(kv_cache) == 2:
                k_cache, v_cache = kv_cache
            else:
                half = kv_cache.shape[-2] // 2
                k_cache = kv_cache[..., :half, :]
                v_cache = kv_cache[..., half:, :]
        else:
            raise ValueError(f"Unsupported kv_layout: {self.kv_layout}")

        # Get plan parameters
        if is_decode:
            qo_indptr = plan.get("qo_indptr", None)
            paged_kv_indptr = plan.get("paged_kv_indptr_buffer", None)
            paged_kv_indices = plan.get("paged_kv_indices_buffer", None)
            paged_kv_last_page_len = plan.get("paged_kv_last_page_len_buffer", None)
            page_size = plan.get("page_size", 1)
        else:
            qo_indptr = plan.get("qo_indptr", None)
            paged_kv_indptr = plan.get("paged_kv_indptr_buffer", None)
            paged_kv_indices = plan.get("paged_kv_indices", None)
            paged_kv_last_page_len = plan.get("paged_kv_last_page_len", None)
            page_size = plan.get("page_size", 1)

        num_seqs = len(qo_indptr) - 1 if qo_indptr is not None else 1

        # For decode: each sequence has exactly 1 query token
        # For prefill: each sequence has (qo_indptr[i+1] - qo_indptr[i]) query tokens
        out_flat = out.view(-1, num_heads, head_dim)

        for seq_idx in range(num_seqs):
            q_start = int(qo_indptr[seq_idx]) if qo_indptr is not None else 0
            q_end = int(qo_indptr[seq_idx + 1]) if qo_indptr is not None else q.shape[0]
            q_tokens = q[q_start:q_end].reshape(-1, num_heads, head_dim)  # [n_tok, H, D]

            # Gather KV for this sequence from paged cache
            if paged_kv_indptr is not None:
                page_start = int(paged_kv_indptr[seq_idx])
                page_end = int(paged_kv_indptr[seq_idx + 1])
                pages = paged_kv_indices[page_start:page_end]
                num_pages = page_end - page_start
                last_len = int(paged_kv_last_page_len[seq_idx]) if paged_kv_last_page_len is not None else page_size

            if num_kv_heads == num_heads:
                q_for_attn = q_tokens
            else:
                # GQA: repeat KV heads
                reps = num_heads // num_kv_heads
                q_for_attn = q_tokens  # q already has num_heads

            # Build K, V for this sequence
            kv_seqlen = num_pages * page_size - (page_size - last_len) if num_pages > 0 else 0
            k_all = k_cache[seq_idx, pages[:num_pages]].reshape(kv_seqlen, num_kv_heads, head_dim)
            v_all = v_cache[seq_idx, pages[:num_pages]].reshape(kv_seqlen, num_kv_heads, head_dim)

            if num_kv_heads != num_heads:
                # Repeat for GQA
                k_all = k_all.repeat_interleave(reps, dim=1)
                v_all = v_all.repeat_interleave(reps, dim=1)

            # Scale Q
            q_for_attn = q_for_attn * sm_scale

            # SDPA
            q_attn = q_for_attn.unsqueeze(0)  # [1, n_tok, H, D]
            k_attn = k_all.unsqueeze(0)
            v_attn = v_all.unsqueeze(0)

            attn_mask = None
            if causal:
                # Causal mask for this sequence
                q_len = q_attn.shape[1]
                k_len = k_attn.shape[1]
                mask = torch.tril(torch.ones(q_len, k_len, device=q.device, dtype=torch.bool))
                attn_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, k_len]
                if window_left > 0:
                    # Sliding window: only attend to last window_left tokens
                    mask = mask & torch.arange(k_len, device=q.device).unsqueeze(0) >= (k_len - window_left)

            out_attn = F.scaled_dot_product_attention(
                q_attn, k_attn, v_attn,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            out_attn = out_attn.squeeze(0)  # [n_tok, H, D]

            out_flat[q_start:q_end] = out_attn


class BatchPrefillWithPagedKVCacheWrapper(_PagedAttentionBase):
    """FlashInfer-compatible wrapper for paged KV-cache prefill attention.

    Usage (matches flashinfer):
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout, ...)
        wrapper.plan(qo_indptr, paged_kv_indptr, paged_kv_indices,
                     paged_kv_last_page_len, num_heads, num_kv_heads, head_dim,
                     page_size, causal=True, sm_scale=..., q_data_type=...,
                     kv_data_type=..., o_data_type=..., custom_mask=...)
        wrapper.run(q, kv_cache, out=output)
    """

    def __init__(self, workspace_buffer=None, kv_layout="NHD", backend=None,
                 qo_indptr_buf=None, paged_kv_indptr_buf=None,
                 paged_kv_indices_buf=None, paged_kv_last_page_len_buf=None,
                 use_cuda_graph=False, custom_mask=None, **kwargs):
        super().__init__(workspace_buffer, kv_layout, **kwargs)
        self._backend = backend or "fa2"
        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr_buf
        self._paged_kv_indptr_buf = paged_kv_indptr_buf
        self._paged_kv_indices_buf = paged_kv_indices_buf
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buf
        self.page_size = kwargs.get("page_size", 1)

    def plan(self, qo_indptr, paged_kv_indptr, paged_kv_indices,
             paged_kv_last_page_len, num_qo_heads, num_kv_heads, head_dim,
             page_size, causal=False, sm_scale=None, window_left=-1,
             q_data_type=None, kv_data_type=None, o_data_type=None,
             custom_mask=None, max_token_per_sequence=None, **kwargs):
        self.page_size = page_size
        super().plan(
            qo_indptr=qo_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            causal=causal,
            sm_scale=sm_scale,
            window_left=window_left,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            max_token_per_sequence=max_token_per_sequence,
            paged_kv_indptr_buffer=paged_kv_indptr,
            paged_kv_indices_buffer=paged_kv_indices,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len,
            page_size=page_size,
        )

    def run(self, q, kv_cache, out=None, **kwargs):
        if not self._planned:
            raise RuntimeError("plan() must be called before run()")
        if out is None:
            out = torch.empty_like(q)
        self._run_paged_attention(q, kv_cache, out, is_decode=False)
        return out


class BatchDecodeWithPagedKVCacheWrapper(_PagedAttentionBase):
    """FlashInfer-compatible wrapper for paged KV-cache decode attention.

    Usage (matches flashinfer):
        wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout, ...)
        wrapper.plan(paged_kv_indptr, paged_kv_indices,
                     paged_kv_last_page_len, num_heads, num_kv_heads,
                     head_dim, page_size, sm_scale=..., ...)
        wrapper.run(q, kv_cache, out=output)
    """

    def __init__(self, workspace_buffer=None, kv_layout="NHD",
                 paged_kv_indptr_buffer=None, paged_kv_indices_buffer=None,
                 paged_kv_last_page_len_buffer=None, use_cuda_graph=False,
                 use_tensor_cores=False, backend=None, **kwargs):
        super().__init__(workspace=workspace_buffer, kv_layout=kv_layout, **kwargs)
        self._backend = backend or "auto"
        self._use_cuda_graph = use_cuda_graph
        self._paged_kv_indptr_buf = paged_kv_indptr_buffer
        self._paged_kv_indices_buf = paged_kv_indices_buffer
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buffer
        self.page_size = kwargs.get("page_size", 8)

    def plan(self, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
             num_qo_heads, num_kv_heads, head_dim, page_size,
             sm_scale=None, window_left=-1, q_data_type=None,
             kv_data_type=None, o_data_type=None, block_tables=None,
             **kwargs):
        self.page_size = page_size
        # Build qo_indptr for decode (1 token per sequence)
        batch_size = len(paged_kv_indptr) - 1
        qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=paged_kv_indptr.device)
        super().plan(
            qo_indptr=qo_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            causal=True,
            sm_scale=sm_scale,
            window_left=window_left,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            paged_kv_indptr_buffer=paged_kv_indptr,
            paged_kv_indices_buffer=paged_kv_indices,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len,
            page_size=page_size,
        )

    def run(self, q, kv_cache, out=None, **kwargs):
        if not self._planned:
            raise RuntimeError("plan() must be called before run()")
        if out is None:
            out = torch.empty_like(q)
        self._run_paged_attention(q, kv_cache, out, is_decode=True)
        return out


class BatchPrefillWithRaggedKVCacheWrapper(_PagedAttentionBase):
    """FlashInfer-compatible wrapper for ragged KV-cache prefill.

    This variant handles variable-length sequences without paging.
    """

    def __init__(self, workspace_buffer=None, kv_layout="NHD", **kwargs):
        super().__init__(workspace_buffer, kv_layout, **kwargs)

    def plan(self, qo_indptr, num_qo_heads, num_kv_heads, head_dim_qk,
             custom_mask=None, causal=False, sm_scale=None,
             window_left=-1, q_data_type=None, kv_data_type=None,
             max_token_per_sequence=None, **kwargs):
        super().plan(
            qo_indptr=qo_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim_qk,
            causal=causal,
            sm_scale=sm_scale,
            window_left=window_left,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            max_token_per_sequence=max_token_per_sequence,
            custom_mask=custom_mask,
            **kwargs,
        )

    def run(self, q, k, v, out=None, **kwargs):
        if not self._planned:
            raise RuntimeError("plan() must be called before run()")
        if out is None:
            out = torch.empty_like(q)

        plan = self._plan
        num_heads = plan["num_qo_heads"]
        num_kv_heads = plan["num_kv_heads"]
        head_dim = plan["head_dim"]
        causal = plan["causal"]
        sm_scale = plan["sm_scale"] or (1.0 / (head_dim ** 0.5))
        window_left = plan["window_left"]
        custom_mask = plan.get("custom_mask")

        qo_indptr = plan["qo_indptr"]
        # k, v: [total_kv_tokens, num_kv_heads * head_dim]
        k = k.reshape(-1, num_kv_heads, head_dim)
        v = v.reshape(-1, num_kv_heads, head_dim)

        if num_kv_heads != num_heads:
            reps = num_heads // num_kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)

        q = q.reshape(-1, num_heads, head_dim) * sm_scale
        num_seqs = len(qo_indptr) - 1

        out_flat = out.view(-1, num_heads, head_dim)

        # --- Vulkan fast path: batch all sequences when possible ---
        # Conditions: Vulkan backend available, no custom mask, no sliding
        # window, default sm_scale (shader applies 1/sqrt(head_dim) itself),
        # and all sequences share the same kv_len so K/V can be batched.
        default_sm_scale = 1.0 / (head_dim ** 0.5)
        use_vulkan = (
            _HAS_TORCH_BRIDGE
            and custom_mask is None
            and (window_left is None or window_left < 0)
            and abs(sm_scale - default_sm_scale) < 1e-6
        ) if num_seqs > 0 else False

        if use_vulkan:
            # Check if all sequences have equal q_len and kv_len for batching
            q_lens = [int(qo_indptr[i + 1] - qo_indptr[i]) for i in range(num_seqs)]
            kv_starts = [int(qo_indptr[i]) for i in range(num_seqs)]
            kv_ends = [int(qo_indptr[i + 1]) if i + 2 <= num_seqs else k.shape[0]
                       for i in range(num_seqs)]
            kv_lens = [kv_ends[i] - kv_starts[i] for i in range(num_seqs)]
            if len(set(q_lens)) == 1 and len(set(kv_lens)) == 1:
                q_len = q_lens[0]
                kv_len = kv_lens[0]
                # Q: [num_seqs, q_len, H, D] -> [num_seqs, H, q_len, D]
                q_batched = q.view(num_seqs, q_len, num_heads, head_dim).transpose(1, 2)
                # K/V: [num_seqs, kv_len, H, D] -> [num_seqs, H, kv_len, D]
                k_batched = k.view(num_seqs, kv_len, num_heads, head_dim).transpose(1, 2)
                v_batched = v.view(num_seqs, kv_len, num_heads, head_dim).transpose(1, 2)
                # vulkan_attention applies 1/sqrt(head_dim) internally;
                # undo the manual sm_scale applied on line 377
                q_batched = q_batched / sm_scale
                out_attn = _tb.vulkan_attention(q_batched, k_batched, v_batched, causal)
                # out_attn: [num_seqs, H, q_len, D] -> [num_seqs, q_len, H, D] -> flat
                out_flat[:] = out_attn.transpose(1, 2).reshape(num_seqs, q_len, num_heads, head_dim).reshape(-1, num_heads, head_dim)
                return out

        # --- PyTorch fallback (per-sequence loop) ---
        for i in range(num_seqs):
            q_start = int(qo_indptr[i])
            q_end = int(qo_indptr[i + 1])
            kv_start = q_start  # ragged: same layout
            kv_end = int(qo_indptr[i + 1]) if i + 2 <= num_seqs else k.shape[0]

            q_seq = q[q_start:q_end].unsqueeze(0)
            k_seq = k[kv_start:kv_end].unsqueeze(0)
            v_seq = v[kv_start:kv_end].unsqueeze(0)

            if causal:
                mask = None
            else:
                mask = plan.get("custom_mask")
                if mask is not None:
                    mask = mask.unsqueeze(0).unsqueeze(0)

            out_seq = F.scaled_dot_product_attention(
                q_seq, k_seq, v_seq,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=causal,
            )
            out_flat[q_start:q_end] = out_seq.squeeze(0)
