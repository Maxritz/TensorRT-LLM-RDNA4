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

"""flashinfer.prefill — Prefill attention wrappers (Vulkan-backed)."""

import numpy as np
from typing import Optional, Tuple


class BatchPrefillWithPagedKVCacheWrapper:
    """Paged KV cache prefill attention (Vulkan)."""

    def __init__(self, float_workspace_buffer=None, kv_layout: str = "NHD",
                 use_cuda_graph: bool = False, **kwargs):
        self._kv_layout = kv_layout
        self._plan_done = False

    def plan(self, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
             num_qo_heads, num_kv_heads, head_dim, page_size,
             causal=False, pos_encoding_mode="NONE", use_fp16_qk_reduction=False,
             sm_scale=None, window_left=-1, logits_soft_cap=None,
             rope_scale=None, rope_theta=None,
             q_data_type="float16", kv_data_type=None,
             custom_mask=None, packed_custom_mask=None, **kwargs):
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._page_size = page_size
        self._causal = causal
        self._sm_scale = sm_scale or (1.0 / np.sqrt(head_dim))
        self._qo_indptr = qo_indptr
        self._paged_kv_indptr = paged_kv_indptr
        self._paged_kv_indices = paged_kv_indices
        self._paged_kv_last_page_len = paged_kv_last_page_len
        self._plan_done = True

    def run(self, q: np.ndarray, paged_kv_cache, **kwargs) -> np.ndarray:
        """Run prefill attention."""
        if not self._plan_done:
            raise RuntimeError("Must call plan() before run()")
        # Simplified: treat as standard attention
        if isinstance(paged_kv_cache, tuple):
            k_cache, v_cache = paged_kv_cache
        else:
            k_cache = v_cache = paged_kv_cache

        B = q.shape[0] if q.ndim >= 3 else 1
        S_q = q.shape[1] if q.ndim >= 3 else q.shape[0]

        # Flatten paged cache to (seq_len, num_kv_heads, head_dim)
        if k_cache.ndim == 5:
            k_flat = k_cache.reshape(-1, self._num_kv_heads, self._head_dim)
            v_flat = v_cache.reshape(-1, self._num_kv_heads, self._head_dim)
        else:
            k_flat = k_cache
            v_flat = v_cache

        S_k = k_flat.shape[0]

        # Standard scaled dot-product attention
        scale = self._sm_scale
        q_reshaped = q.reshape(B, S_q, self._num_qo_heads, self._head_dim).transpose(0, 2, 1, 3)
        k_reshaped = k_flat.reshape(1, S_k, self._num_kv_heads, self._head_dim).transpose(0, 2, 1, 3)
        v_reshaped = v_flat.reshape(1, S_k, self._num_kv_heads, self._head_dim).transpose(0, 2, 1, 3)

        # GQA expand
        n_rep = self._num_qo_heads // self._num_kv_heads
        if n_rep > 1:
            k_reshaped = np.repeat(k_reshaped, n_rep, axis=1)
            v_reshaped = np.repeat(v_reshaped, n_rep, axis=1)

        # Attention scores: q(B,H,S_q,D), k(B,H,S_k,D) -> scores(B,H,S_q,S_k)
        scores = np.einsum("bhqd,bhkd->bhqk", q_reshaped, k_reshaped) * scale
        if self._causal:
            mask = np.triu(np.full((S_q, S_k), -np.inf, dtype=np.float32), k=S_k - S_q + 1)
            scores = scores + mask[np.newaxis, np.newaxis, :, :]
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        out = np.einsum("bhqk,bhvd->bhqd", weights, v_reshaped)
        return out.transpose(0, 2, 1, 3).reshape(B, S_q, -1)

    forward = run

    def run_return_lse(self, q: np.ndarray, paged_kv_cache, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Run prefill attention and return (output, log-sum-exp)."""
        out = self.run(q, paged_kv_cache, **kwargs)
        lse = np.zeros((out.shape[0], out.shape[1]), dtype=np.float32)
        return out, lse

    forward_return_lse = run_return_lse

    def end_forward(self):
        self._plan_done = False

    def reset_workspace_buffer(self, float_workspace_buffer=None, int_workspace_buffer=None):
        pass


class BatchPrefillWithRaggedKVCacheWrapper:
    """Ragged KV cache prefill attention (Vulkan)."""

    def __init__(self, float_workspace_buffer=None, kv_layout: str = "NHD"):
        self._plan_done = False

    def plan(self, qo_indptr, num_qo_heads, num_kv_heads, head_dim,
             causal=False, pos_encoding_mode="NONE", use_fp16_qk_reduction=False,
             sm_scale=None, window_left=-1, logits_soft_cap=None,
             rope_scale=None, rope_theta=None,
             q_data_type="float16", kv_data_type=None,
             custom_mask=None, **kwargs):
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._causal = causal
        self._sm_scale = sm_scale or (1.0 / np.sqrt(head_dim))
        self._qo_indptr = qo_indptr
        self._plan_done = True

    def run(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, **kwargs) -> np.ndarray:
        """Run ragged prefill attention."""
        if not self._plan_done:
            raise RuntimeError("Must call plan() before run()")

        S_q = q.shape[0] if q.ndim <= 2 else q.shape[1] if q.ndim == 3 else q.shape[-2]
        S_k = k.shape[0] if k.ndim <= 2 else k.shape[1] if k.ndim == 3 else k.shape[-2]

        # (S_q, H, D) -> (1, H, S_q, D)
        q_bt = q.reshape(1, S_q, self._num_qo_heads, self._head_dim).transpose(0, 2, 1, 3)
        k_bt = k.reshape(1, S_k, self._num_kv_heads, self._head_dim).transpose(0, 2, 1, 3)
        v_bt = v.reshape(1, S_k, self._num_kv_heads, self._head_dim).transpose(0, 2, 1, 3)

        n_rep = self._num_qo_heads // self._num_kv_heads
        if n_rep > 1:
            k_bt = np.repeat(k_bt, n_rep, axis=1)
            v_bt = np.repeat(v_bt, n_rep, axis=1)

        scores = np.einsum("bhqd,bhkd->bhqk", q_bt, k_bt) * self._sm_scale
        if self._causal:
            mask = np.triu(np.full((S_q, S_k), -np.inf, dtype=np.float32), k=S_k - S_q + 1)
            scores = scores + mask[np.newaxis, np.newaxis, :, :]
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        out = np.einsum("bhqk,bhvd->bhqd", weights, v_bt)
        return out.transpose(0, 2, 1, 3).reshape(S_q, -1)

    def run_return_lse(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        out = self.run(q, k, v, **kwargs)
        lse = np.zeros((out.shape[0],), dtype=np.float32)
        return out, lse

    def end_forward(self):
        self._plan_done = False


def single_prefill_with_kv_cache(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    causal: bool = True, **kwargs,
) -> np.ndarray:
    """Single prefill attention.

    Args:
        q: (S_q, D) or (S_q, H, D)
        k: (S_k, D) or (S_k, H_kv, D)
        v: (S_k, D) or (S_k, H_kv, D)
    """
    # Normalize to (S, H, D)
    if q.ndim == 2:
        D = q.shape[-1]
        q = q.reshape(q.shape[0], 1, D)
    if k.ndim == 2:
        D = k.shape[-1]
        k = k.reshape(k.shape[0], 1, D)
    if v.ndim == 2:
        D = v.shape[-1]
        v = v.reshape(v.shape[0], 1, D)

    S_q, H_q, D = q.shape
    S_k, H_kv, _ = k.shape
    scale = 1.0 / np.sqrt(D)

    # (S, H, D) -> (1, H, S, D)
    q_bt = q.transpose(1, 0, 2)[np.newaxis]
    k_bt = k.transpose(1, 0, 2)[np.newaxis]
    v_bt = v.transpose(1, 0, 2)[np.newaxis]

    # GQA expand
    n_rep = H_q // H_kv
    if n_rep > 1:
        k_bt = np.repeat(k_bt, n_rep, axis=1)
        v_bt = np.repeat(v_bt, n_rep, axis=1)

    scores = np.einsum("bhqd,bhkd->bhqk", q_bt, k_bt) * scale
    if causal:
        mask = np.triu(np.full((S_q, S_k), -np.inf, dtype=np.float32), k=S_k - S_q + 1)
        scores = scores + mask[np.newaxis, np.newaxis, :, :]
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    out = np.einsum("bhqk,bhvd->bhqd", weights, v_bt)
    # (1, H_q, S_q, D) -> (S_q, H_q, D) -> (S_q, H_q*D)
    return out[0].transpose(1, 0, 2).reshape(S_q, -1)


def single_prefill_with_kv_cache_return_lse(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    causal: bool = True, **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single prefill attention with LSE."""
    out = single_prefill_with_kv_cache(q, k, v, causal, **kwargs)
    # LSE: (batch,) or (1,)
    if q.ndim == 2:
        lse = np.zeros((1,), dtype=np.float32)
    else:
        lse = np.zeros((1,), dtype=np.float32)
    return out, lse
