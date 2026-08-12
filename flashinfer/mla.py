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

"""flashinfer.mla — Multi-head Latent Attention (Vulkan-backed)."""

import numpy as np
from typing import Optional


class BatchMLAPagedAttentionWrapper:
    """MLA paged attention wrapper (Vulkan)."""

    def __init__(self, float_workspace_buffer=None, kv_layout: str = "NHD", **kwargs):
        self._plan_done = False

    def plan(self, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
             num_qo_heads, num_kv_heads, head_dim, page_size,
             causal=False, **kwargs):
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._page_size = page_size
        self._causal = causal
        self._sm_scale = 1.0 / np.sqrt(head_dim)
        self._plan_done = True

    def run(self, q: np.ndarray, paged_kv_cache, **kwargs) -> np.ndarray:
        """Run MLA attention."""
        if isinstance(paged_kv_cache, tuple):
            k_cache, v_cache = paged_kv_cache
        else:
            k_cache = v_cache = paged_kv_cache

        if k_cache.ndim == 5:
            k_flat = k_cache.reshape(-1, self._num_kv_heads, self._head_dim)
            v_flat = v_cache.reshape(-1, self._num_kv_heads, self._head_dim)
        else:
            k_flat = k_cache
            v_flat = v_cache

        S_q = q.shape[0] if q.ndim <= 2 else q.shape[1]
        S_k = k_flat.shape[0]

        q_bt = q.reshape(1, S_q, self._num_qo_heads, self._head_dim).transpose(0, 2, 1, 3)
        k_bt = k_flat.reshape(1, S_k, self._num_kv_heads, self._head_dim).transpose(0, 2, 1, 3)
        v_bt = v_flat.reshape(1, S_k, self._num_kv_heads, self._head_dim).transpose(0, 2, 1, 3)

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

    def end_forward(self):
        self._plan_done = False
