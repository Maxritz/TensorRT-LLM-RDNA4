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

"""flashinfer.decode — Decode attention wrappers (Vulkan-backed)."""

import numpy as np
from typing import Optional, Tuple


class BatchDecodeWithPagedKVCacheWrapper:
    """Paged KV cache decode attention (Vulkan)."""

    def __init__(self, float_workspace_buffer=None, kv_layout: str = "NHD",
                 use_tensor_cores: bool = False, **kwargs):
        self._kv_layout = kv_layout
        self._forward_done = False

    def begin_forward(self, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
                      num_qo_heads, num_kv_heads, head_dim, page_size,
                      pos_encoding_mode="NONE", data_type="float16",
                      sm_scale=None, rope_scale=None, rope_theta=None,
                      window_left=-1, logits_soft_cap=None, **kwargs):
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._page_size = page_size
        self._sm_scale = sm_scale or (1.0 / np.sqrt(head_dim))
        self._paged_kv_indptr = paged_kv_indptr
        self._paged_kv_indices = paged_kv_indices
        self._paged_kv_last_page_len = paged_kv_last_page_len
        self._forward_done = True

    def run(self, q: np.ndarray, paged_kv_cache, **kwargs) -> np.ndarray:
        """Run decode attention."""
        if not self._forward_done:
            raise RuntimeError("Must call begin_forward() before run()")

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

        S_k = k_flat.shape[0]
        scale = self._sm_scale

        # q: (num_qo_heads, head_dim) or (batch, num_qo_heads, head_dim)
        if q.ndim == 2:
            q_bt = q.transpose(1, 0)[np.newaxis]  # (1, H, D) -> (1, H, 1, D)
        else:
            B = q.shape[0]
            q_bt = q.reshape(B, self._num_qo_heads, self._head_dim).transpose(0, 2, 1)[:, :, np.newaxis, :]

        k_bt = k_flat.T.reshape(self._num_kv_heads, self._head_dim, -1).transpose(0, 2, 1)[np.newaxis]  # (1, H, S, D)
        v_bt = v_flat.T.reshape(self._num_kv_heads, self._head_dim, -1).transpose(0, 2, 1)[np.newaxis]

        n_rep = self._num_qo_heads // self._num_kv_heads
        if n_rep > 1:
            k_bt = np.repeat(k_bt, n_rep, axis=1)
            v_bt = np.repeat(v_bt, n_rep, axis=1)

        scores = np.einsum("bhqd,bhkd->bhqk", q_bt, k_bt) * scale
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        out = np.einsum("bhqk,bhkd->bhqd", weights, v_bt)

        if q.ndim == 2:
            return out[0, :, 0, :]  # (num_qo_heads, head_dim)
        return out.squeeze(2)  # (batch, num_qo_heads, head_dim)

    forward = run

    def run_return_lse(self, q: np.ndarray, paged_kv_cache, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        out = self.run(q, paged_kv_cache, **kwargs)
        lse = np.zeros(out.shape[:-1], dtype=np.float32)
        return out, lse

    forward_return_lse = run_return_lse

    def end_forward(self):
        self._forward_done = False

    def reset_workspace_buffer(self, float_workspace_buffer=None, int_workspace_buffer=None):
        pass


class BatchDecodeMlaWithPagedKVCacheWrapper(BatchDecodeWithPagedKVCacheWrapper):
    """MLA paged KV cache decode attention (Vulkan)."""
    pass


class CUDAGraphBatchDecodeWithPagedKVCacheWrapper(BatchDecodeWithPagedKVCacheWrapper):
    """CUDA-graph compatible decode (Vulkan, no-op wrapper)."""
    pass


def single_decode_with_kv_cache(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    sm_scale: Optional[float] = None,
    **kwargs,
) -> np.ndarray:
    """Single decode attention.

    Args:
        q: (D,) or (H, D)
        k: (S_k, D) or (S_k, H_kv, D)
        v: (S_k, D) or (S_k, H_kv, D)
    Returns:
        (H, D) output
    """
    # Normalize to (H, D) for q and (S, H, D) for k/v
    if q.ndim == 1:
        D = q.shape[-1]
        q = q.reshape(1, D)  # (1, D)
    else:
        D = q.shape[-1]

    if k.ndim == 2:
        k = k.reshape(k.shape[0], 1, D)  # (S, 1, D)
    if v.ndim == 2:
        v = v.reshape(v.shape[0], 1, D)

    H_q, _ = q.shape
    S_k, H_kv, D = k.shape
    scale = sm_scale or (1.0 / np.sqrt(D))

    # (H, D) -> (1, H, 1, D)
    q_bt = q.reshape(1, H_q, 1, D)
    # (S, H, D) -> (1, H, S, D)
    k_bt = k.transpose(1, 0, 2)[np.newaxis]
    v_bt = v.transpose(1, 0, 2)[np.newaxis]

    # GQA expand
    n_rep = H_q // H_kv
    if n_rep > 1:
        k_bt = np.repeat(k_bt, n_rep, axis=1)
        v_bt = np.repeat(v_bt, n_rep, axis=1)

    scores = np.einsum("bhqd,bhkd->bhqk", q_bt, k_bt) * scale
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    out = np.einsum("bhqk,bhkd->bhqd", weights, v_bt)
    return out[0, :, 0, :]  # (H, D)
