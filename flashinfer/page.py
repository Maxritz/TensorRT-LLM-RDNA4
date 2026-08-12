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

"""flashinfer.page — Paged KV cache management (Vulkan-backed)."""

import numpy as np
from typing import Optional, Tuple, Union


def append_paged_kv_cache(
    append_key: np.ndarray,
    append_value: np.ndarray,
    batch_indices: np.ndarray,
    positions: np.ndarray,
    paged_kv_cache: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    kv_indices: np.ndarray,
    kv_indptr: np.ndarray,
    kv_last_page_len: np.ndarray,
    kv_layout: str = "NHD",
) -> None:
    """Append key/value tokens to a paged KV cache (in-place).

    Args:
        append_key: (n_tokens, num_kv_heads, head_dim)
        append_value: (n_tokens, num_kv_heads, head_dim)
        batch_indices: (n_tokens,) batch slot per token
        positions: (n_tokens,) position within sequence
        paged_kv_cache: (batch, max_pages, num_kv_heads, page_size, head_dim) or (K, V) tuple
        kv_indices: page index mapping
        kv_indptr: (batch+1,) per-batch page offsets
        kv_last_page_len: (batch,) valid entries in last page
        kv_layout: "NHD" or "HND" (ignored, always NHD)
    """
    if isinstance(paged_kv_cache, tuple):
        kv_k, kv_v = paged_kv_cache
    else:
        kv_k = paged_kv_cache
        kv_v = paged_kv_cache

    n_tokens = append_key.shape[0]
    num_kv_heads = append_key.shape[1]
    head_dim = append_key.shape[2]
    page_size = kv_k.shape[3] if kv_k.ndim == 5 else kv_k.shape[2]

    for t in range(n_tokens):
        b = int(batch_indices[t])
        pos = int(positions[t])
        page_idx = pos // page_size
        page_offset = pos % page_size

        seq_start = int(kv_indptr[b])
        local_page = seq_start + page_idx

        k_val = append_key[t].reshape(num_kv_heads, head_dim)
        v_val = append_value[t].reshape(num_kv_heads, head_dim)

        if kv_k.ndim == 5:
            kv_k[b, local_page, :, page_offset, :] = k_val
            kv_v[b, local_page, :, page_offset, :] = v_val
        else:
            # Flat layout: (total_pages * page_size, num_kv_heads, head_dim)
            base = local_page * page_size + page_offset
            kv_k[base] = k_val
            kv_v[base] = v_val


def append_paged_mla_kv_cache(
    append_ckv: np.ndarray,
    append_kpe: np.ndarray,
    batch_indices: np.ndarray,
    positions: np.ndarray,
    ckv_cache: Optional[np.ndarray] = None,
    kpe_cache: Optional[np.ndarray] = None,
    kv_indices: Optional[np.ndarray] = None,
    kv_indptr: Optional[np.ndarray] = None,
    kv_last_page_len: Optional[np.ndarray] = None,
) -> None:
    """Append MLA key/value tokens to paged cache (Vulkan)."""
    # MLA uses compressed KV — delegate to standard append
    append_paged_kv_cache(
        append_ckv, append_kpe, batch_indices, positions,
        (ckv_cache, kpe_cache) if ckv_cache is not None else append_ckv,
        kv_indices, kv_indptr, kv_last_page_len,
    )


def get_batch_indices_positions(
    append_indptr: np.ndarray,
    seq_lens: np.ndarray,
    nnz: int,
    batch_indices: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand indptr into (batch_indices, positions) arrays."""
    bi = []
    pos = []
    for i in range(len(append_indptr) - 1):
        for j in range(int(seq_lens[i])):
            bi.append(i)
            pos.append(j)
    return (
        np.array(bi, dtype=np.int32),
        np.array(pos, dtype=np.int32),
    )


def get_seq_lens(
    kv_indptr: np.ndarray,
    kv_last_page_len: np.ndarray,
    page_size: int,
) -> np.ndarray:
    """Compute sequence lengths from paged KV cache metadata."""
    seq_lens = np.empty(len(kv_indptr) - 1, dtype=np.int64)
    for i in range(len(seq_lens)):
        seq_lens[i] = (int(kv_indptr[i + 1]) - int(kv_indptr[i]) - 1) * page_size + int(kv_last_page_len[i])
    return seq_lens
