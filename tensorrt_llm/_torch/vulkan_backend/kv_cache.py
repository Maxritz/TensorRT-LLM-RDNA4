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
Vulkan-backed paged KV cache management.

Provides `append_paged_kv_cache` and `append_paged_mla_kv_cache` matching
the flashinfer.page API, plus `get_batch_indices_positions` and
`get_seq_lens` utility functions.
"""

from typing import Optional, Tuple

import torch

# Import the C API bridge
try:
    from . import vulkan_compute as _vk
    _HAS_VULKAN = _vk.is_available()
except Exception:
    _HAS_VULKAN = False


def append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_kv_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    kv_cache_manager=None,  # optional, for compatibility
) -> None:
    """Append key/value tokens to a paged KV cache.

    Matches ``flashinfer.page.append_paged_kv_cache``.
    """
    if _HAS_VULKAN:
        # Use Vulkan compute shader for the scatter operation
        n = append_key.shape[0]
        elem_size = append_key.element_size()
        ok = _vk.tllm_vulkan_append_paged_kv_cache(
            _ptr_of(append_key), _ptr_of(append_value),
            _ptr_of(batch_indices), _ptr_of(positions),
            _ptr_of(paged_kv_cache), _ptr_of(kv_indices),
            _ptr_of(kv_indptr), _ptr_of(kv_last_page_len),
            ctypes.c_uint32(page_size),
            ctypes.c_uint32(num_kv_heads),
            ctypes.c_uint32(head_dim),
            ctypes.c_uint32(elem_size),
            ctypes.c_uint32(n),
        )
        _check(ok, "append_paged_kv_cache")
    else:
        # Fallback: pure PyTorch scatter implementation
        _append_paged_kv_cache_pytorch(
            append_key, append_value, batch_indices, positions,
            paged_kv_cache, kv_indices, kv_indptr, kv_last_page_len,
            page_size, num_kv_heads, head_dim)

    _sync()


def append_paged_mla_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    *args,
    **kwargs,
) -> None:
    """Append MLA key/value tokens to a paged KV cache.

    Matches ``flashinfer.page.append_paged_mla_kv_cache``.
    Currently delegates to the standard paged KV cache append since the
    MLA cache has a similar layout.
    """
    append_paged_kv_cache(append_key, append_value, *args, **kwargs)


def get_seq_lens(
    paged_kv_indptr: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Compute sequence lengths from paged KV cache metadata.

    Matches ``flashinfer.get_seq_lens``.
    """
    indptr = paged_kv_indptr.cpu()
    last_page = paged_kv_last_page_len.cpu()
    seq_lens = torch.empty(paged_kv_indptr.numel() - 1, dtype=torch.long)
    for i in range(seq_lens.numel()):
        seq_lens[i] = (indptr[i + 1] - indptr[i] - 1) * page_size + last_page[i]
    return seq_lens


def get_batch_indices_positions(
    paged_kv_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand (indptr, last_page_len) into (batch_indices, positions).

    Matches ``flashinfer.get_batch_indices_positions``.
    """
    indptr = paged_kv_indptr.cpu()
    batch_indices = []
    positions = []
    for i in range(indptr.numel() - 1):
        for j in range(seq_lens[i].item()):
            batch_indices.append(i)
            positions.append(j)
    return (
        torch.tensor(batch_indices, dtype=torch.int32, device=paged_kv_indptr.device),
        torch.tensor(positions, dtype=torch.int32, device=paged_kv_indptr.device),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

import ctypes


def _ptr_of(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _check(ret: int, name: str):
    if ret == 0:
        raise RuntimeError(f"Vulkan compute [{name}] failed")


def _append_paged_kv_cache_pytorch(
    append_key, append_value, batch_indices, positions,
    paged_kv_cache, kv_indices, kv_indptr, kv_last_page_len,
    page_size, num_kv_heads, head_dim,
):
    """Pure PyTorch fallback for paged KV cache append."""
    if isinstance(paged_kv_cache, tuple):
        kv = paged_kv_cache
    else:
        chunks = torch.chunk(paged_kv_cache, 2, dim=-1)
        kv = (chunks[0], chunks[1])

    paged_kv_indptr_cpu = kv_indptr.cpu()
    kv_last_page_len_cpu = kv_last_page_len.cpu()

    for t in range(append_key.shape[0]):
        batch_idx = batch_indices[t].item()
        pos = positions[t].item()

        seq_start = paged_kv_indptr_cpu[batch_idx].item()
        seq_end = paged_kv_indptr_cpu[batch_idx + 1].item()
        num_pages = seq_end - seq_start
        page_idx = pos // page_size
        page_offset = pos % page_size

        if page_idx == num_pages - 1:
            target_len = kv_last_page_len_cpu[batch_idx].item()
        else:
            target_len = page_size

        if page_offset < target_len:
            page_loc = seq_start + page_idx
            k_page = kv[0][batch_idx, page_loc]
            v_page = kv[1][batch_idx, page_loc]
            k_page[:, page_offset] = append_key[t].view(num_kv_heads, head_dim)
            v_page[:, page_offset] = append_value[t].view(num_kv_heads, head_dim)
