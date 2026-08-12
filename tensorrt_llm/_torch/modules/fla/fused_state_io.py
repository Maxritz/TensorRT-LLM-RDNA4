# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused state I/O kernels for the FlashInfer GDN prefill adapter.

The SSM state pool and FlashInfer share the ``[slots, HV, V, K]`` layout
(K innermost), so no transpose is needed. These functions fuse the gather +
dtype cast (and the reverse cast + scatter) that bridge the bf16 pool and
FlashInfer's fp32 state, avoiding the naive multi-launch PyTorch chain
(``pool[indices].to(fp32)`` / ``.to(bf16)`` + indexed scatter) and its large
intermediate buffers:

- ``gather_cast_vk_to_fp32_vk``: gather pool slots by index + cast bf16->fp32.
- ``cast_scatter_fp32_vk_to_vk``: cast fp32->bf16 + scatter back to pool slots.

When TLLM_VULKAN_BACKEND=1, these are pure numpy implementations.
When torch is available, they provide torch-compatible wrappers.
"""

from typing import Optional, Union

import numpy as np


def _to_numpy(arr):
    """Convert numpy array or torch tensor to numpy."""
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray(arr)


def gather_cast(initial_state: np.ndarray,
                initial_state_indices: Optional[np.ndarray] = None,
                out_dtype: Union[np.dtype, str] = np.float32) -> np.ndarray:
    """Fused ``initial_state[indices].to(out_dtype).contiguous()`` for ``[N, H, V, K]`` state.

    ``out_dtype`` defaults to ``np.float32``.
    """
    assert initial_state.ndim == 4, f"initial_state must be 4D, got {initial_state.shape}"
    n_pool, h, v, k = initial_state.shape
    if initial_state_indices is not None:
        indices = _to_numpy(initial_state_indices).astype(np.int64)
        output = initial_state[indices].astype(out_dtype)
    else:
        output = initial_state.astype(out_dtype).copy()
    return output


def cast_scatter(src: np.ndarray,
                 dst: np.ndarray,
                 scatter_indices: Optional[np.ndarray] = None) -> None:
    """Fused dtype cast plus optional indexed scatter for ``[N, H, V, K]`` state."""
    assert src.ndim == 4, f"src must be 4D, got {src.shape}"
    assert dst.ndim == 4, f"dst must be 4D, got {dst.shape}"
    num_seqs, h, v, k = src.shape
    assert dst.shape[1:] == (h, v, k), (
        f"dst shape {tuple(dst.shape)} incompatible with src {tuple(src.shape)}"
    )
    if scatter_indices is not None:
        indices = _to_numpy(scatter_indices).astype(np.int64)
        assert indices.shape == (num_seqs,)
        for i in range(num_seqs):
            dst[indices[i]] = src[i].astype(dst.dtype)
    else:
        dst[...] = src.astype(dst.dtype)


# Aliases matching the original torch-based names
gather_cast_vk_to_fp32_vk = gather_cast
cast_scatter_fp32_vk_to_vk = cast_scatter


# Torch-compatible wrappers
def gather_cast_vk_to_fp32_vk_torch(initial_state, initial_state_indices=None, out_dtype=None):
    """Torch wrapper for gather_cast."""
    if out_dtype is None:
        import torch
        out_dtype = torch.float32
    np_result = gather_cast(
        _to_numpy(initial_state),
        _to_numpy(initial_state_indices) if initial_state_indices is not None else None,
        out_dtype=str(out_dtype) if isinstance(out_dtype, str) else out_dtype,
    )
    import torch
    return torch.from_numpy(np_result).to(initial_state.device)


def cast_scatter_fp32_vk_to_vk_torch(src_vk, dst, scatter_indices=None):
    """Torch wrapper for cast_scatter."""
    cast_scatter(
        _to_numpy(src_vk),
        _to_numpy(dst),
        _to_numpy(scatter_indices) if scatter_indices is not None else None,
    )