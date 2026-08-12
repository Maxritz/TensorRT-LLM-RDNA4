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

"""flashinfer.rope — Rotary Positional Embedding (Vulkan-backed)."""

import numpy as np
from typing import Optional, Tuple


def _precompute_freqs(
    rotary_dim: int,
    rope_scale: float = 1.0,
    rope_theta: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute cos/sin frequency tables."""
    half = rotary_dim // 2
    freqs = 1.0 / (rope_theta ** (np.arange(0, half, dtype=np.float32) / half))
    return freqs, rope_scale


def _apply_rope_single(
    x: np.ndarray,
    pos: int,
    rotary_dim: int,
    freqs: np.ndarray,
    rope_scale: float,
    interleave: bool = False,
) -> np.ndarray:
    """Apply RoPE to a single sequence.

    Args:
        x: (..., seq_len, dim) where dim >= rotary_dim
        pos: starting position
        rotary_dim: number of dimensions to rotate
        freqs: (rotary_dim//2,) precomputed frequencies
        rope_scale: scale factor
        interleave: if True, pairs (x0,x1),(x2,x3),...; else NeoX first-half/second-half
    """
    seq_len = x.shape[-2]
    half = rotary_dim // 2

    t = np.arange(pos, pos + seq_len, dtype=np.float32) * rope_scale
    angles = np.outer(t, freqs)  # (seq_len, half)

    cos = np.cos(angles).astype(np.float32)  # (seq_len, half)
    sin = np.sin(angles).astype(np.float32)

    # Reshape cos/sin to broadcast with x: (..., 1, seq_len, half)
    shape = [1] * (x.ndim - 2) + [seq_len, half]
    cos = cos.reshape(shape)
    sin = sin.reshape(shape)

    result = x.copy()

    if interleave:
        x_even = x[..., 0:rotary_dim:2]
        x_odd = x[..., 1:rotary_dim:2]
        result[..., 0:rotary_dim:2] = x_even * cos - x_odd * sin
        result[..., 1:rotary_dim:2] = x_odd * cos + x_even * sin
    else:
        x1 = x[..., :half]
        x2 = x[..., half:rotary_dim]
        result[..., :half] = x1 * cos - x2 * sin
        result[..., half:rotary_dim] = x2 * cos + x1 * sin

    return result


def apply_rope(
    q: np.ndarray,
    k: np.ndarray,
    indptr: np.ndarray,
    offsets: np.ndarray,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1.0,
    rope_theta: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply RoPE with variable-length sequences (indptr format)."""
    if rotary_dim is None:
        rotary_dim = q.shape[-1]
    freqs, scale = _precompute_freqs(rotary_dim, rope_scale, rope_theta)

    q_out = q.copy()
    k_out = k.copy()

    for i in range(len(indptr) - 1):
        start = int(indptr[i])
        end = int(indptr[i + 1])
        pos = int(offsets[i]) if offsets.ndim > 0 else int(offsets)

        q_out[start:end] = _apply_rope_single(q[start:end], pos, rotary_dim, freqs, scale, interleave)
        k_out[start:end] = _apply_rope_single(k[start:end], pos, rotary_dim, freqs, scale, interleave)

    return q_out, k_out


def apply_rope_inplace(
    q: np.ndarray,
    k: np.ndarray,
    indptr: np.ndarray,
    offsets: np.ndarray,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1.0,
    rope_theta: float = 10000.0,
) -> None:
    """Apply RoPE in-place."""
    q_out, k_out = apply_rope(q, k, indptr, offsets, rotary_dim, interleave, rope_scale, rope_theta)
    np.copyto(q, q_out)
    np.copyto(k, k_out)


def apply_rope_pos_ids(
    q: np.ndarray,
    k: np.ndarray,
    pos_ids: np.ndarray,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1.0,
    rope_theta: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply RoPE with explicit position IDs."""
    if rotary_dim is None:
        rotary_dim = q.shape[-1]
    freqs, scale = _precompute_freqs(rotary_dim, rope_scale, rope_theta)

    q_out = q.copy()
    k_out = k.copy()

    seq_len = q.shape[0] if q.ndim >= 2 else 1
    for i in range(seq_len):
        pos = int(pos_ids[i]) if pos_ids.ndim > 0 else int(pos_ids)
        qi = q[i] if q.ndim >= 2 else q
        ki = k[i] if k.ndim >= 2 else k
        q_rot = _apply_rope_single(qi.reshape(1, -1), pos, rotary_dim, freqs, scale, interleave).reshape(qi.shape)
        k_rot = _apply_rope_single(ki.reshape(1, -1), pos, rotary_dim, freqs, scale, interleave).reshape(ki.shape)
        if q.ndim >= 2:
            q_out[i] = q_rot
            k_out[i] = k_rot
        else:
            q_out[:] = q_rot
            k_out[:] = k_rot

    return q_out, k_out


def apply_rope_pos_ids_inplace(
    q: np.ndarray,
    k: np.ndarray,
    pos_ids: np.ndarray,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1.0,
    rope_theta: float = 10000.0,
) -> None:
    """Apply RoPE with position IDs in-place."""
    q_out, k_out = apply_rope_pos_ids(q, k, pos_ids, rotary_dim, interleave, rope_scale, rope_theta)
    np.copyto(q, q_out)
    np.copyto(k, k_out)


def apply_rope_with_cos_sin_cache(
    positions: np.ndarray,
    query: np.ndarray,
    key: np.ndarray,
    head_size: int,
    cos_sin_cache: np.ndarray,
    is_neox: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply RoPE using precomputed cos/sin cache."""
    half = head_size // 2
    q_out = query.copy()
    k_out = key.copy()
    pos_len = len(positions) if positions.ndim > 0 else 1

    for i in range(pos_len):
        pos = int(positions[i]) if positions.ndim > 0 else int(positions)
        pos = min(pos, cos_sin_cache.shape[0] - 1)
        cos = cos_sin_cache[pos, :half]    # (half,)
        sin = cos_sin_cache[pos, half:]    # (half,)

        if query.ndim >= 2:
            qi = query[i]
            ki = key[i]
        else:
            qi = query
            ki = key

        qi_out = qi.copy()
        ki_out = ki.copy()

        if is_neox:
            qi_out[..., :half] = qi[..., :half] * cos - qi[..., half:] * sin
            qi_out[..., half:] = qi[..., half:] * cos + qi[..., :half] * sin
            ki_out[..., :half] = ki[..., :half] * cos - ki[..., half:] * sin
            ki_out[..., half:] = ki[..., half:] * cos + ki[..., :half] * sin
        else:
            # Interleaved
            qi_out[..., 0:head_size:2] = qi[..., 0:head_size:2] * cos[::1] - qi[..., 1:head_size:2] * sin[::1]
            qi_out[..., 1:head_size:2] = qi[..., 1:head_size:2] * cos[::1] + qi[..., 0:head_size:2] * sin[::1]
            ki_out[..., 0:head_size:2] = ki[..., 0:head_size:2] * cos[::1] - ki[..., 1:head_size:2] * sin[::1]
            ki_out[..., 1:head_size:2] = ki[..., 1:head_size:2] * cos[::1] + ki[..., 0:head_size:2] * sin[::1]

        if query.ndim >= 2:
            q_out[i] = qi_out
            k_out[i] = ki_out
        else:
            q_out[:] = qi_out
            k_out[:] = ki_out

    return q_out, k_out


def apply_rope_with_cos_sin_cache_inplace(
    positions: np.ndarray,
    query: np.ndarray,
    key: np.ndarray,
    head_size: int,
    cos_sin_cache: np.ndarray,
    is_neox: bool = False,
) -> None:
    """Apply RoPE with cos/sin cache in-place."""
    q_out, k_out = apply_rope_with_cos_sin_cache(positions, query, key, head_size, cos_sin_cache, is_neox)
    np.copyto(query, q_out)
    np.copyto(key, k_out)


def precompute_rope_freqs(
    rotary_dim: int,
    rope_scale: float = 1.0,
    rope_theta: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute RoPE frequency table (cos, sin) for all positions up to max_seq_len.

    Returns (cos, sin) arrays of shape (max_seq_len, rotary_dim).
    """
    half = rotary_dim // 2
    freqs = 1.0 / (rope_theta ** (np.arange(0, half, dtype=np.float32) / half))
    # Generate for a large default sequence length (512)
    max_seq_len = 512
    t = np.arange(max_seq_len, dtype=np.float32) * rope_scale
    angles = np.outer(t, freqs)  # (max_seq_len, half)
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    cos_full = np.concatenate([cos, cos], axis=-1)  # (max_seq_len, rotary_dim)
    sin_full = np.concatenate([sin, sin], axis=-1)
    return cos_full, sin_full


# Alias for compatibility
def precompute_rope_freqs_cores(
    rotary_dim: int,
    rope_scale: float = 1.0,
    rope_theta: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Alias for precompute_rope_freqs."""
    return precompute_rope_freqs(rotary_dim, rope_scale, rope_theta)


# Llama3.1 extended RoPE aliases
apply_llama31_rope = apply_rope
apply_llama31_rope_inplace = apply_rope_inplace
apply_llama31_rope_pos_ids = apply_rope_pos_ids
apply_llama31_rope_pos_ids_inplace = apply_rope_pos_ids_inplace
