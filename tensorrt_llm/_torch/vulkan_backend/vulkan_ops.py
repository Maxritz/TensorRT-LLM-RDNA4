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

"""Numpy implementations for ops not yet in the Vulkan backend.

These run on CPU. The heavy compute (GEMM, attention, norm) goes through
Vulkan; these are lightweight helper ops that complete the transformer.
"""

import numpy as np
from typing import Optional, Tuple


def embedding(weights: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Look up embeddings. weights: (vocab, dim), indices: (...) -> (..., dim)."""
    return weights[indices]


def rope(
    q: np.ndarray,
    k: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply rotary position embedding.

    Args:
        q, k: (tokens, heads, head_dim) or (batch, seq, heads, head_dim) float32
        cos: (seq, head_dim//2) float32
        sin: (seq, head_dim//2) float32
    Returns:
        (q_rot, k_rot) same shape as input
    """
    head_dim = q.shape[-1]
    half = head_dim // 2

    # cos/sin are (seq, head_dim//2)
    # q is (tokens, heads, head_dim) or (batch, seq, heads, head_dim)
    # Need to broadcast cos/sin to match q shape
    if q.ndim == 3:  # (tokens, heads, head_dim)
        # cos: (tokens, head_dim//2) -> (tokens, 1, head_dim//2)
        cos = cos[:, np.newaxis, :]
        sin = sin[:, np.newaxis, :]
    elif q.ndim == 4:  # (batch, seq, heads, head_dim)
        # cos: (seq, head_dim//2) -> (1, seq, 1, head_dim//2)
        cos = cos[np.newaxis, :, np.newaxis, :]
        sin = sin[np.newaxis, :, np.newaxis, :]

    # Split q/k into first half and second half
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]

    # Apply rotation: [q1, q2] -> [q1*cos - q2*sin, q2*cos + q1*sin]
    # Interleave: rotate_half = [-q2, q1]
    q_rot = np.concatenate([-q2, q1], axis=-1)
    k_rot = np.concatenate([-k2, k1], axis=-1)

    # broadcast cos/sin to full head_dim
    cos_full = np.concatenate([cos, cos], axis=-1)
    sin_full = np.concatenate([sin, sin], axis=-1)

    q_out = q * cos_full + q_rot * sin_full
    k_out = k * cos_full + k_rot * sin_full
    return q_out, k_out


def precompute_rope_freqs(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute RoPE frequency table.

    Returns:
        (cos_table, sin_table) each (seq_len, head_dim//2) float32
    """
    half = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, half, 2, dtype=np.float32) / half))
    t = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(t, freqs)  # (seq_len, half//2)

    cos_table = np.cos(angles).astype(np.float32)
    sin_table = np.sin(angles).astype(np.float32)

    # Expand to full head_dim by repeating
    cos_full = np.repeat(cos_table, 2, axis=-1)  # (seq_len, head_dim)
    sin_full = np.repeat(sin_table, 2, axis=-1)

    return cos_full, sin_full


def topk_cpu(logits: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Top-K sampling on CPU (numpy). Returns (indices, values)."""
    # logits: (vocab_size,) or (batch, vocab_size)
    if logits.ndim == 1:
        idx = np.argpartition(-logits, k)[:k]
        idx = idx[np.argsort(-logits[idx])]
        return idx, logits[idx]
    else:
        results_idx = np.empty((logits.shape[0], k), dtype=np.int64)
        results_val = np.empty((logits.shape[0], k), dtype=np.float32)
        for i in range(logits.shape[0]):
            idx = np.argpartition(-logits[i], k)[:k]
            idx = idx[np.argsort(-logits[i][idx])]
            results_idx[i] = idx
            results_val[i] = logits[i][idx]
        return results_idx, results_val


def softmax_cpu(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax on CPU."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def sample_from_probs(
    probs: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> int:
    """Sample a token from probability distribution (CPU).

    Args:
        probs: (vocab_size,) probability distribution
        temperature: sampling temperature
        top_k: keep only top k tokens (0 = disabled)
        top_p: nucleus sampling threshold (0.0 = disabled)
    Returns:
        sampled token index
    """
    logits = np.log(probs + 1e-30) / max(temperature, 1e-6)

    if top_k > 0:
        kth_val = np.sort(-logits)[min(top_k - 1, len(logits) - 1)]
        logits[logits < -kth_val] = -np.inf

    if top_p > 0.0:
        sorted_idx = np.argsort(-logits)
        sorted_logits = logits[sorted_idx]
        cumsum = np.cumsum(softmax_cpu(sorted_logits))
        cutoff = cumsum > top_p
        if np.any(cutoff):
            first_true = np.argmax(cutoff)
            sorted_logits[first_true + 1:] = -np.inf
            logits[sorted_idx] = sorted_logits

    probs_final = softmax_cpu(logits)
    return int(np.random.choice(len(probs_final), p=probs_final))


def rms_norm_cpu(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMS normalization on CPU."""
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    x_norm = x * (1.0 / np.sqrt(variance + eps))
    return x_norm * weight


def append_paged_kv_cache(
    append_key: np.ndarray,
    append_value: np.ndarray,
    batch_indices: np.ndarray,
    positions: np.ndarray,
    kv_pages: np.ndarray,
    kv_indices: np.ndarray,
    kv_indptr: np.ndarray,
    kv_last_page_len: np.ndarray,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    """Append key/value tokens to a paged KV cache (pure numpy).

    Matches the flashinfer.page.append_paged_kv_cache API.

    Args:
        append_key: (n_tokens, num_kv_heads, head_dim) new keys
        append_value: (n_tokens, num_kv_heads, head_dim) new values
        batch_indices: (n_tokens,) which batch slot each token belongs to
        positions: (n_tokens,) position within the sequence for each token
        kv_pages: (batch, max_pages, num_kv_heads, page_size, head_dim) cache
        kv_indices: (total_pages,) page index mapping (flattened indptr→page)
        kv_indptr: (batch+1,) per-batch page start offsets into kv_indices
        kv_last_page_len: (batch,) valid entries in last page per batch
        page_size: tokens per page
        num_kv_heads: number of KV heads
        head_dim: dimension per head
    """
    n_tokens = append_key.shape[0]
    for t in range(n_tokens):
        b = int(batch_indices[t])
        pos = int(positions[t])
        page_idx = pos // page_size
        page_offset = pos % page_size

        seq_start = int(kv_indptr[b])
        seq_end = int(kv_indptr[b + 1])
        local_page = seq_start + page_idx

        k_val = append_key[t].reshape(num_kv_heads, head_dim)
        v_val = append_value[t].reshape(num_kv_heads, head_dim)

        kv_pages[b, local_page, :, page_offset, :] = k_val
        kv_pages[b, local_page, :, page_offset, :] = v_val
