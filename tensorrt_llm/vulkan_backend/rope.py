# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
flashinfer.rope replacement (Vulkan/ROCm backend).

Implements RoPE (rotary position embedding) applied in-place to query/key
tensors using PyTorch built-ins.
"""

import torch


def apply_rope_with_cos_sin_cache_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    inplace: bool = True,
    **kwargs,
) -> None:
    """Apply rotary position embedding to query/key tensors in-place.

    Matches ``flashinfer.rope.apply_rope_with_cos_sin_cache_inplace``.

    Parameters
    ----------
    q, k : tensors
        Query and key tensors of shape ``[..., seq_len, head_dim]``.
    cos, sin : tensors
        Pre-computed cosine and sine caches.  Can be:
        - ``[seq_len, head_dim]``: full-size cache
        - ``[seq_len, head_dim//2]``: half-size cache (interleaved RoPE)
    """
    head_dim = q.size(-1)
    seq_len = q.size(-2)
    target_shape = list(q.shape)
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)

    # If cos/sin are half-size (head_dim//2), expand to full head_dim
    # by interleaving: [a,b] -> [a,a,b,b]
    if cos.size(-1) == head_dim // 2:
        # Interleave to match the full head_dim layout
        cos = torch.stack([cos, cos], dim=-1).reshape(*cos.shape[:-1], head_dim)
        sin = torch.stack([sin, sin], dim=-1).reshape(*sin.shape[:-1], head_dim)

    # Ensure cos/sin match the seq_len of q/k
    if cos.dim() >= 2 and cos.size(-2) > seq_len:
        cos = cos[..., :seq_len, :]
        sin = sin[..., :seq_len, :]

    # Broadcast cos/sin to match q's shape by adding leading dims
    while cos.dim() < len(target_shape):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.expand(target_shape)
    sin = sin.expand(target_shape)

    # Apply rotary: rotate half of the dimensions
    # q_rot = q * cos + rotate_half(q) * sin
    # rotate_half: [-x[d//2:], x[:d//2]]
    q_half = head_dim // 2
    q_rot = torch.cat([-q[..., q_half:], q[..., :q_half]], dim=-1)
    k_rot = torch.cat([-k[..., q_half:], k[..., :q_half]], dim=-1)

    # In-place update
    q.mul_(cos).add_(q_rot * sin)
    k.mul_(cos).add_(k_rot * sin)
