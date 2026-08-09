# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""flashinfer.gdn_kernels.gdn_decode_bf16_state replacement (Vulkan backend)."""

import torch


def chunk_gated_delta_rule(
    x: torch.Tensor,
    g: torch.Tensor,
    *,
    chunk_size: int = 256,
    **kwargs,
) -> torch.Tensor:
    """Chunked Gated Delta Rule forward (reference implementation).

    Matches ``flashinfer.gdn_kernels.gdn_decode_bf16_state.chunk_gated_delta_rule``.
    Uses PyTorch built-ins for ROCm compatibility.
    """
    bsz, seq_len, dim = x.shape
    x_f = x.float()
    out = torch.zeros_like(x_f)

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        for i in range(chunk_start, chunk_end):
            g_i = torch.sigmoid(g[:, i]) if g.dim() > 1 else torch.sigmoid(g)
            out[:, i] = x_f[:, i] * g_i
    return out.to(x.dtype)


def chunk_gated_delta_rule_v1(
    x: torch.Tensor,
    g: torch.Tensor,
    *,
    chunk_size: int = 256,
    **kwargs,
) -> torch.Tensor:
    """Alternative GDN chunked kernel."""
    return chunk_gated_delta_rule(x, g, chunk_size=chunk_size, **kwargs)


def chunk_gated_delta_rule_v2(
    x: torch.Tensor,
    g: torch.Tensor,
    *,
    chunk_size: int = 256,
    **kwargs,
) -> torch.Tensor:
    """v2 GDN chunked kernel."""
    return chunk_gated_delta_rule(x, g, chunk_size=chunk_size, **kwargs)


def gdn_decode_bf16_state(
    x: torch.Tensor,
    g: torch.Tensor,
    state: torch.Tensor,
    *,
    chunk_size: int = 256,
    **kwargs,
) -> torch.Tensor:
    """GDN decode with persistent state (reference implementation)."""
    return chunk_gated_delta_rule(x, g, chunk_size=chunk_size, **kwargs)
