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
flashinfer.sampling replacement using Vulkan compute + PyTorch/ROCm built-ins.

All functions accept the same signatures as flashinfer.sampling so callers
do not need to change their code.
"""

import torch


def _as_float(val):
    """Convert scalar tensor or float to a Python float."""
    if isinstance(val, torch.Tensor):
        return float(val.item())
    return float(val)


def _top_k_val(probs_or_logits, top_k, dim=-1):
    """Return the k-th value tensor for top-k filtering."""
    if isinstance(top_k, torch.Tensor):
        if top_k.numel() == 1:
            k = int(top_k.item())
        else:
            k = top_k
    else:
        k = int(top_k)
    kth = max(1, min(k, probs_or_logits.shape[dim]))
    vals, _ = torch.topk(probs_or_logits, kth, dim=dim)
    kth_val = vals[..., -1:]
    return kth_val


def softmax(logits: torch.Tensor, temperature=None, enable_pdl: bool = True) -> torch.Tensor:
    """FlashInfer-compatible softmax with optional temperature scaling."""
    logits = logits.to(torch.float32)
    if temperature is not None:
        if isinstance(temperature, torch.Tensor):
            temp = temperature.to(torch.float32)
            if temp.dim() > 0 and temp.numel() > 1:
                logits = logits / temp.unsqueeze(-1)
            else:
                logits = logits / float(temp.item())
        else:
            logits = logits / float(temperature)
    m = logits.max(dim=-1, keepdim=True).values
    e = torch.exp(logits - m)
    return e / e.sum(dim=-1, keepdim=True)


def top_k_mask_logits(logits: torch.Tensor, top_k) -> torch.Tensor:
    """Set logits of all but the top-k tokens to -inf."""
    kth_val = _top_k_val(logits, top_k)
    mask = torch.finfo(logits.dtype).min
    return torch.where(logits >= kth_val, logits, torch.full_like(logits, mask))


def top_k_renorm_probs(probs: torch.Tensor, top_k) -> torch.Tensor:
    """Renormalize probabilities after keeping top-k."""
    kth_val = _top_k_val(probs, top_k)
    mask = probs >= kth_val
    probs = probs * mask.to(probs.dtype)
    s = probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return probs / s


def top_p_renorm_probs(probs: torch.Tensor, top_p) -> torch.Tensor:
    """Renormalize probabilities using nucleus (top-p) sampling."""
    p = _as_float(top_p)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    cutoff = (cum > p).to(torch.bool)
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = 0
    sorted_probs = sorted_probs * (~cutoff).to(sorted_probs.dtype)
    s = sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    sorted_probs = sorted_probs / s
    return torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)


def _sample_from_cdf(probs: torch.Tensor, generator=None, seed=None, offset=None) -> torch.Tensor:
    """Inverse-CDF sampling from probability distribution."""
    cum = torch.cumsum(probs, dim=-1)
    if seed is not None and isinstance(seed, torch.Tensor):
        if seed.numel() == 1:
            torch.manual_seed(int(seed.item()))
    u = torch.rand(
        probs.shape[:-1], generator=generator,
        device=probs.device, dtype=torch.float32,
    ).unsqueeze(-1)
    idx = torch.searchsorted(cum, u, right=True)
    idx = idx.clamp(max=probs.size(-1) - 1)
    return idx.squeeze(-1).to(torch.long)


def top_k_sampling_from_probs(
    probs: torch.Tensor,
    top_k,
    *,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    check_nan: bool = False,
    generator=None,
    seed=None,
    offset=None,
) -> torch.Tensor:
    """Top-k filtered sampling from probabilities."""
    probs = top_k_renorm_probs(probs, top_k)
    return _sample_from_cdf(probs, generator, seed, offset)


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_p,
    *,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    check_nan: bool = False,
    generator=None,
    seed=None,
    offset=None,
) -> torch.Tensor:
    """Top-p filtered sampling from probabilities."""
    probs = top_p_renorm_probs(probs, top_p)
    return _sample_from_cdf(probs, generator, seed, offset)


def sampling_from_probs(
    probs: torch.Tensor,
    *,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    check_nan: bool = False,
    generator=None,
    seed=None,
    offset=None,
) -> torch.Tensor:
    """Categorical sampling from probabilities."""
    return _sample_from_cdf(probs, generator, seed, offset)


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k,
    top_p,
    *,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    check_nan: bool = False,
    generator=None,
    seed=None,
    offset=None,
) -> torch.Tensor:
    """Combined top-k + top-p sampling from probabilities."""
    if filter_apply_order == "top_k_first":
        probs = top_k_renorm_probs(probs, top_k)
        probs = top_p_renorm_probs(probs, top_p)
    else:
        probs = top_p_renorm_probs(probs, top_p)
        probs = top_k_renorm_probs(probs, top_k)
    return _sample_from_cdf(probs, generator, seed, offset)


def top_k_top_p_sampling_from_logits(
    logits: torch.Tensor,
    top_k,
    top_p,
    *,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    check_nan: bool = False,
    generator=None,
    seed=None,
    offset=None,
) -> torch.Tensor:
    """Combined top-k + top-p sampling from logits."""
    probs = softmax(logits)
    return top_k_top_p_sampling_from_probs(
        probs, top_k, top_p,
        filter_apply_order=filter_apply_order,
        deterministic=deterministic,
        check_nan=check_nan,
        generator=generator,
        seed=seed,
        offset=offset,
    )


def get_sampling_module():
    """Return this module (no-op in our pure-Python implementation)."""
    import sys
    return sys.modules[__name__]
