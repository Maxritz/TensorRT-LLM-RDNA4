# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Torch <-> Vulkan compute bridge for the _torch backend.

Converts torch GPU tensors into Vulkan compute dispatches by driving the
``vulkan_compute`` ctypes bridge (``tensorrt_llm/common/vulkan_c_api.h``).

Staging model
-------------
The C++ Vulkan runtime currently allocates and owns its VkDeviceMemory; it does
not yet import external device-memory handles (e.g. hip/Vulkan external memory),
so torch device pointers cannot be handed to the kernels directly today. This
module therefore stages each tensor through host memory:

    torch GPU -> host (np) -> Vulkan device buffer -> shader -> host (np) -> torch GPU

This is the **correctness-first, fully on-device validated** path. A
direct device-pointer fast path (VK_EXTERNAL_MEMORY + hip external-memory
import in the C++ runtime) is the documented performance follow-up that would
remove the host round-trip.

The dispatchers operate on fp32. Half-precision torch tensors are upcast to
fp32 for compute and cast back to the caller's dtype on output. The attention
shader applies the standard ``1/sqrt(head_dim)`` scaling internally, so this
helper must be used with the default scale (callers needing a custom
``sm_scale`` should stay on the PyTorch SDPA fallback).
"""

import ctypes

import numpy as np
import torch

from . import vulkan_compute as _vc

__all__ = ["is_available", "vulkan_attention", "vulkan_topk"]

_CUDA = "cuda"


def is_available() -> bool:
    """True when the staged ``libvulkan_backend`` shared library is loadable."""
    return _vc.is_available()


def _init():
    inst = _vc._vk
    if not inst.init(0):
        raise RuntimeError("Vulkan backend failed to initialize on GPU 0")
    return inst


def _as_host_array(t: torch.Tensor) -> np.ndarray:
    """Return a C-contiguous fp32 host view of a (GPU) tensor."""
    return np.ascontiguousarray(t.detach().float().contiguous().cpu().numpy())


def _free(inst, *ptrs):
    for p in ptrs:
        inst.free(p)


def vulkan_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     causal: bool) -> torch.Tensor:
    """Scaled dot-product attention via the Vulkan ``attention.comp`` shader.

    Parameters
    ----------
    q, k, v : torch.Tensor
        ``[B, num_heads, seq_q/k, head_dim]`` on a GPU, float16 or float32.
    causal : bool
        If True, keys ``j > i`` are masked.

    Returns
    -------
    torch.Tensor
        ``[B, num_heads, seq_q, head_dim]`` on the input device and dtype.

    Notes
    -----
    The shader scales by ``1/sqrt(head_dim)`` (standard SDPA default). For a
    custom scale, use ``torch.nn.functional.scaled_dot_product_attention``
    instead. Sliding-window / custom masks are not supported here (no Vulkan
    shader path); callers must fall back to SDPA for those cases.
    """
    if not is_available():
        raise RuntimeError("vulkan_backend shared library is not available")
    for t in (q, k, v):
        if t.device.type != _CUDA:
            raise RuntimeError("vulkan_attention requires GPU tensors")

    inst = _init()
    dt = q.dtype
    qf = q.detach().float().contiguous()
    kf = k.detach().float().contiguous()
    vf = v.detach().float().contiguous()
    B, nh, sq, hd = qf.shape
    _, _, sk, _hd = kf.shape

    qh = _as_host_array(qf)
    kh = _as_host_array(kf)
    vh = _as_host_array(vf)

    pq = inst.malloc(qh.nbytes)
    pk = inst.malloc(kh.nbytes)
    pv = inst.malloc(vh.nbytes)
    po = inst.malloc(qh.nbytes)
    try:
        inst.memcpy_h2d(pq, qh.ctypes.data, qh.nbytes)
        inst.memcpy_h2d(pk, kh.ctypes.data, kh.nbytes)
        inst.memcpy_h2d(pv, vh.ctypes.data, vh.nbytes)
        if not inst.attention(pq, pk, pv, po, B, nh, sq, sk, hd, int(causal)):
            raise RuntimeError("tllm_vulkan_attention dispatch failed")
        oh = np.empty(qh.shape, dtype=np.float32)
        inst.memcpy_d2h(oh.ctypes.data, po, qh.nbytes)
    finally:
        _free(inst, pq, pk, pv, po)

    out = torch.from_numpy(oh).to(device=q.device, dtype=dt)
    return out


def vulkan_topk(scores: torch.Tensor, topk: int,
                input_offsets: torch.Tensor, output_offsets: torch.Tensor) -> torch.Tensor:
    """Top-k per (batch, head) row via the Vulkan ``topk.comp`` shader.

    Parameters
    ----------
    scores : torch.Tensor
        ``[num_heads, total_tokens]`` float32 GPU tensor (flat per head).
    input_offsets, output_offsets : torch.Tensor
        ``[batch_size + 1]`` uint32 exclusive scans.
    topk : int
        Maximum picks per row.

    Returns
    -------
    torch.Tensor
        ``[num_heads, total_output_tokens]`` int32 row-local selections.
    """
    if not is_available():
        raise RuntimeError("vulkan_backend shared library is not available")
    for t in (scores, input_offsets, output_offsets):
        if t.device.type != _CUDA:
            raise RuntimeError("vulkan_topk requires GPU tensors")

    inst = _init()
    num_heads = scores.shape[0]
    total_tokens = scores.shape[1]
    batch_size = input_offsets.shape[0] - 1
    total_output_tokens = int(output_offsets[-1].item())

    sh = scores.detach().float().contiguous().cpu().numpy()
    inh = input_offsets.detach().to(torch.uint32).contiguous().cpu().numpy()
    outh = output_offsets.detach().to(torch.uint32).contiguous().cpu().numpy()

    p_scores = inst.malloc(sh.nbytes)
    p_in = inst.malloc(inh.nbytes)
    p_out = inst.malloc(outh.nbytes)
    n_out = num_heads * total_output_tokens
    n_out_bytes = n_out * 4
    p_idx = inst.malloc(n_out_bytes)
    try:
        inst.memcpy_h2d(p_scores, sh.ctypes.data, sh.nbytes)
        inst.memcpy_h2d(p_in, inh.ctypes.data, inh.nbytes)
        inst.memcpy_h2d(p_out, outh.ctypes.data, outh.nbytes)
        if not inst.topk(p_scores, p_in, p_out, p_idx,
                         topk, num_heads, batch_size, total_tokens, total_output_tokens):
            raise RuntimeError("tllm_vulkan_topk dispatch failed")
        oh = np.empty(n_out, dtype=np.int32)
        inst.memcpy_d2h(oh.ctypes.data, p_idx, n_out_bytes)
    finally:
        _free(inst, p_scores, p_in, p_out, p_idx)

    return torch.from_numpy(oh).to(device=scores.device, dtype=torch.int32)
