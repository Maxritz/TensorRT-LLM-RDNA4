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
from typing import Optional

import numpy as np
import torch

from . import vulkan_compute as _vc

__all__ = [
    "is_available", "vulkan_attention", "vulkan_topk",
    "vulkan_softmax", "vulkan_rms_norm", "vulkan_gemm",
    "vulkan_elementwise_add",
]

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


def _dispatch_elementwise(fn_name: str, inst, host_arr: np.ndarray,
                          *dims) -> np.ndarray:
    """Generic elementwise dispatch: stage host->vulkan->host."""
    pin = inst.malloc(host_arr.nbytes)
    pout = inst.malloc(host_arr.nbytes)
    try:
        inst.memcpy_h2d(pin, host_arr.ctypes.data, host_arr.nbytes)
        fn = getattr(inst, fn_name)
        if not fn(pin, pout, *dims):
            raise RuntimeError(f"tllm_vulkan_{fn_name} dispatch failed")
        oh = np.empty(host_arr.shape, dtype=np.float32)
        inst.memcpy_d2h(oh.ctypes.data, pout, host_arr.nbytes)
    finally:
        _free(inst, pin, pout)
    return oh


def vulkan_softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax along the last dimension via the Vulkan ``softmax.comp`` shader.

    Parameters
    ----------
    x : torch.Tensor
        ``[batch, vocab]`` or ``[batch, heads, seq_len]`` GPU tensor, fp16 or fp32.

    Returns
    -------
    torch.Tensor
        Same shape, device, and dtype as *x*.
    """
    if not is_available():
        raise RuntimeError("vulkan_backend shared library is not available")
    if x.device.type != _CUDA:
        raise RuntimeError("vulkan_softmax requires GPU tensors")

    inst = _init()
    dt = x.dtype
    xf = x.detach().float().contiguous()
    xh = _as_host_array(xf)

    # Reshape to 3D [batch, heads, seq_len] for the shader
    if xh.ndim == 2:
        batch, seq_len = xh.shape
        heads = 1
    elif xh.ndim == 3:
        batch, heads, seq_len = xh.shape
    else:
        xh = xh.reshape(-1, 1, xh.shape[-1])
        batch, heads, seq_len = xh.shape

    oh = _dispatch_elementwise("softmax", inst, xh, batch, heads, seq_len, 0)
    oh = oh.reshape(xf.shape)
    out = torch.from_numpy(oh).to(device=x.device, dtype=dt)
    return out


def vulkan_rms_norm(input: torch.Tensor, weight: torch.Tensor,
                    beta: Optional[torch.Tensor] = None,
                    eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization via the Vulkan ``rms_norm.comp`` shader.

    Parameters
    ----------
    input : torch.Tensor
        ``[tokens, hidden_dim]`` GPU tensor, fp16 or fp32.
    weight : torch.Tensor
        ``[hidden_dim]`` GPU tensor (gamma).
    beta : torch.Tensor, optional
        ``[hidden_dim]`` GPU tensor (bias/β).  Defaults to zeros.
    eps : float
        Epsilon for numerical stability.

    Returns
    -------
    torch.Tensor
        Same shape, device, and dtype as *input*.
    """
    if not is_available():
        raise RuntimeError("vulkan_backend shared library is not available")
    for t in (input, weight):
        if t.device.type != _CUDA:
            raise RuntimeError("vulkan_rms_norm requires GPU tensors")
    if beta is not None and beta.device.type != _CUDA:
        raise RuntimeError("vulkan_rms_norm requires GPU beta tensor")

    inst = _init()
    dt = input.dtype
    inp = input.detach().float().contiguous()
    w = weight.detach().float().contiguous()
    b = beta.detach().float().contiguous() if beta is not None else torch.zeros_like(w)

    tokens, hidden = inp.shape
    ih = _as_host_array(inp)
    wh = _as_host_array(w)
    bh = _as_host_array(b)

    pin = inst.malloc(ih.nbytes)
    pg = inst.malloc(wh.nbytes)
    pb = inst.malloc(bh.nbytes)
    pout = inst.malloc(ih.nbytes)
    try:
        inst.memcpy_h2d(pin, ih.ctypes.data, ih.nbytes)
        inst.memcpy_h2d(pg, wh.ctypes.data, wh.nbytes)
        inst.memcpy_h2d(pb, bh.ctypes.data, bh.nbytes)
        if not inst.rms_norm(pin, pg, pb, pout, ctypes.c_float(eps), hidden, tokens):
            raise RuntimeError("tllm_vulkan_rms_norm dispatch failed")
        oh = np.empty(ih.shape, dtype=np.float32)
        inst.memcpy_d2h(oh.ctypes.data, pout, ih.nbytes)
    finally:
        _free(inst, pin, pg, pb, pout)

    out = torch.from_numpy(oh).to(device=input.device, dtype=dt)
    return out


def vulkan_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP32 GEMM: C[M, N] = A[M, K] * B[K, N] via the Vulkan ``gemm.comp`` shader.

    Parameters
    ----------
    a, b : torch.Tensor
        2-D GPU tensors, fp16 or fp32.  ``a`` is ``[M, K]``, ``b`` is ``[K, N]``.

    Returns
    -------
    torch.Tensor
        ``[M, N]`` on the input device and dtype.
    """
    if not is_available():
        raise RuntimeError("vulkan_backend shared library is not available")
    for t in (a, b):
        if t.device.type != _CUDA:
            raise RuntimeError("vulkan_gemm requires GPU tensors")

    inst = _init()
    dt = a.dtype
    af = a.detach().float().contiguous()
    bf = b.detach().float().contiguous()
    M, K = af.shape
    K2, N = bf.shape
    assert K == K2, f"shape mismatch: a[K]={K} vs b[K]={K2}"

    ah = _as_host_array(af)
    bh = _as_host_array(bf)
    ch = np.empty((M, N), dtype=np.float32)

    pa = inst.malloc(ah.nbytes)
    pb = inst.malloc(bh.nbytes)
    pc = inst.malloc(ch.nbytes)
    try:
        inst.memcpy_h2d(pa, ah.ctypes.data, ah.nbytes)
        inst.memcpy_h2d(pb, bh.ctypes.data, bh.nbytes)
        if not inst.gemm(pa, pb, pc, M, N, K):
            raise RuntimeError("tllm_vulkan_gemm dispatch failed")
        inst.memcpy_d2h(ch.ctypes.data, pc, ch.nbytes)
    finally:
        _free(inst, pa, pb, pc)

    out = torch.from_numpy(ch).to(device=a.device, dtype=dt)
    return out


def vulkan_elementwise_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise add via the Vulkan ``elementwise.comp`` shader.

    Parameters
    ----------
    a, b : torch.Tensor
        Same-shape GPU tensors, fp16 or fp32.

    Returns
    -------
    torch.Tensor
        Elementwise sum, same shape, device, and dtype.
    """
    if not is_available():
        raise RuntimeError("vulkan_backend shared library is not available")
    for t in (a, b):
        if t.device.type != _CUDA:
            raise RuntimeError("vulkan_elementwise_add requires GPU tensors")

    inst = _init()
    dt = a.dtype
    af = a.detach().float().contiguous()
    bf = b.detach().float().contiguous()
    assert af.shape == bf.shape, f"shape mismatch: {af.shape} vs {bf.shape}"

    ah = _as_host_array(af)
    bh = _as_host_array(bf)
    n = ah.size

    pa = inst.malloc(ah.nbytes)
    pb = inst.malloc(bh.nbytes)
    po = inst.malloc(ah.nbytes)
    try:
        inst.memcpy_h2d(pa, ah.ctypes.data, ah.nbytes)
        inst.memcpy_h2d(pb, bh.ctypes.data, bh.nbytes)
        if not inst.elementwise_add(pa, pb, po, n):
            raise RuntimeError("tllm_vulkan_elementwise_add dispatch failed")
        oh = np.empty(ah.shape, dtype=np.float32)
        inst.memcpy_d2h(oh.ctypes.data, po, ah.nbytes)
    finally:
        _free(inst, pa, pb, po)

    out = torch.from_numpy(oh).to(device=a.device, dtype=dt)
    return out
