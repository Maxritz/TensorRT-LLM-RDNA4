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

"""Pure-numpy Vulkan compute bridge.

Replaces torch_bridge.py with zero torch dependency. All staging uses
numpy arrays and the vulkan_compute ctypes API directly.
"""

import ctypes
from typing import Optional, Tuple

import numpy as np

from . import vulkan_compute as _vc

__all__ = [
    "VulkanDevice",
    "vulkan_attention",
    "vulkan_softmax",
    "vulkan_rms_norm",
    "vulkan_layer_norm",
    "vulkan_gemm",
    "vulkan_elementwise_add",
    "vulkan_elementwise_mul",
    "vulkan_silu",
    "vulkan_gelu",
    "vulkan_swiglu",
    "vulkan_sigmoid",
    "vulkan_relu",
    "vulkan_sigmoid_mul",
    "vulkan_scale_rows",
    "vulkan_topk_general",
]


class VulkanDevice:
    """Manages Vulkan device memory and compute operations."""

    def __init__(self):
        self._vk = _vc._vk
        if not self._vk.init(0):
            raise RuntimeError("Vulkan backend failed to initialize on GPU 0")

    @property
    def is_active(self) -> bool:
        return self._vk.is_active()

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        return self._vk.malloc(nbytes)

    def free(self, ptr: ctypes.c_void_p):
        self._vk.free(ptr)

    def memcpy_h2d(self, dst: ctypes.c_void_p, src: np.ndarray, nbytes: int):
        self._vk.memcpy_h2d(dst, src.ctypes.data, nbytes)

    def memcpy_d2h(self, dst: np.ndarray, src: ctypes.c_void_p, nbytes: int):
        self._vk.memcpy_d2h(dst.ctypes.data, src, nbytes)

    def synchronize(self):
        self._vk.device_synchronize()

    def upload(self, arr: np.ndarray) -> ctypes.c_void_p:
        """Upload numpy array to Vulkan device. Returns device pointer."""
        arr = np.ascontiguousarray(arr)
        ptr = self.malloc(arr.nbytes)
        self.memcpy_h2d(ptr, arr, arr.nbytes)
        return ptr

    def download(self, ptr: ctypes.c_void_p, shape: tuple, dtype=np.float32) -> np.ndarray:
        """Download Vulkan device buffer to numpy array."""
        arr = np.empty(shape, dtype=dtype)
        self.memcpy_d2h(arr, ptr, arr.nbytes)
        return arr


def _ensure_f32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.float32:
        return arr.astype(np.float32)
    return arr


def vulkan_attention(
    dev: VulkanDevice,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal: bool = True,
) -> np.ndarray:
    """Scaled dot-product attention via Vulkan.

    Args:
        dev: VulkanDevice
        q, k, v: (B, H, S, D) float32 arrays
        causal: whether to apply causal mask
    Returns:
        output: (B, H, S, D) float32 array
    """
    B, H, S_q, D = q.shape
    S_k = k.shape[2]

    pq = dev.upload(q)
    pk = dev.upload(k)
    pv = dev.upload(v)
    pout = dev.upload(np.zeros((B, H, S_q, D), dtype=np.float32))

    dev._vk.attention(pq, pk, pv, pout,
                      B, H, S_q, S_k, D, 1 if causal else 0)
    dev.synchronize()

    result = dev.download(pout, (B, H, S_q, D))
    dev.free(pq); dev.free(pk); dev.free(pv); dev.free(pout)
    return result


def vulkan_softmax(dev: VulkanDevice, x: np.ndarray) -> np.ndarray:
    """Softmax over last dimension via Vulkan.

    Args:
        dev: VulkanDevice
        x: (batch, heads, seqlen) or (batch, heads, seqlen, 1) or (batch, seqlen) float32
    Returns:
        softmax output, same shape
    """
    original_ndim = x.ndim
    if x.ndim == 4:
        B, H, S, _ = x.shape
    elif x.ndim == 3:
        B, H, S = x.shape
    elif x.ndim == 2:
        # (batch, seqlen) -> treat as (batch, 1, seqlen)
        B, S = x.shape
        H = 1
        x = x.reshape(B, H, S)
    else:
        raise ValueError(f"vulkan_softmax: unsupported shape {x.shape}")
    dtype_val = 0  # fp32

    px = dev.upload(x)
    pout = dev.upload(np.zeros_like(x))
    dev._vk.softmax(px, pout, B, H, S, dtype_val)
    dev.synchronize()

    result = dev.download(pout, (B, H, S))
    dev.free(px); dev.free(pout)
    if original_ndim == 2:
        result = result.reshape(B, S)
    return result


def vulkan_rms_norm(
    dev: VulkanDevice,
    input_arr: np.ndarray,
    weight: np.ndarray,
    beta: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """RMS normalization via Vulkan."""
    hidden_dim = input_arr.shape[-1]
    token_count = input_arr.size // hidden_dim
    flat = input_arr.ravel().astype(np.float32)
    w_flat = np.broadcast_to(weight, hidden_dim).astype(np.float32).ravel()
    b_flat = np.zeros(hidden_dim, dtype=np.float32) if beta is None else beta.astype(np.float32).ravel()

    pin = dev.upload(flat)
    pw = dev.upload(w_flat)
    pb = dev.upload(b_flat)
    pout = dev.upload(np.zeros_like(flat))

    dev._vk.rms_norm(pin, pw, pb, pout, eps, hidden_dim, token_count)
    dev.synchronize()

    result = dev.download(pout, input_arr.shape)
    dev.free(pin); dev.free(pw); dev.free(pb); dev.free(pout)
    return result


def vulkan_layer_norm(
    dev: VulkanDevice,
    input_arr: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Standard LayerNorm via Vulkan."""
    hidden_dim = input_arr.shape[-1]
    token_count = input_arr.size // hidden_dim
    flat = input_arr.ravel().astype(np.float32)
    w_flat = np.broadcast_to(weight, hidden_dim).astype(np.float32).ravel()
    b_flat = np.zeros(hidden_dim, dtype=np.float32) if bias is None else bias.astype(np.float32).ravel()

    pin = dev.upload(flat)
    pw = dev.upload(w_flat)
    pb = dev.upload(b_flat)
    pout = dev.upload(np.zeros_like(flat))

    dev._vk.layer_norm(pin, pw, pb, pout, eps, hidden_dim, token_count)
    dev.synchronize()

    result = dev.download(pout, input_arr.shape)
    dev.free(pin); dev.free(pw); dev.free(pb); dev.free(pout)
    return result


def vulkan_gemm(dev: VulkanDevice, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply C = A @ B via Vulkan. All fp32."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    pa = dev.upload(a.ravel())
    pb = dev.upload(b.ravel())
    pout = dev.upload(np.zeros(M * N, dtype=np.float32))

    dev._vk.gemm(pa, pb, pout, M, N, K)
    dev.synchronize()

    result = dev.download(pout, (M, N))
    dev.free(pa); dev.free(pb); dev.free(pout)
    return result


def vulkan_q8_0_gemm(
    dev: VulkanDevice,
    weight: np.ndarray,
    activation: np.ndarray,
    M: int, N: int, K: int,
    blocks_per_row: int,
) -> np.ndarray:
    """Dequantized matmul via Vulkan. weight is packed Q8_0."""
    pw = dev.upload(weight.ravel().astype(np.uint8))
    pa = dev.upload(activation.ravel().astype(np.float32))
    pout = dev.upload(np.zeros(M * N, dtype=np.float32))

    dev._vk.q8_0_gemm(pw, pa, pout, M, N, K, blocks_per_row)
    dev.synchronize()

    result = dev.download(pout, (M, N))
    dev.free(pw); dev.free(pa); dev.free(pout)
    return result


def vulkan_elementwise_add(dev: VulkanDevice, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise add via Vulkan."""
    out_shape = a.shape
    a = _ensure_f32(a).ravel()
    b = _ensure_f32(b).ravel()
    assert a.shape == b.shape
    N = a.size

    pa = dev.upload(a); pb = dev.upload(b)
    pout = dev.upload(np.zeros(N, dtype=np.float32))
    dev._vk.elementwise_add(pa, pb, pout, N)
    dev.synchronize()
    result = dev.download(pout, out_shape)
    dev.free(pa); dev.free(pb); dev.free(pout)
    return result


def vulkan_elementwise_mul(dev: VulkanDevice, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise multiply via Vulkan."""
    out_shape = a.shape
    a = _ensure_f32(a).ravel()
    b = _ensure_f32(b).ravel()
    assert a.shape == b.shape
    N = a.size

    pa = dev.upload(a); pb = dev.upload(b)
    pout = dev.upload(np.zeros(N, dtype=np.float32))
    dev._vk.elementwise_mul(pa, pb, pout, N)
    dev.synchronize()
    result = dev.download(pout, out_shape)
    dev.free(pa); dev.free(pb); dev.free(pout)
    return result


def vulkan_silu(dev: VulkanDevice, x: np.ndarray) -> np.ndarray:
    """SiLU activation via Vulkan."""
    flat = _ensure_f32(x).ravel()
    N = flat.size
    px = dev.upload(flat)
    pout = dev.upload(np.zeros(N, dtype=np.float32))
    dev._vk.silu(px, pout, N)
    dev.synchronize()
    result = dev.download(pout, x.shape)
    dev.free(px); dev.free(pout)
    return result


def vulkan_gelu(dev: VulkanDevice, x: np.ndarray) -> np.ndarray:
    """GELU activation via Vulkan."""
    flat = _ensure_f32(x).ravel()
    N = flat.size
    px = dev.upload(flat)
    pout = dev.upload(np.zeros(N, dtype=np.float32))
    dev._vk.gelu(px, pout, N)
    dev.synchronize()
    result = dev.download(pout, x.shape)
    dev.free(px); dev.free(pout)
    return result


def vulkan_swiglu(dev: VulkanDevice, x: np.ndarray, hidden_dim: int) -> np.ndarray:
    """SwiGLU: input [N, 2*H] -> output [N, H] via Vulkan."""
    x = _ensure_f32(x)
    flat = x.ravel()
    N = flat.size // (hidden_dim * 2)
    px = dev.upload(flat)
    pout = dev.upload(np.zeros(N * hidden_dim, dtype=np.float32))
    dev._vk.swiglu(px, pout, hidden_dim, N)
    dev.synchronize()
    result = dev.download(pout, (N, hidden_dim))
    dev.free(px); dev.free(pout)
    return result


def vulkan_sigmoid(dev: VulkanDevice, x: np.ndarray) -> np.ndarray:
    """Sigmoid activation via Vulkan."""
    flat = _ensure_f32(x).ravel()
    N = flat.size
    px = dev.upload(flat)
    pout = dev.upload(np.zeros(N, dtype=np.float32))
    dev._vk.sigmoid(px, pout, N)
    dev.synchronize()
    result = dev.download(pout, x.shape)
    dev.free(px); dev.free(pout)
    return result


def vulkan_relu(dev: VulkanDevice, x: np.ndarray) -> np.ndarray:
    """ReLU activation via Vulkan."""
    flat = _ensure_f32(x).ravel()
    N = flat.size
    px = dev.upload(flat)
    pout = dev.upload(np.zeros(N, dtype=np.float32))
    dev._vk.relu(px, pout, N)
    dev.synchronize()
    result = dev.download(pout, x.shape)
    dev.free(px); dev.free(pout)
    return result


def vulkan_sigmoid_mul(dev: VulkanDevice, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fused a * sigmoid(b) via Vulkan."""
    a = _ensure_f32(a).ravel()
    b = _ensure_f32(b).ravel()
    assert a.shape == b.shape
    N = a.size
    pa = dev.upload(a); pb = dev.upload(b)
    pout = dev.upload(np.zeros(N, dtype=np.float32))
    dev._vk.sigmoid_mul(pa, pb, pout, N)
    dev.synchronize()
    result = dev.download(pout, a.shape)
    dev.free(pa); dev.free(pb); dev.free(pout)
    return result


def vulkan_scale_rows(
    dev: VulkanDevice, input_arr: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Row-wise broadcast multiply: out[i,j] = in[i,j] * scale[i]."""
    input_arr = _ensure_f32(input_arr)
    scale = _ensure_f32(scale).ravel()
    rows, cols = input_arr.shape
    assert scale.size == rows

    pin = dev.upload(input_arr.ravel())
    pscale = dev.upload(scale)
    pout = dev.upload(np.zeros(rows * cols, dtype=np.float32))
    dev._vk.scale_rows(pin, pscale, pout, rows, cols)
    dev.synchronize()
    result = dev.download(pout, (rows, cols))
    dev.free(pin); dev.free(pscale); dev.free(pout)
    return result


def vulkan_topk_general(
    dev: VulkanDevice,
    scores: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-K per row via Vulkan. Returns (indices, values)."""
    scores = _ensure_f32(scores)
    rows, cols = scores.shape
    N = rows * cols

    pscores = dev.upload(scores.ravel())
    pidx = dev.upload(np.zeros(N, dtype=np.uint32))
    pval = dev.upload(np.zeros(N, dtype=np.float32))
    dev._vk.topk_general(pscores, pidx, pval, rows, cols, k)
    dev.synchronize()

    idx = dev.download(pidx, (rows, k), dtype=np.uint32)
    val = dev.download(pval, (rows, k))
    dev.free(pscores); dev.free(pidx); dev.free(pval)
    return idx, val
