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

"""flashinfer.norm — Normalization kernels (Vulkan-backed)."""

import numpy as np
from typing import Optional, Tuple, Union


def rmsnorm(
    input: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
    out: Optional[np.ndarray] = None,
    enable_pdl: Optional[bool] = None,
) -> np.ndarray:
    """RMS normalization."""
    variance = np.mean(input.astype(np.float32) ** 2, axis=-1, keepdims=True)
    x_norm = input.astype(np.float32) / np.sqrt(variance + eps)
    result = (x_norm * weight.astype(np.float32)).astype(input.dtype)
    if out is not None:
        np.copyto(out, result)
        return out
    return result


def rmsnorm_quant(
    out: np.ndarray,
    input: np.ndarray,
    weight: np.ndarray,
    scale: Union[float, np.ndarray],
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    """RMS norm + quantize output (in-place)."""
    normed = rmsnorm(input, weight, eps)
    if isinstance(scale, np.ndarray):
        out[:] = (normed / scale).astype(out.dtype)
    else:
        out[:] = (normed / max(scale, 1e-30)).astype(out.dtype)


def fused_add_rmsnorm(
    input: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    """Fused add residual + RMS norm (in-place on input and residual)."""
    residual += input
    normed = rmsnorm(residual, weight, eps)
    input[:] = normed.astype(input.dtype)


def fused_add_rmsnorm_quant(
    out: np.ndarray,
    input: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    scale: Union[float, np.ndarray],
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    """Fused add residual + RMS norm + quantize."""
    residual += input
    normed = rmsnorm(residual, weight, eps)
    if isinstance(scale, np.ndarray):
        out[:] = (normed / scale).astype(out.dtype)
    else:
        out[:] = (normed / max(scale, 1e-30)).astype(out.dtype)


def gemma_rmsnorm(
    input: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
    out: Optional[np.ndarray] = None,
    enable_pdl: Optional[bool] = None,
) -> np.ndarray:
    """Gemma RMS norm (float32 precision)."""
    x = input.astype(np.float32)
    w = weight.astype(np.float32)
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    result = (x / rms * w).astype(input.dtype)
    if out is not None:
        np.copyto(out, result)
        return out
    return result


def gemma_fused_add_rmsnorm(
    input: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    """Gemma fused add residual + RMS norm (in-place)."""
    residual += input
    x = residual.astype(np.float32)
    w = weight.astype(np.float32)
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    normed = (x / rms * w).astype(input.dtype)
    input[:] = normed


def layernorm(
    input: np.ndarray,
    gemma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Layer normalization."""
    x = input.astype(np.float32)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return (x_norm * gemma.astype(np.float32) + beta.astype(np.float32)).astype(input.dtype)


def layernorm_quant(
    out: np.ndarray,
    input: np.ndarray,
    gemma: np.ndarray,
    beta: np.ndarray,
    scale: Union[float, np.ndarray],
    eps: float = 1e-5,
) -> None:
    """Layer norm + quantize."""
    normed = layernorm(input, gemma, beta, eps)
    if isinstance(scale, np.ndarray):
        out[:] = (normed / scale).astype(out.dtype)
    else:
        out[:] = (normed / max(scale, 1e-30)).astype(out.dtype)


def fused_rmsnorm_silu(
    input: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
    out: Optional[np.ndarray] = None,
    block_scale: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Fused RMS norm + SiLU activation."""
    normed = rmsnorm(input, weight, eps)
    activated = normed * (1.0 / (1.0 + np.exp(-np.clip(normed, -500, 500))))
    if out is not None:
        np.copyto(out, activated)
    if block_scale is not None:
        return out, block_scale
    return out if out is not None else activated
