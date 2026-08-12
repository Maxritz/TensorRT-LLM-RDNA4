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

"""flashinfer.activation — Activation functions (Vulkan-backed)."""

import numpy as np
from typing import Optional, Tuple


def silu_and_mul(
    input: np.ndarray,
    out: Optional[np.ndarray] = None,
    enable_pdl: Optional[bool] = None,
) -> np.ndarray:
    """SiLU activation on first half, multiply with second half.

    input: (N, 2*D) -> output: (N, D) where output = silu(input[:,:D]) * input[:,D:]
    """
    D = input.shape[-1] // 2
    a = input[..., :D].astype(np.float32)
    b = input[..., D:].astype(np.float32)
    silu_a = a * (1.0 / (1.0 + np.exp(-np.clip(a, -500, 500))))
    result = (silu_a * b).astype(input.dtype)
    if out is not None:
        if out.shape != result.shape:
            out = out[..., :result.shape[-1]]
        np.copyto(out, result)
        return out
    return result


def gelu_and_mul(
    input: np.ndarray,
    out: Optional[np.ndarray] = None,
    enable_pdl: Optional[bool] = None,
) -> np.ndarray:
    """GELU activation on first half, multiply with second half."""
    D = input.shape[-1] // 2
    a = input[..., :D].astype(np.float32)
    b = input[..., D:].astype(np.float32)
    gelu_a = 0.5 * a * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (a + 0.044715 * a ** 3)))
    result = (gelu_a * b).astype(input.dtype)
    if out is not None:
        if out.shape != result.shape:
            out = out[..., :result.shape[-1]]
        np.copyto(out, result)
        return out
    return result


def gelu_tanh_and_mul(
    input: np.ndarray,
    out: Optional[np.ndarray] = None,
    enable_pdl: Optional[bool] = None,
) -> np.ndarray:
    """GELU (tanh approximation) on first half, multiply with second half."""
    D = input.shape[-1] // 2
    a = input[..., :D].astype(np.float32)
    b = input[..., D:].astype(np.float32)
    gelu_a = 0.5 * a * (1.0 + np.tanh(0.7978845608 * (a + 0.044715 * a ** 3)))
    result = (gelu_a * b).astype(input.dtype)
    if out is not None:
        if out.shape != result.shape:
            out = out[..., :result.shape[-1]]
        np.copyto(out, result)
        return out
    return result
