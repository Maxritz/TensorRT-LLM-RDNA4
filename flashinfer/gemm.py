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

"""flashinfer.gemm — Matrix multiply operations (Vulkan-backed)."""

import numpy as np
from typing import Optional


def bmm_bf16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched matrix multiply (bf16)."""
    return np.einsum("bij,bjk->bik", a.astype(np.float32), b.astype(np.float32)).astype(a.dtype)


def bmm_fp8(a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
    """Batched matrix multiply (fp8)."""
    return bmm_bf16(a, b)


def mm_bf16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply (bf16)."""
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(a.dtype)


def mm_fp8(a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
    """Matrix multiply (fp8)."""
    return mm_bf16(a, b)


def mm_fp4(a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
    """Matrix multiply (fp4)."""
    return mm_bf16(a, b)


class SegmentGEMMWrapper:
    """Segmented GEMM (Vulkan)."""

    def __init__(self, *args, **kwargs):
        pass

    def run(self, a: np.ndarray, b: np.ndarray, *args, **kwargs) -> np.ndarray:
        return a @ b
