# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""flashinfer.activation replacement (Vulkan/ROCm backend).

Element-wise activation kernels used by TRT-LLM: SiLU+mul, GELU-tanh+mul.
On ROCm these dispatch to torch._C._nn silu / python autograd; when the
Vulkan C++ backend is loaded they may call into compiled compute shaders.
"""

import torch
import torch.nn.functional as F


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


import math  # noqa: E402


def silu_and_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: silu(x) * y.

    Matches ``flashinfer.activation.silu_and_mul``.
    """
    return _swish(x) * y


def gelu_tanh_and_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """GELU-tanh activation: gelu_tanh(x) * y.

    Matches ``flashinfer.activation.gelu_tanh_and_mul``.
    """
    return _gelu_tanh(x) * y


def bmm_fp8(q, k, sm_scale, scale_q, scale_k):
    """FP8 batch matrix multiplication (stub for ROCm)."""
    import torch
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), k.float(), scale=sm_scale
    )
