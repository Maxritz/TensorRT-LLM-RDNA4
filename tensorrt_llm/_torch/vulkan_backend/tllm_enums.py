# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""flashinfer.tllm_enums replacement (Vulkan backend).

Enums used by the fused_moe kernels.
"""

from enum import IntEnum


class ActivationType(IntEnum):
    Sigmoid = 0
    Relu = 1
    Gelu = 2
    Silu = 3
    Swiglu = 4
    Relu2 = 5
    GeluTanh = 6


class RoutingMethodType(IntEnum):
    Default = 0
    Renormalize = 1
    TopK = 2
    DeepSeekV3 = 3
    Llama4 = 4
    Blockwise = 5
    RenormalizeNaive = 6
