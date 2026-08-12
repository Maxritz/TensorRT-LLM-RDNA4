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

"""flashinfer.fused_moe — Fused MoE operations (Vulkan-backed)."""

import numpy as np
from typing import Optional
from .tllm_enums import ActivationType, RoutingMethodType


class B12xMoEWrapper:
    """B12x MoE wrapper (Vulkan)."""

    def __init__(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        raise NotImplementedError("B12x MoE not yet implemented for Vulkan")


class MoELayer:
    """High-level MoE layer (Vulkan)."""

    def __init__(self, *args, **kwargs):
        pass


class MoEConfig:
    """MoE configuration."""
    def __init__(self, *args, **kwargs):
        pass


def fused_topk_deepseek(*args, **kwargs):
    """DeepSeek routing (stub)."""
    raise NotImplementedError("fused_topk_deepseek not implemented for Vulkan")


def cutlass_fused_moe(*args, **kwargs):
    """Cutlass fused MoE (stub)."""
    raise NotImplementedError("cutlass_fused_moe not available on Vulkan")
