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

"""flashinfer.tllm_enums — TensorRT-LLM enumerations (pure Python)."""

from enum import IntEnum


class RoutingMethodType(IntEnum):
    Default = 0
    Renormalize = 1
    DeepSeekV3 = 2
    Llama4 = 3
    RenormalizeNaive = 4
    TopK = 5
    SigmoidRenorm = 6
    MiniMax2 = 7
    Sigmoid = 8
    Unspecified = 9


class RoutingInputMode(IntEnum):
    FromLogits = 0
    PackedPrecomputed = 1
    UnpackedPrecomputed = 2


class ActivationType(IntEnum):
    Gelu = 0
    Relu = 1
    Silu = 2
    Swiglu = 3
    Geglu = 4
    SwigluBias = 5
    Relu2 = 6
    SwigluStep = 7
    GegluTanh = 8
    Identity = 9
    Situ = 10
    InvalidType = 11


class SfLayout(IntEnum):
    layout_128x4 = 0
    layout_8x4 = 1
    layout_linear = 2


class WeightLayout(IntEnum):
    MajorK = 0
    MajorMn = 1
    BlockMajorK = 2


class GatedActType(IntEnum):
    SwiGlu = 0
    GeGlu = 1


class Fp8QuantizationType(IntEnum):
    NoneFp8 = 0
    DeepSeekFp8 = 1
    MxFp8 = 2


def is_gated_activation(activation_type) -> bool:
    """Check if activation type is a gated activation."""
    at = ActivationType(activation_type) if not isinstance(activation_type, ActivationType) else activation_type
    return at in (ActivationType.Swiglu, ActivationType.Geglu, ActivationType.SwigluBias,
                   ActivationType.SwigluStep, ActivationType.GegluTanh)
