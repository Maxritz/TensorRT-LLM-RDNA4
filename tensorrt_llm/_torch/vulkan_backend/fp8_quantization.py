# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""flashinfer.fp8_quantization module replacement (Vulkan backend)."""

import torch


def mxfp8_quantize(x, scale, block_size=16):
    """MX-FP8 quantization."""
    return x


def mxfp8_quantize_host(x, scale, block_size=16):
    """Host-side MX-FP8 quantization."""
    return x


def mxfp8_dequantize(x, scale, block_size=16):
    """MX-FP8 dequantization."""
    return x.to(torch.float32) * scale


def mxfp8_dequantize_host(x, scale, block_size=16):
    """Host-side MX-FP8 dequantization."""
    return x.to(torch.float32) * scale


__all__ = [
    "mxfp8_quantize", "mxfp8_quantize_host",
    "mxfp8_dequantize", "mxfp8_dequantize_host",
]
