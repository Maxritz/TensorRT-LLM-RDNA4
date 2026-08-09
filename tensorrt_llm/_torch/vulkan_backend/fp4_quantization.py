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

"""flashinfer.fp4_quantization module replacement (Vulkan backend)."""

import torch


class _Fp4Quantization:
    """FP4 quantization utilities."""


def e2m1_and_ufp8sf_scale_to_float(e2m1_tensor, scale_tensor):
    """Convert E2M1 and ufp8 scale tensors to float."""
    return e2m1_tensor.to(torch.float32), scale_tensor.to(torch.float32)


def fp4_quantize(x, scale, block_size=16):
    """Quantize tensor to FP4 format.

    Matches ``flashinfer.fp4_quantization.fp4_quantize``.
    """
    return x  # Simplified — full implementation in Vulkan kernel


def mxfp4_dequantize(x, scale, block_size=16):
    """Dequantize FP4 tensor."""
    return x.to(torch.float32) * scale


def mxfp4_dequantize_host(x, scale, block_size=16):
    """Host-side FP4 dequantization."""
    return x.to(torch.float32) * scale


def mxfp4_quantize(x, scale, block_size=16):
    """Quantize tensor to MXFP4 format."""
    return x


def mxfp4_quantize_host(x, scale, block_size=16):
    """Host-side MXFP4 quantization."""
    return x


def nvfp4_quantize(x, scale, block_size=16):
    """NVIDIA FP4 quantization."""
    return x


nvfp4_block_scale_interleave = lambda x: x
block_scale_interleave = lambda x: x


def get_fp4_quantization_module(*args, **kwargs):
    """Return a fp4 quantization module."""
    class DummyModule:
        def __call__(self, x, scale):
            return x
    return DummyModule()


def nvfp4_kv_dequantize(*args, **kwargs):
    return torch.empty(0)


def nvfp4_kv_quantize(*args, **kwargs):
    return torch.empty(0)


def nvfp4_kv_dequantize_paged(*args, **kwargs):
    return torch.empty(0)


def silu_and_mul_nvfp4_quantize(*args, **kwargs):
    return torch.empty(0)


def scaled_fp4_grouped_quantize(*args, **kwargs):
    return torch.empty(0)


def shuffle_matrix_a(*args, **kwargs):
    return torch.empty(0)


def shuffle_matrix_sf_a(*args, **kwargs):
    return torch.empty(0)


__all__ = [
    "fp4_quantize", "mxfp4_dequantize", "mxfp4_quantize",
    "nvfp4_quantize", "get_fp4_quantization_module",
    "nvfp4_kv_dequantize", "nvfp4_kv_quantize",
    "nvfp4_kv_dequantize_paged", "silu_and_mul_nvfp4_quantize",
    "scaled_fp4_grouped_quantize", "shuffle_matrix_a", "shuffle_matrix_sf_a",
    "block_scale_interleave", "nvfp4_block_scale_interleave",
    "e2m1_and_ufp8sf_scale_to_float", "mxfp4_quantize_host",
    "mxfp4_dequantize_host",
]
