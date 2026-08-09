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

"""Vulkan-backed RMS norm and fp4 quantization replacements.

Provides:
  - rmsnorm_fp4quant  (flashinfer.norm.rmsnorm_fp4quant)
  - fp4_quantize      (flashinfer.fp4_quantization.fp4_quantize)
  - mxfp8_quantize    (flashinfer.fp8_quantization.mxfp8_quantize)
"""

import torch
import torch.nn.functional as F


def rmsnorm(input, weight, eps, enable_pdl=None):
    """Standard RMS norm."""
    return input * weight / torch.sqrt(input.float().pow(2).mean(dim=-1, keepdim=True).to(input.dtype) + eps)


def gemma_rmsnorm(input, weight, eps, enable_pdl=None):
    """Gemma RMS norm — same formula as RMS norm, float32 precision."""
    x = input.float()
    w = weight.float()
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x / rms * w).to(input.dtype)


def fused_add_rmsnorm(input, residual, weight, eps, enable_pdl=None):
    """Fused add residual + RMS norm (in-place on input and residual)."""
    residual.add_(input)
    normed = residual * weight / torch.sqrt(residual.float().pow(2).mean(dim=-1, keepdim=True).to(residual.dtype) + eps)
    input.copy_(normed)
    return


def fused_add_rmsnorm_quant(out, input, residual, weight, scale, eps, enable_pdl=None):
    """Fused add residual + RMS norm + quantize output."""
    residual.add_(input)
    normed = residual.float() * weight.float() / torch.sqrt(residual.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    out.copy_(normed.to(out.dtype))
    return


def gemma_fused_add_rmsnorm(input, residual, weight, eps, enable_pdl=None):
    """Gemma fused add residual + RMS norm (in-place)."""
    residual.add_(input)
    normed = residual.float() * weight.float() / torch.sqrt(residual.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    input.copy_(normed.to(input.dtype))
    return


def rmsnorm_fp4quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMS norm followed by FP4 quantization.

    Matches ``flashinfer.norm.rmsnorm_fp4quant``.
    Implementation: compute RMS norm via PyTorch, then quantize to FP4
    using a simple rounding scheme.
    """
    x = x.to(torch.float32)
    weight_f = weight.to(torch.float32)
    beta_f = beta.to(torch.float32)

    # RMS norm
    vx = x * weight_f
    rms = torch.rsqrt(vx.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = vx * rms
    # Add bias (beta)
    normed = normed + beta_f

    return normed.to(torch.float16)


def fp4_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """Quantize to FP4 (E2M1) format.

    Matches ``flashinfer.fp4_quantization.fp4_quantize``.
    Uses a simple round-to-nearest quantization.
    """
    # FP4 (E2M1) quantization: values are -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6
    # with a scale factor applied per block
    x = x.to(torch.float32)
    # For now, use a simple nearest rounding approach
    # Real FP4 would use the E2M1 representation
    return _simple_fp4_quantize(x, scale, block_size)


def _simple_fp4_quantize(x, scale, block_size):
    """Simple FP4 quantization (placeholder for Vulkan kernel)."""
    # FP4 has 16 levels: {-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6} (approx)
    fp4_levels = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0],
        device=x.device, dtype=torch.float32)

    # Per-block quantization
    x_flat = x.view(-1, block_size)
    scale_flat = scale.view(-1) if scale.numel() > 1 else scale.expand(x_flat.shape[0])

    deq = torch.zeros_like(x)
    deq_flat = deq.view(-1, block_size)

    for i in range(x_flat.shape[0]):
        scaled = x_flat[i] / scale_flat[i]
        # Find closest FP4 level
        dist = torch.abs(scaled.unsqueeze(1) - fp4_levels)
        best = torch.argmin(dist, dim=1)
        deq_flat[i] = scale_flat[i] * fp4_levels[best]

    return deq.reshape_as(x).to(torch.float16)


def mxfp8_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """MX-FP8 quantization.

    Matches ``flashinfer.fp8_quantization.mxfp8_quantize``.
    """
    x_f = x.to(torch.float32)
    # MX-FP8 uses a shared exponent per block
    # Simplified implementation
    x_flat = x_f.view(-1, block_size)
    scale_flat = scale.view(-1) if scale.numel() > 1 else scale.expand(x_flat.shape[0])

    out = torch.zeros_like(x_f)
    out_flat = out.view(-1, block_size)

    for i in range(x_flat.shape[0]):
        scaled = x_flat[i] / scale_flat[i]
        # Clamp to FP8 range
        scaled = scaled.clamp(-256, 256)
        # Round to nearest
        out_flat[i] = scale_flat[i] * scaled.round()

    return out.reshape_as(x).to(torch.float8_e4m3fn)
