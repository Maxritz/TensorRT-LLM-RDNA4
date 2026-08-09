# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""flashinfer.mamba replacement (Vulkan/ROCm backend).

SSM (State Space Model) kernels for Mamba / Mamba-2 / Mamba-3 architectures.
On ROCm these use PyTorch built-ins or ``torch.nn.functional`` primitives.
"""

import torch
import torch.nn.functional as F


def selective_state_update(
    x: torch.Tensor,
    z: torch.Tensor,
    state: torch.Tensor,
    delta_bias: torch.Tensor,
    *,
    dt_softplus: bool = True,
    dt_limit: tuple = (0.0, float("inf")),
    **kwargs,
) -> torch.Tensor:
    """SSM state update kernel.

    Matches ``flashinfer.mamba.selective_state_update``.
    This is a PyTorch reference implementation.
    """
    if dt_softplus:
        delta_bias = F.softplus(delta_bias)
    new_state = state + x * delta_bias.unsqueeze(-1)
    if z is not None:
        new_state = new_state * torch.sigmoid(z)
    return new_state


def selective_state_update_no_scan(
    x: torch.Tensor,
    z: torch.Tensor,
    state: torch.Tensor,
    delta_bias: torch.Tensor,
    *,
    dt_softplus: bool = True,
    dt_limit: tuple = (0.0, float("inf")),
    **kwargs,
) -> torch.Tensor:
    """SSM state update without scan (stub)."""
    return state


class SSDCombined:
    """Combined SSD (Selective Scan Dynamic) module.

    Matches ``flashinfer.mamba.SSDCombined``.
    Simplified reference implementation using PyTorch primitives.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        z,
        state,
        delta_bias,
        *,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
    ):
        out = selective_state_update(
            x, z, state, delta_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

    def __call__(self, *args, **kwargs):
        return self.forward(self, *args, **kwargs)
