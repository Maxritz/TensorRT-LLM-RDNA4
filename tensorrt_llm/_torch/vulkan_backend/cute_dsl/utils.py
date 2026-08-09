# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""flashinfer.cute_dsl.utils replacement (Vulkan backend)."""

import torch


def convert_sf_to_mma_layout(tensor: torch.Tensor, *args, **kwargs):
    """Convert a tensor's scale factor to MMA layout.

    On Vulkan this is a no-op — the scale factor is used as-is.
    """
    return tensor
