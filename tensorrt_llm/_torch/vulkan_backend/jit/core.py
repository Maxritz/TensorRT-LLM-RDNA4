# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""flashinfer.jit.core replacement (Vulkan/ROCm backend)."""


def check_cuda_arch():
    """No-op on Vulkan/ROCm — arch checking is not needed."""
    pass
