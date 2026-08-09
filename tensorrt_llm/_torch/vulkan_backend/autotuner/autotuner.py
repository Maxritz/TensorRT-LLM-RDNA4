# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""flashinfer.autotuner.autotuner replacement (Vulkan backend)."""


class DynamicTensorSpec:
    """Stub for flashinfer's autotuner.DynamicTensorSpec.

    Used by the trtllm-gen attention backend for kernel autotuning.
    On Vulkan/ROCm this is a no-op placeholder.
    """

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_tensor(cls, tensor, *args, **kwargs):
        return cls()

    def __call__(self, *args, **kwargs):
        return self
