# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""flashinfer.comm.mnnvl replacement (Vulkan backend)."""


class MnnvlMemory:
    """MNNVL (Multi-Node NVLink) memory handle.

    On AMD RDNA4 / Vulkan there is no MNNVL equivalent.
    This stub returns standard device memory handles.
    """

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def register_buffer(tensor, group=None, *args, **kwargs):
        return tensor

    @staticmethod
    def get_buffer(tensor, *args, **kwargs):
        return tensor

    @staticmethod
    def put_buffer(tensor, *args, **kwargs):
        return tensor
