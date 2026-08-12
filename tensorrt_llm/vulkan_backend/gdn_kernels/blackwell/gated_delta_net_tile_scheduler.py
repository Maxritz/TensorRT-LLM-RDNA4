# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""flashinfer.gdn_kernels.blackwell.gated_delta_net_tile_scheduler replacement (Vulkan backend)."""


class GatedDeltaNetTileScheduler:
    """Tile scheduler for GDN (Gated Delta Network) attention.

    On Vulkan/ROCm this is a no-op stub — the Vulkan compute shaders
    handle tile scheduling internally.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None

    @staticmethod
    def make_tile_scheduler(*args, **kwargs):
        return GatedDeltaNetTileScheduler()
