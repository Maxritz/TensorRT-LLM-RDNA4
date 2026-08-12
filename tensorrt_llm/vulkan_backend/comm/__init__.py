# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""flashinfer.comm submodule replacement (Vulkan backend).

MNNVL and all-to-all communication are no-ops on single-GPU AMD RDNA4.
For multi-GPU, ROCM's RCCL provides equivalent collectives.
"""
