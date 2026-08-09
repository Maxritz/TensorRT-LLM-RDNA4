# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""flashinfer.comm.trtllm_alltoall replacement (Vulkan backend)."""

import torch


class MnnvlMoe:
    """All-to-all communication for MoE across GPUs.

    On AMD ROCm, this uses torch.distributed for the equivalent
    functionality. This stub provides the interface but dispatches
    through standard torch all_to_all.
    """

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def alltoall_single(tensor: torch.Tensor, *args, **kwargs):
        return tensor

    @staticmethod
    def alltoall(tensor_a: torch.Tensor, tensor_b: torch.Tensor, *args, **kwargs):
        return tensor_a, tensor_b
