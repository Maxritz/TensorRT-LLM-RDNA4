# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stub for tensorrt_llm.bindings.internal.thop."""

from enum import IntEnum


class BufferKind(IntEnum):
    INPUT = 0
    OUTPUT = 1
    INTERMEDIATE = 2
    PARAM = 3
    DEFAULT = 0
    USERBUFFERS = 1
    NCCL_WINDOW = 2


class Buffer:
    pass


class TensorOp:
    pass


__all__ = ["BufferKind", "Buffer", "TensorOp"]
