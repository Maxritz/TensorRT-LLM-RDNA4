# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stub for tensorrt_llm.bindings.internal.processgroup."""

import os


def init_pg(tp_rank, tp_size, pp_rank, pp_size, world_size, world_rank, *args, **kwargs):
    """Initialize a process group (stub on single-process Windows)."""
    return 0


def shutdown_pg(*args, **kwargs):
    """Shutdown a process group (stub)."""
    pass


__all__ = ["init_pg", "shutdown_pg"]
