# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .runtime import *  # noqa: F401,F403


def __getattr__(name):
    from . import runtime as _rt
    return getattr(_rt, name)
