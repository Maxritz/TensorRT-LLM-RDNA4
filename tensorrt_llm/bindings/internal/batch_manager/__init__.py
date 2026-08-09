# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-export from the C++ extension or pure-Python stubs."""

from .batch_manager import *  # noqa: F401,F403


def __getattr__(name):
    """Delegate to the submodule's __getattr__ for names not in __all__."""
    from . import batch_manager as _bm
    return getattr(_bm, name)
