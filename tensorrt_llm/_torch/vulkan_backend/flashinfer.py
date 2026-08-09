# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vulkan-backed drop-in replacement for the ``flashinfer`` package.

This is a **module** (not a package) that re-exports all the sub-modules
from ``tensorrt_llm._torch.vulkan_backend``.  When it is registered as
``sys.modules["flashinfer"]`` the ``__path__`` attribute is set so that
``import flashinfer.sampling`` (etc.) resolves to the already-loaded
sub-modules rather than searching the file-system for a ``flashinfer``
package directory.
"""

import os
import sys
import types

import torch

_self = sys.modules[__name__]

# Re-export the sub-modules from the vulkan_backend package.
_pkg_dir = os.path.dirname(os.path.abspath(__file__))

# Make Python treat this module as a package so that
# ``import flashinfer.X`` resolves via sys.modules lookups.
_self.__path__ = [_pkg_dir]

from . import sampling
from . import activation
from . import attention as _attention
from . import page as _page
from . import kv_cache as _kv_cache
from . import norm as _norm
from . import fused_moe as _fused_moe
from . import fp4_quantization as _fp4_quantization
from . import fp8_quantization as _fp8_quantization
from . import mla as _mla
from . import rope
from . import mamba
from . import tllm_enums
from . import jit
from .jit import core as _jit_core
from . import autotuner
from . import comm
from .comm import mnnvl as _comm_mnnvl
from .comm import trtllm_alltoall as _comm_alltoall
from . import cute_dsl
from .cute_dsl import utils as _cute_utils
from . import gdn_kernels
from .gdn_kernels import gdn_decode_bf16_state as _gdn_decode
from .gdn_kernels import blackwell as _gdn_blackwell
from .gdn_kernels.blackwell import gated_delta_net_tile_scheduler as _gdn_tile_sched

# Register every sub-module under the ``flashinfer.*`` namespace so that
# ``import flashinfer.sampling`` works without a real package on disk.
for _name, _mod in [
    ("sampling", sampling),
    ("activation", activation),
    ("attention", _attention),
    ("page", _page),
    ("kv_cache", _kv_cache),
    ("norm", _norm),
    ("fused_moe", _fused_moe),
    ("fp4_quantization", _fp4_quantization),
    ("fp8_quantization", _fp8_quantization),
    ("mla", _mla),
    ("rope", rope),
    ("mamba", mamba),
    ("tllm_enums", tllm_enums),
    ("jit", jit),
    ("jit.core", _jit_core),
    ("autotuner", autotuner),
    ("comm", comm),
    ("comm.mnnvl", _comm_mnnvl),
    ("comm.trtllm_alltoall", _comm_alltoall),
    ("cute_dsl", cute_dsl),
    ("cute_dsl.utils", _cute_utils),
    ("gdn_kernels", gdn_kernels),
    ("gdn_kernels.gdn_decode_bf16_state", _gdn_decode),
    ("gdn_kernels.blackwell", _gdn_blackwell),
    ("gdn_kernels.blackwell.gated_delta_net_tile_scheduler", _gdn_tile_sched),
]:
    sys.modules[f"flashinfer.{_name}"] = _mod

# Re-export commonly used callables
from .sampling import get_sampling_module  # noqa: E402,F401
from .kv_cache import (  # noqa: E402,F401
    append_paged_kv_cache,
    get_batch_indices_positions,
    get_seq_lens,
)
from .attention import (  # noqa: E402,F401
    BatchPrefillWithPagedKVCacheWrapper,
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from .mla import BatchMLAPagedAttentionWrapper  # noqa: E402,F401
from .fused_moe import B12xMoEWrapper  # noqa: E402,F401

# Submodules accessible as attributes
page = _page
prefill = _attention
decode = _attention
ml = _mla
norm = _norm
fused_moe = _fused_moe
fp4_quantization = _fp4_quantization
fp8_quantization = _fp8_quantization

# Enums — flat module-level constants matching flashinfer usage
Default = 0
Renormalize = 1
DeepSeekV3 = 2
Llama4 = 3
RenormalizeNaive = 4
MiniMax2 = 5
SigmoidRenorm = 6
Unspecified = 7
Swiglu = 0
Relu2 = 1
Silu = 2

# Top-level B12xMoE alias
B12xMoE = B12xMoEWrapper

__version__ = "1.0.0-vulkan"


def __getattr__(name):
    """Fallback for any attribute not explicitly defined above."""
    try:
        return _self.__dict__[name]
    except KeyError:
        pass
    raise AttributeError(f"module 'flashinfer' has no attribute '{name}'")
