# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Vulkan compute backend for TRT-LLM _torch path on AMD RDNA (and other
Vulkan-capable GPUs).

This package provides drop-in replacements for the CUDA-dependent
flashinfer / Triton operations by dispatching compute through Vulkan
compute shaders.  Simple operations are implemented with PyTorch/ROCm
built-ins (tensor stays on GPU); complex kernels are dispatched to
pre-compiled SPIR-V shaders via the shared-library bridge.

Modules:
  - vulkan_compute: ctypes bridge to libvulkan_backend shared library
  - sampling      : flashinfer.sampling API replacements
  - attention     : attention wrapper classes
  - kv_cache      : paged KV cache management
  - moe           : fused MoE replacement
  - norm          : RMS norm + fp4 quantization
  - page          : KV cache page management helpers
  - mla           : MLA (multi-layer attention) wrappers

The module is activated by setting the environment variable
``TLLM_VULKAN_BACKEND=1`` on import.
"""

import os

_TLLM_VULKAN_BACKEND_AVAILABLE = True

_VK_BACKEND_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib", "libvulkan_backend.so"),
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib", "vulkan_backend.dll"),
    "libvulkan_backend.so",
    "vulkan_backend.dll",
]

def _find_vulkan_lib():
    import ctypes
    for path in _VK_BACKEND_PATHS:
        full = os.path.normpath(os.path.join(os.path.dirname(__file__), path))
        if os.path.isfile(full):
            return ctypes.CDLL(full)
    # Try loading from PATH / default
    try:
        return ctypes.CDLL("vulkan_backend")
    except OSError:
        return None

_vulkan_lib = None

def get_vulkan_lib():
    global _vulkan_lib
    if _vulkan_lib is None:
        _vulkan_lib = _find_vulkan_lib()
    return _vulkan_lib

def is_available():
    return get_vulkan_lib() is not None
