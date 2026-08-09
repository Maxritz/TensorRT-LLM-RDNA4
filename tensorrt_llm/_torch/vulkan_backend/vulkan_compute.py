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
Vulkan compute C-API bridge (ctypes wrapper).

Loads the ``vulkan_backend`` shared library and wraps each exported
``extern "C"`` function as a Python callable with proper ctypes signatures.
"""

import ctypes
import os
import sys

def _load_lib():
    names = []
    if sys.platform == "win32":
        names = ["vulkan_backend.dll", "libvulkan_backend.dll"]
    else:
        names = ["libvulkan_backend.so", "vulkan_backend.so"]

    # Search paths
    here = os.path.dirname(__file__)
    search_paths = [
        here,
        os.path.join(here, "..", "..", "..", "..", "lib"),
        os.path.join(here, "..", "..", "..", "..", "build", "lib"),
        os.environ.get("TLLM_VULKAN_LIB_DIR", ""),
    ]

    for d in search_paths:
        for n in names:
            p = os.path.normpath(os.path.join(d, n))
            if os.path.isfile(p):
                return ctypes.CDLL(p)

    # Last resort: try loading from PATH / ldconfig
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            pass

    return None


_lib = _load_lib()


class _VulkanCompute:
    """Thin ctypes wrapper around the Vulkan backend C API."""

    _initialized = False

    def __init__(self):
        self._init_funcs()

    @classmethod
    def is_available(cls):
        return _lib is not None

    def _init_funcs(self):
        lib = _lib
        if lib is None:
            return

        # tllm_vulkan_init(uint32_t gpu_id) -> int32
        lib.tllm_vulkan_init.argtypes = [ctypes.c_uint32]
        lib.tllm_vulkan_init.restype = ctypes.c_int32

        # tllm_vulkan_is_active() -> int32
        lib.tllm_vulkan_is_active.argtypes = []
        lib.tllm_vulkan_is_active.restype = ctypes.c_int32

        # tllm_vulkan_device_synchronize() -> void
        lib.tllm_vulkan_device_synchronize.argtypes = []
        lib.tllm_vulkan_device_synchronize.restype = None

        # tllm_vulkan_malloc(void**, size_t)
        lib.tllm_vulkan_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.tllm_vulkan_malloc.restype = None

        # tllm_vulkan_free(void*)
        lib.tllm_vulkan_free.argtypes = [ctypes.c_void_p]
        lib.tllm_vulkan_free.restype = None

        # tllm_vulkan_memcpy_h2d(void*, const void*, size_t)
        lib.tllm_vulkan_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.tllm_vulkan_memcpy_h2d.restype = None

        # tllm_vulkan_memcpy_d2h(void*, const void*, size_t)
        lib.tllm_vulkan_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.tllm_vulkan_memcpy_d2h.restype = None

        # tllm_vulkan_softmax(...)
        lib.tllm_vulkan_softmax.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,   # input_ptr, output_ptr
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ]
        lib.tllm_vulkan_softmax.restype = ctypes.c_int32

        # tllm_vulkan_gemm(void* a, void* b, void* output, uint32_t M, uint32_t N, uint32_t K)
        lib.tllm_vulkan_gemm.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ]
        lib.tllm_vulkan_gemm.restype = ctypes.c_int32

        # tllm_vulkan_rms_norm(void* input, void* gamma, void* beta, void* output,
        #                      float eps, size_t hiddenDim, size_t tokenCount)
        lib.tllm_vulkan_rms_norm.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_size_t, ctypes.c_size_t,
        ]
        lib.tllm_vulkan_rms_norm.restype = ctypes.c_int32

        # tllm_vulkan_elementwise_add(void* a, void* b, void* output, size_t elementCount)
        lib.tllm_vulkan_elementwise_add.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ]
        lib.tllm_vulkan_elementwise_add.restype = ctypes.c_int32

        # tllm_vulkan_silu(void* input, void* output, size_t elementCount)
        lib.tllm_vulkan_silu.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ]
        lib.tllm_vulkan_silu.restype = ctypes.c_int32

        # tllm_vulkan_gelu(void* input, void* output, size_t elementCount)
        lib.tllm_vulkan_gelu.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ]
        lib.tllm_vulkan_gelu.restype = ctypes.c_int32

        # tllm_vulkan_swiglu(void* input, void* output, uint32_t hiddenDim, uint32_t tokenCount)
        if hasattr(lib, "tllm_vulkan_swiglu"):
            lib.tllm_vulkan_swiglu.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_uint32, ctypes.c_uint32,
            ]
            lib.tllm_vulkan_swiglu.restype = ctypes.c_int32

        # tllm_vulkan_attention(void* q, void* k, void* v, void* output,
        #                       uint32_t batchSize, uint32_t numHeads,
        #                       uint32_t seqLenQ, uint32_t seqLenK, uint32_t headDim,
        #                       uint32_t causal)
        lib.tllm_vulkan_attention.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ]
        lib.tllm_vulkan_attention.restype = ctypes.c_int32

        # tllm_vulkan_topk(void* scores, void* inputOffsets, void* outputOffsets,
        #                  void* topkIndices, uint32_t topk, uint32_t numHeads,
        #                  uint32_t batchSize, uint32_t totalTokens,
        #                  uint32_t totalOutputTokens)
        lib.tllm_vulkan_topk.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
        ]
        lib.tllm_vulkan_topk.restype = ctypes.c_int32

        # tllm_vulkan_q8_0_gemm(void* weight, void* activation, void* output,
        #                       uint32_t M, uint32_t N, uint32_t K, uint32_t blocksPerRow)
        lib.tllm_vulkan_q8_0_gemm.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ]
        lib.tllm_vulkan_q8_0_gemm.restype = ctypes.c_int32

        # tllm_vulkan_kv_cache_update_2d(...)
        lib.tllm_vulkan_kv_cache_update_2d.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,   # kv_cache_k, kv_cache_v
            ctypes.c_void_p, ctypes.c_void_p,   # accepted_indices, num_accepted_tokens
            ctypes.c_void_p, ctypes.c_void_p,   # past_key_value_lens, rewind_adjustments
            ctypes.c_void_p,                     # seq_slot_remapping
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32,
        ]
        lib.tllm_vulkan_kv_cache_update_2d.restype = ctypes.c_int32

        # tllm_vulkan_tree_spec_build(...)
        lib.tllm_vulkan_tree_spec_build.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ]
        lib.tllm_vulkan_tree_spec_build.restype = ctypes.c_int32

        # tllm_vulkan_tree_spec_rejection(...)
        lib.tllm_vulkan_tree_spec_rejection.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ]
        lib.tllm_vulkan_tree_spec_rejection.restype = ctypes.c_int32

        # tllm_vulkan_spec_accept(...)
        lib.tllm_vulkan_spec_accept.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_float, ctypes.c_float,
        ]
        lib.tllm_vulkan_spec_accept.restype = ctypes.c_int32

        # tllm_vulkan_append_paged_kv_cache(...)
        if hasattr(lib, "tllm_vulkan_append_paged_kv_cache"):
            lib.tllm_vulkan_append_paged_kv_cache.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ]
            lib.tllm_vulkan_append_paged_kv_cache.restype = ctypes.c_int32

        self._funcs = lib

    def init(self, gpu_id=0):
        if not self.is_available():
            return False
        return bool(self._funcs.tllm_vulkan_init(gpu_id))

    def is_active(self):
        if not self.is_available():
            return False
        return bool(self._funcs.tllm_vulkan_is_active())

    def device_synchronize(self):
        if self.is_available():
            self._funcs.tllm_vulkan_device_synchronize()

    def softmax(self, input_ptr, output_ptr, batch, heads, seqlen, dtype):
        return self._funcs.tllm_vulkan_softmax(input_ptr, output_ptr, batch, heads, seqlen, dtype)

    def gemm(self, a, b, output, M, N, K):
        return self._funcs.tllm_vulkan_gemm(a, b, output, M, N, K)

    def rms_norm(self, input, gamma, beta, output, eps, hidden_dim, token_count):
        return self._funcs.tllm_vulkan_rms_norm(input, gamma, beta, output, eps, hidden_dim, token_count)

    def elementwise_add(self, a, b, output, element_count):
        return self._funcs.tllm_vulkan_elementwise_add(a, b, output, element_count)

    def silu(self, input_ptr, output_ptr, element_count):
        return self._funcs.tllm_vulkan_silu(input_ptr, output_ptr, element_count)

    def gelu(self, input_ptr, output_ptr, element_count):
        return self._funcs.tllm_vulkan_gelu(input_ptr, output_ptr, element_count)

    def swiglu(self, input_ptr, output_ptr, hidden_dim, token_count):
        return self._funcs.tllm_vulkan_swiglu(input_ptr, output_ptr, hidden_dim, token_count)

    def attention(self, q, k, v, output, batch_size, num_heads, seq_len_q, seq_len_k, head_dim, causal):
        return self._funcs.tllm_vulkan_attention(q, k, v, output,
            batch_size, num_heads, seq_len_q, seq_len_k, head_dim, causal)
    def topk(self, scores, input_offsets, output_offsets, topk_indices,
             topk, num_heads, batch_size, total_tokens, total_output_tokens):
        return self._funcs.tllm_vulkan_topk(scores, input_offsets, output_offsets, topk_indices,
            topk, num_heads, batch_size, total_tokens, total_output_tokens)

    def q8_0_gemm(self, weight, activation, output, M, N, K, blocks_per_row):
        return self._funcs.tllm_vulkan_q8_0_gemm(weight, activation, output,
            M, N, K, blocks_per_row)

    def kv_cache_update_2d(self, *args):
        return self._funcs.tllm_vulkan_kv_cache_update_2d(*args)

    def tree_spec_build(self, *args):
        return self._funcs.tllm_vulkan_tree_spec_build(*args)

    def tree_spec_rejection(self, *args):
        return self._funcs.tllm_vulkan_tree_spec_rejection(*args)

    def spec_accept(self, *args):
        return self._funcs.tllm_vulkan_spec_accept(*args)

    def append_paged_kv_cache(self, *args):
        return self._funcs.tllm_vulkan_append_paged_kv_cache(*args)

    def malloc(self, n):
        ptr = ctypes.c_void_p()
        self._funcs.tllm_vulkan_malloc(ctypes.byref(ptr), n)
        return ptr

    def free(self, ptr):
        self._funcs.tllm_vulkan_free(ptr)

    def memcpy_h2d(self, dst, src, n):
        self._funcs.tllm_vulkan_memcpy_h2d(dst, src, n)

    def memcpy_d2h(self, dst, src, n):
        self._funcs.tllm_vulkan_memcpy_d2h(dst, src, n)


_vk = _VulkanCompute()

def init(gpu_id=0):
    return _vk.init(gpu_id)

def is_available():
    return _VulkanCompute.is_available()

def is_active():
    return _vk.is_active()

def device_synchronize():
    _vk.device_synchronize()

# Re-export
tllm_vulkan_init = _vk.init if is_available() else None
tllm_vulkan_softmax = _vk.softmax if is_available() else None
tllm_vulkan_gemm = _vk.gemm if is_available() else None
tllm_vulkan_rms_norm = _vk.rms_norm if is_available() else None
tllm_vulkan_elementwise_add = _vk.elementwise_add if is_available() else None
tllm_vulkan_silu = _vk.silu if is_available() else None
tllm_vulkan_gelu = _vk.gelu if is_available() else None
tllm_vulkan_swiglu = _vk.swiglu if is_available() else None
tllm_vulkan_attention = _vk.attention if is_available() else None
tllm_vulkan_topk = _vk.topk if is_available() else None
tllm_vulkan_q8_0_gemm = _vk.q8_0_gemm if is_available() else None
tllm_vulkan_kv_cache_update_2d = _vk.kv_cache_update_2d if is_available() else None
tllm_vulkan_tree_spec_build = _vk.tree_spec_build if is_available() else None
tllm_vulkan_tree_spec_rejection = _vk.tree_spec_rejection if is_available() else None
tllm_vulkan_spec_accept = _vk.spec_accept if is_available() else None
tllm_vulkan_append_paged_kv_cache = _vk.append_paged_kv_cache if is_available() else None
