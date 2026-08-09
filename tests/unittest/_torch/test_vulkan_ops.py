# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Hardware-backed tests for the Vulkan compute C-API.

These exercise the on-device GEMM, RMSNorm and elementwise-add kernels and
compare them against a NumPy reference. They are intentionally tolerant of
hosts that lack a staged ``libvulkan_backend`` build or a Vulkan GPU: when the
shared library cannot be located or loaded the module is skipped (rather than
failing CI). The ctypes bridge is loaded by file path so that collecting these
tests does not require importing the full ``tensorrt_llm`` package (which pulls
in CUDA-dependent extensions on some hosts).
"""

import importlib.util
import os

import numpy as np
import pytest

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VC_PATH = os.path.join(
    _REPO_ROOT, "tensorrt_llm", "_torch", "vulkan_backend", "vulkan_compute.py")

# Resolve + load the ctypes bridge by path (avoids importing the package).
if not os.path.isfile(_VC_PATH):
    pytest.skip(f"vulkan_compute.py not found at {_VC_PATH}", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("vk_compute_test_mod", _VC_PATH)
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)

if not vc.is_available():
    pytest.skip(
        "libvulkan_backend not staged; build the 'vulkan_backend' target from "
        "cpp/trace_test and place it alongside vulkan_compute.py",
        allow_module_level=True,
    )

_vk = vc._vk


@pytest.fixture(scope="module")
def vk():
    if not _vk.init(0):
        pytest.skip("tllm_vulkan_init failed (no Vulkan device?)")
    return _vk


FTY = np.float32


def _h2d(vk, arr):
    a = np.ascontiguousarray(arr, dtype=FTY)
    ptr = vk.malloc(a.nbytes)
    vk.memcpy_h2d(ptr, a.ctypes.data, a.nbytes)
    return ptr


def _d2h(vk, dev_ptr, nbytes, shape=None):
    out = np.empty(nbytes // 4, dtype=FTY)
    vk.memcpy_d2h(out.ctypes.data, dev_ptr, nbytes)
    return out.reshape(shape) if shape else out


def _free(vk, *ptrs):
    for p in ptrs:
        vk.free(p)


# ---------------------------------------------------------------------------
# GEMM: C[M, N] = A[M, K] * B[K, N] (fp32). launchFp16Gemm hard-codes aT=bT=false.
# ---------------------------------------------------------------------------
def test_gemm_fp32_matches_numpy(vk):
    M, N, K = 8, 16, 32
    A = np.random.randn(M, K).astype(FTY)
    B = np.random.randn(K, N).astype(FTY)
    C_ref = (A.astype(np.float64) @ B.astype(np.float64)).astype(FTY)
    pa, pb, pc = _h2d(vk, A), _h2d(vk, B), vk.malloc(C_ref.nbytes)
    try:
        assert vk.gemm(pa, pb, pc, M, N, K), "gemm launch failed"
        C = _d2h(vk, pc, C_ref.nbytes, C_ref.shape)
    finally:
        _free(vk, pa, pb, pc)
    assert np.max(np.abs(C - C_ref)) <= 1e-3


# ---------------------------------------------------------------------------
# RMSNorm: out = in * invRms * gamma + beta, invRms = 1/sqrt(mean(in^2)+eps).
# ---------------------------------------------------------------------------
def test_rms_norm_fp32_matches_numpy(vk):
    hidden, tokens, eps = 64, 4, 1e-5
    total = hidden * tokens
    x = np.random.randn(total).astype(FTY)
    g = np.random.randn(hidden).astype(FTY)
    beta = np.random.randn(hidden).astype(FTY)
    xrow = x.reshape(tokens, hidden)
    inv = 1.0 / np.sqrt(np.mean(xrow ** 2, axis=1, keepdims=True) + eps)
    ref = ((xrow * inv).astype(FTY) * g[None, :] + beta[None, :]).reshape(-1)
    px, pg, pb, po = (
        _h2d(vk, x), _h2d(vk, g), _h2d(vk, beta), vk.malloc(ref.nbytes))
    try:
        assert vk.rms_norm(px, pg, pb, po, eps, hidden, tokens), "rms_norm launch failed"
        out = _d2h(vk, po, ref.nbytes, ref.shape)
    finally:
        _free(vk, px, pg, pb, po)
    assert np.max(np.abs(out - ref)) <= 1e-3


# ---------------------------------------------------------------------------
# Elementwise add: out = a + b (fp32).
# ---------------------------------------------------------------------------
def test_elementwise_add_fp32_matches_python(vk):
    n = 4096
    a = np.random.randn(n).astype(FTY)
    b = np.random.randn(n).astype(FTY)
    ref = (a.astype(np.float64) + b.astype(np.float64)).astype(FTY)
    pa, pb, po = _h2d(vk, a), _h2d(vk, b), vk.malloc(ref.nbytes)
    try:
        assert vk.elementwise_add(pa, pb, po, n), "elementwise_add launch failed"
        out = _d2h(vk, po, ref.nbytes, ref.shape)
    finally:
        _free(vk, pa, pb, po)
    assert np.max(np.abs(out - ref)) <= 1e-4
