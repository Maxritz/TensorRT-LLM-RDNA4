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
    # Preserve dtype: uint32/int32 buffers (topk offsets/indices) must not be
    # coerced to float32.
    a = np.ascontiguousarray(arr)
    ptr = vk.malloc(a.nbytes)
    vk.memcpy_h2d(ptr, a.ctypes.data, a.nbytes)
    return ptr


def _d2h(vk, dev_ptr, nbytes, shape=None, dtype=FTY):
    out = np.empty(nbytes // 4, dtype=dtype)
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


def _pack_q8_0(W_float):
    """Pack [N, K] fp32 weights (K % 32 == 0) to GGML block_q8_0 bytes and
    reconstruct the dequantized float32 matrix from the SAME bytes so the CPU
    reference and the GPU kernel use identical scales/quantization."""
    N, K = W_float.shape
    B = K // 32
    raw = np.empty((N, B, 36), dtype=np.uint8)
    W_dq = np.zeros((N, K), dtype=np.float64)
    for n in range(N):
        for b in range(B):
            block = W_float[n, b * 32:(b + 1) * 32].astype(np.float64)
            amax = float(np.max(np.abs(block))) if block.size else 0.0
            scale = (amax / 127.0) if amax > 0.0 else 1.0
            q = np.round(block / scale).astype(np.int8)
            raw[n, b, 0:4] = np.frombuffer(np.float32(scale).tobytes(), dtype=np.uint8)
            raw[n, b, 4:36] = np.frombuffer(q.tobytes(), dtype=np.uint8)
            W_dq[n, b * 32:(b + 1) * 32] = q.astype(np.float64) * scale
    return raw.reshape(-1), W_dq


# ---------------------------------------------------------------------------
# Q8_0 (block-quantized) GEMM: C[M,N] = sum_k A[m,k] * (int8_w * scale).
# ---------------------------------------------------------------------------
def test_q8_0_gemm_dequant_matches_numpy(vk):
    M, N, K = 4, 6, 32  # one q8_0 block per weight row -> blocksPerRow = 1
    A = np.random.randn(M, K).astype(FTY)
    W = (np.random.randn(N, K) * 2.0).astype(FTY)
    weight_bytes, W_dq = _pack_q8_0(W)
    C_ref = (A.astype(np.float64) @ W_dq.T).astype(FTY)
    pA, pW, pC = _h2d(vk, A), _h2d(vk, weight_bytes), vk.malloc(C_ref.nbytes)
    try:
        assert vk.q8_0_gemm(pW, pA, pC, M, N, K, K // 32), "q8_0_gemm launch failed"
        C = _d2h(vk, pC, C_ref.nbytes, C_ref.shape)
    finally:
        _free(vk, pA, pW, pC)
    assert np.allclose(C, C_ref, rtol=1e-2, atol=1e-2), np.max(np.abs(C - C_ref))


# ---------------------------------------------------------------------------
# Attention: O = softmax((Q K^T)/sqrt(headDim)) V, optional causal mask.
# Q/K/V/O fp32 laid out [batch, numHeads, seq, headDim] (row-major).
# ---------------------------------------------------------------------------
def test_attention_fp32_matches_numpy(vk):
    B, nh, sq, sk, hd, causal = 2, 3, 4, 5, 8, True
    Q = np.random.randn(B, nh, sq, hd).astype(FTY)
    K = np.random.randn(B, nh, sk, hd).astype(FTY)
    V = np.random.randn(B, nh, sk, hd).astype(FTY)

    iscore = 1.0 / np.sqrt(hd)
    scores = np.einsum("bnid,bnjd->bnij", Q, K).astype(np.float64) * iscore
    if causal:
        keep = np.tri(sq, sk, k=0, dtype=bool)  # True where j <= i
        scores = np.where(keep[None, None, :, :], scores, -1e9)
    scores = scores - scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p = p / p.sum(axis=-1, keepdims=True)
    O_ref = np.einsum("bnij,bnjd->bnid", p, V).astype(FTY)

    pQ, pK, pV = _h2d(vk, Q), _h2d(vk, K), _h2d(vk, V)
    pO = vk.malloc(O_ref.nbytes)
    try:
        assert vk.attention(pQ, pK, pV, pO, B, nh, sq, sk, hd, int(causal))
        O = _d2h(vk, pO, O_ref.nbytes, O_ref.shape)
    finally:
        _free(vk, pQ, pK, pV, pO)
    assert np.max(np.abs(O - O_ref)) <= 1e-3


# ---------------------------------------------------------------------------
# Top-K (sparse attention token selection). For each (batch, head) row we
# emit `topk` row-local token offsets of the largest scores, descending,
# first-max wins ties (strict >). Exact integer comparison.
# ---------------------------------------------------------------------------
def test_topk_matches_reference(vk):
    num_heads, batch, total_tokens, topk = 1, 2, 8, 3
    total_out = 6
    # single head, one row of 8 distinct scores
    scores = np.array([10, 30, 20, 50, 40, 70, 60, 80], dtype=FTY)
    # request 0 spans scores[0:5], request 1 spans scores[5:8]
    in_off = np.array([0, 5, 8], dtype=np.uint32)
    out_off = np.array([0, 3, 6], dtype=np.uint32)
    # Expected row-local argmax-by-value (descending), first-max wins:
    #   req0 top-3 of [10,30,20,50,40] -> 50(i3),40(i4),30(i1) -> [3,4,1]
    #   req1 top-3 of [70,60,80]        -> 80(i2),70(i0),60(i1) -> [2,0,1]
    expected = np.array([3, 4, 1, 2, 0, 1], dtype=np.int32)

    p_scores = _h2d(vk, scores)
    p_inoff = _h2d(vk, in_off)
    p_outoff = _h2d(vk, out_off)
    p_topk = vk.malloc(expected.nbytes)
    try:
        assert vk.topk(p_scores, p_inoff, p_outoff, p_topk,
                       topk, num_heads, batch, total_tokens, total_out)
        got = _d2h(vk, p_topk, expected.nbytes, expected.shape, dtype=np.int32)
    finally:
        _free(vk, p_scores, p_inoff, p_outoff, p_topk)
    assert np.array_equal(got, expected), f"topk mismatch: {got} != {expected}"
