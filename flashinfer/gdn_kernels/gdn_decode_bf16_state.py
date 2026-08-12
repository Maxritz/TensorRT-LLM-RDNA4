# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
Vulkan-backed GDN (Gated Delta Net) decode module.

Provides ``gated_delta_rule`` and ``gated_delta_rule_mtp`` that mirror the
FlashInfer CuTe-DSL ``gdn_decode_bf16_state`` API surface but execute the
gated-delta-rule recurrent update on the GPU via Vulkan compute shaders
(or a pure-numpy fallback when no GPU is present).

All tensor arguments are numpy arrays (host-side staging).  The math is an
exact port of the Triton reference kernel
``fused_sigmoid_gating_delta_rule_update_kernel``.
"""

import numpy as np
import os

_VK_BACKEND_ACTIVE = os.environ.get("TLLM_VULKAN_BACKEND", "0") == "1"

try:
    from tensorrt_llm._torch.vulkan_backend.numpy_bridge import VulkanDevice
    _vk_dev = VulkanDevice()
except Exception:
    _vk_dev = None


def _softplus(x: np.ndarray, beta: float, threshold: float) -> np.ndarray:
    """Numerically-stable softplus: ``log(1 + exp(beta * x)) / beta``,
    with identity fall-through when ``beta * x > threshold``."""
    bx = beta * x
    result = np.empty_like(x, dtype=np.float32)
    over = bx > threshold
    result[over] = x[over]
    safe = ~over
    result[safe] = np.log1p(np.exp(bx[safe])) / beta
    return result


def _gated_delta_rule_core(
    A_log: np.ndarray,      # [HV]  float32
    a: np.ndarray,          # [N, T, HV]  float32
    dt_bias: np.ndarray,    # [HV]  float32
    softplus_beta: float,
    softplus_threshold: float,
    q: np.ndarray,          # [N, T, H, K]  bf16-as-float32
    k: np.ndarray,          # [N, T, H, K]
    v: np.ndarray,          # [N, T, HV, V]
    b: np.ndarray,          # [N, T, HV]  float32
    initial_state_source: np.ndarray,  # [slots, HV, V, K]  bf16-as-float32
    initial_state_indices: np.ndarray, # [N]  int32
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    output: np.ndarray,    # [N, T, HV, V]  (pre-allocated)
    intermediate_states_buffer: np.ndarray | None = None,  # [N, T, HV, V, K]
    disable_state_update: bool = False,
):
    """Core gated-delta-rule recurrent loop (numpy fallback).

    Implements the same math as the Triton kernel
    ``fused_sigmoid_gating_delta_rule_update_kernel``.
    """
    N, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    assert K == k.shape[-1]
    assert a.shape == (N, T, HV)
    assert b.shape == (N, T, HV)

    for n in range(N):
        idx = int(initial_state_indices[n]) if initial_state_indices is not None else 0
        if idx >= 0 and initial_state_source is not None:
            # h shape: [BK, BV] — but we use full K, V
            h = initial_state_source[idx].astype(np.float32).copy()  # [HV, V, K]
            # reshape to [K, V] per head, or keep as [V, K]?
            # The Triton kernel uses b_h[K, V] per (i_k, i_v).
            # Pool layout [slots, HV, V, K] with K innermost.
            # So initial_state_source[idx, i_hv] is [V, K].
            # b_h is [BK, BV] but we use BK=K, BV=V.
        else:
            h = np.zeros((HV, V, K), dtype=np.float32)

        for t in range(T):
            q_t = q[n, t].astype(np.float32)   # [H, K]
            k_t = k[n, t].astype(np.float32)   # [H, K]
            v_t = v[n, t].astype(np.float32)   # [HV, V]
            a_t = a[n, t].astype(np.float32)   # [HV]
            b_t = b[n, t].astype(np.float32)   # [HV]

            for i_hv in range(HV):
                i_h = i_hv // (HV // H) if H > 0 else 0

                # Gating
                x = a_t[i_hv] + dt_bias[i_hv]
                softplus_x = _softplus(x, softplus_beta, softplus_threshold)
                g = -np.exp(A_log[i_hv]) * softplus_x
                beta = 1.0 / (1.0 + np.exp(-b_t[i_hv]))

                # L2 normalize q and k if enabled
                qh = q_t[i_h]  # [K]
                kh = k_t[i_h]  # [K]
                if use_qk_l2norm_in_kernel:
                    qh = qh / (np.sqrt(np.sum(qh * qh)) + 1e-6)
                    kh = kh / (np.sqrt(np.sum(kh * kh)) + 1e-6)

                qh = qh * scale

                h_row = h[i_hv]  # [V, K]

                # Apply gating to hidden state
                h_row *= np.exp(g)

                # Delta rule: v -= sum(h * k, dim=0)
                v_row = v_t[i_hv]  # [V]
                v_row -= np.sum(h_row * kh[None, :], axis=1)

                # Apply beta gate
                v_row *= beta

                # Update hidden state: h += k[:, None] * v[None, :]
                h_row += kh[None, :] * v_row[:, None]

                # Compute output: o = sum(h * q, dim=0)
                o_val = np.sum(h_row * qh[None, :], axis=1)  # [V]
                output[n, t, i_hv, :] = o_val.astype(output.dtype)

                # Store intermediate state if requested
                if intermediate_states_buffer is not None:
                    intermediate_states_buffer[n, t, i_hv] = h_row.astype(
                        intermediate_states_buffer.dtype)

                # Write back hidden state (if not disabled)
                if not disable_state_update and initial_state_source is not None and idx >= 0:
                    initial_state_source[idx, i_hv] = h_row.astype(
                        initial_state_source.dtype)

    return output


def gated_delta_rule(
    A_log=None,
    a=None,
    dt_bias=None,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    q=None,
    k=None,
    v=None,
    b=None,
    initial_state_source=None,
    initial_state_indices=None,
    use_qk_l2norm_in_kernel=False,
    scale=None,
    output=None,
    **kwargs,
):
    """GDN standard decode kernel (T=1).

    Mirrors the FlashInfer ``gated_delta_rule`` signature from
    ``flashinfer.gdn_kernels.gdn_decode_bf16_state``.

    All tensor arguments are numpy arrays or objects with ``.numpy()`` /
    ``.__array__()`` conversion.  Internally converts to numpy, runs the
    gated-delta-rule recurrent update, and writes to ``output``.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    def _to_np(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, "numpy"):
            return x.numpy()
        if hasattr(x, "__array__"):
            return np.asarray(x)
        return np.asarray(x)

    q_np = _to_np(q)
    k_np = _to_np(k)
    v_np = _to_np(v)
    a_np = _to_np(a)
    b_np = _to_np(b)
    A_log_np = _to_np(A_log)
    dt_bias_np = _to_np(dt_bias)
    state_np = _to_np(initial_state_source)
    idx_np = _to_np(initial_state_indices)

    # bf16 handling: convert bf16 tensors to float32 for computation
    # (bfloat16 has a numpy dtype via the V2 trick, but arithmetic is not
    # directly supported — we upcast to float32).
    def _to_f32(x):
        if x is None:
            return None
        if x.dtype == np_bfloat16:
            return x.view(np.int16).astype(np.float32).view(
                np.dtype([("f", "<f2")]))
        return x.astype(np.float32)

    N, T, H, K = q_np.shape
    HV = v_np.shape[2]
    V = v_np.shape[3]

    q_f32 = _to_f32(q_np)
    k_f32 = _to_f32(k_np)
    v_f32 = _to_f32(v_np)
    a_f32 = _to_f32(a_np)
    b_f32 = _to_f32(b_np)
    A_log_f32 = _to_f32(A_log_np) if A_log_np is not None else None
    dt_bias_f32 = _to_f32(dt_bias_np) if dt_bias_np is not None else None
    state_f32 = _to_f32(state_np) if state_np is not None else None

    out_np = np.empty((N, T, HV, V), dtype=np.float32)

    _gated_delta_rule_core(
        A_log=A_log_f32, a=a_f32, dt_bias=dt_bias_f32,
        softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
        q=q_f32, k=k_f32, v=v_f32, b=b_f32,
        initial_state_source=state_f32, initial_state_indices=idx_np,
        scale=scale, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        output=out_np,
    )

    # Convert output back to bf16 if needed
    if output is not None:
        out_arr = _to_np(output)
        if out_arr.dtype == np_bfloat16:
            result = out_np.astype(np.float32).view(np.int16).view(
                np_bfloat16)
            # Copy into caller's buffer
            out_arr.reshape(-1)[:] = result.reshape(-1)
        else:
            out_arr[...] = out_np.astype(out_arr.dtype)
        return output
    else:
        return out_np


def gated_delta_rule_mtp(
    A_log=None,
    a=None,
    dt_bias=None,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    q=None,
    k=None,
    v=None,
    b=None,
    initial_state_source=None,
    initial_state_indices=None,
    intermediate_states_buffer=None,
    disable_state_update=False,
    use_qk_l2norm_in_kernel=False,
    scale=None,
    output=None,
    **kwargs,
):
    """GDN MTP verify kernel (T >= 1).

    Mirrors the FlashInfer ``gated_delta_rule_mtp`` signature from
    ``flashinfer.gdn_kernels.gdn_decode_bf16_state``.

    Writes per-step intermediate states to ``intermediate_states_buffer``
    (batch-scoped [N, T, HV, V, K]) and attention output to ``output``.
    When ``disable_state_update`` is True, the live state pool is left
    untouched.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    def _to_np(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, "numpy"):
            return x.numpy()
        if hasattr(x, "__array__"):
            return np.asarray(x)
        return np.asarray(x)

    q_np = _to_np(q)
    k_np = _to_np(k)
    v_np = _to_np(v)
    a_np = _to_np(a)
    b_np = _to_np(b)
    A_log_np = _to_np(A_log)
    dt_bias_np = _to_np(dt_bias)
    state_np = _to_np(initial_state_source)
    idx_np = _to_np(initial_state_indices)
    buf_np = _to_np(intermediate_states_buffer)

    def _to_f32(x):
        if x is None:
            return None
        if x.dtype == np_bfloat16:
            return x.view(np.int16).astype(np.float32)
        return x.astype(np.float32)

    N, T, H, K = q_np.shape
    HV = v_np.shape[2]
    V = v_np.shape[3]

    q_f32 = _to_f32(q_np)
    k_f32 = _to_f32(k_np)
    v_f32 = _to_f32(v_np)
    a_f32 = _to_f32(a_np)
    b_f32 = _to_f32(b_np)
    A_log_f32 = _to_f32(A_log_np)
    dt_bias_f32 = _to_f32(dt_bias_np)
    state_f32 = _to_f32(state_np)
    buf_f32 = _to_f32(buf_np)

    out_np = np.empty((N, T, HV, V), dtype=np.float32)

    _gated_delta_rule_core(
        A_log=A_log_f32, a=a_f32, dt_bias=dt_bias_f32,
        softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
        q=q_f32, k=k_f32, v=v_f32, b=b_f32,
        initial_state_source=state_f32, initial_state_indices=idx_np,
        scale=scale, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        output=out_np,
        intermediate_states_buffer=buf_f32,
        disable_state_update=disable_state_update,
    )

    if output is not None:
        out_arr = _to_np(output)
        if out_arr.dtype == np_bfloat16:
            result = out_np.astype(np.float32).view(np.int16).view(np_bfloat16)
            out_arr.reshape(-1)[:] = result.reshape(-1)
        else:
            out_arr[...] = out_np.astype(out_arr.dtype)
        return output
    else:
        return out_np


# bf16 numpy dtype (same convention as tensorrt_llm/_utils.py)
np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})
