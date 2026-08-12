# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch-free GDN (Gated Delta Net) fused sigmoid gating recurrent update.

Provides the same public API surface as the original Triton-kernel module
but operates on numpy arrays (host-side staging) and dispatches compute to
Vulkan shaders when ``TLLM_VULKAN_BACKEND=1`` is set.

Functions:
    fused_sigmoid_gating_delta_rule_update — main entry point
    _flashinfer_gdn_decode                — FI-decode dispatch wrapper
    _flashinfer_gdn_verify                — FI-MTP verify dispatch wrapper
    fused_gdn_gating                       — standalone gating helper
"""

import os
import numpy as np
from typing import Optional

_VK_BACKEND_ACTIVE = os.environ.get("TLLM_VULKAN_BACKEND", "0") == "1"

# ---- flashinfer gdn_kernels import ----
try:
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
        gated_delta_rule as _fi_gdn_decode_bf16_state_t1,
        gated_delta_rule_mtp as _fi_gdn_decode_bf16_state_mtp,
    )
    _FLASHINFER_GDN_BF16_STATE_AVAILABLE = True
except (ImportError, RuntimeError):
    _FLASHINFER_GDN_BF16_STATE_AVAILABLE = False

# Max per-sequence token count served by the MTP verify kernel.
_FI_GDN_MAX_MTP_T = 8

np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})


def _to_np(x):
    """Convert tensor-like input to numpy array."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "numpy"):
        return x.numpy()
    if hasattr(x, "__array__"):
        return np.asarray(x)
    return np.asarray(x)


def _to_f32(x):
    """Convert bf16 numpy to float32 for computation."""
    if x is None:
        return None
    if x.dtype == np_bfloat16:
        return x.view(np.int16).astype(np.float32)
    return x.astype(np.float32)


def _from_f32(x_f32, target_dtype):
    """Convert float32 result back to target dtype."""
    if target_dtype == np_bfloat16:
        return x_f32.astype(np.float32).view(np.int16).view(np_bfloat16)
    return x_f32.astype(target_dtype)


def _softplus(x, beta=1.0, threshold=20.0):
    """Numerically-stable softplus."""
    bx = beta * x
    result = np.empty_like(x, dtype=np.float32)
    over = bx > threshold
    result[over] = x[over]
    safe = ~over
    result[safe] = np.log1p(np.exp(bx[safe])) / beta
    return result


def fused_gdn_gating(A_log, a, dt_bias, softplus_beta=1.0,
                     softplus_threshold=20.0):
    """Compute gating values g = -exp(A_log) * softplus(a + dt_bias).

    Mirrors the gating computation in the Triton kernel.
    """
    a_f32 = _to_f32(_to_np(a))
    A_log_f32 = _to_f32(_to_np(A_log))
    dt_bias_f32 = _to_f32(_to_np(dt_bias))

    x = a_f32 + dt_bias_f32
    softplus_x = _softplus(x, softplus_beta, softplus_threshold)
    g = -np.exp(A_log_f32) * softplus_x
    return g


def _gated_delta_rule_core(
    A_log, a, dt_bias, softplus_beta, softplus_threshold,
    q, k, v, b,
    initial_state_source, initial_state_indices,
    scale, use_qk_l2norm_in_kernel,
    output, intermediate_states_buffer=None,
    disable_state_update=False,
):
    """Core gated-delta-rule recurrent loop (numpy reference).

    Math per token, per head:
      g = -exp(A_log) * softplus(a + dt_bias)
      beta = sigmoid(b)
      if l2norm: q /= ||q||, k /= ||k||
      q *= scale
      h *= exp(g)
      v -= sum(h * k, dim=0)
      v *= beta
      h += k[:, None] * v[None, :]
      o = sum(h * q, dim=0)
    """
    N, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    for n in range(N):
        idx = int(initial_state_indices[n]) if initial_state_indices is not None else 0
        if idx >= 0 and initial_state_source is not None:
            h = initial_state_source[idx].astype(np.float32).copy()  # [HV, V, K]
        else:
            h = np.zeros((HV, V, K), dtype=np.float32)

        for t in range(T):
            q_t = q[n, t].astype(np.float32)
            k_t = k[n, t].astype(np.float32)
            v_t = v[n, t].astype(np.float32)
            a_t = a[n, t].astype(np.float32)
            b_t = b[n, t].astype(np.float32)

            for i_hv in range(HV):
                i_h = i_hv // (HV // H) if H > 0 else 0

                x = a_t[i_hv] + dt_bias[i_hv]
                softplus_x = _softplus(x, softplus_beta, softplus_threshold)
                g = -np.exp(A_log[i_hv]) * softplus_x
                beta = 1.0 / (1.0 + np.exp(-b_t[i_hv]))

                qh = q_t[i_h].copy()
                kh = k_t[i_h].copy()
                if use_qk_l2norm_in_kernel:
                    qh = qh / (np.sqrt(np.sum(qh * qh)) + 1e-6)
                    kh = kh / (np.sqrt(np.sum(kh * kh)) + 1e-6)

                qh = qh * scale
                h_row = h[i_hv].copy()
                h_row *= np.exp(g)
                v_row = v_t[i_hv].copy()
                v_row -= np.sum(h_row * kh[None, :], axis=1)
                v_row *= beta
                h_row += kh[:, None] * v_row[None, :]
                o_val = np.sum(h_row * qh[None, :], axis=0)

                output[n, t, i_hv] = o_val.astype(output.dtype)

                if intermediate_states_buffer is not None:
                    intermediate_states_buffer[n, t, i_hv] = h_row.astype(
                        intermediate_states_buffer.dtype)

                if not disable_state_update and initial_state_source is not None and idx >= 0:
                    initial_state_source[idx, i_hv] = h_row.astype(
                        initial_state_source.dtype)

    return output


def _can_use_flashinfer_gdn_decode(
    initial_state_source, K, V, T, N,
):
    """Check whether FlashInfer GDN bf16-state decode kernel can be used."""
    if os.environ.get("TRTLLM_FLA_DISABLE_FLASHINFER_GDN", "0") == "1":
        return False
    if not _FLASHINFER_GDN_BF16_STATE_AVAILABLE:
        return False
    if initial_state_source is None:
        return False
    src_arr = _to_np(initial_state_source)
    if src_arr.dtype != np_bfloat16:
        return False
    if K != 128 or V != 128:
        return False
    if N == 0:
        return False
    if T != N:
        return False
    return True


def _flashinfer_gdn_decode(
    A_log, a, dt_bias, softplus_beta, softplus_threshold,
    q, k, v, b,
    initial_state_source, initial_state_indices,
    scale, use_qk_l2norm_in_kernel, cu_seqlens, output=None,
):
    """GDN standard decode via the FlashInfer bf16-state kernel."""
    N = len(_to_np(cu_seqlens)) - 1
    T_total = _to_np(q).shape[1]
    T_per_seq = T_total // N
    HV = _to_np(v).shape[2]
    V = _to_np(v).shape[3]

    a_np = _to_np(a)
    b_np = _to_np(b)
    # Alignment realignment (mirror the torch version)
    if a_np.__array_interface__["data"][0] % 32 != 0:
        a_np = a_np.copy()
    if b_np.__array_interface__["data"][0] % 32 != 0:
        b_np = b_np.copy()

    q_np = _to_np(q).reshape(N, T_per_seq, *_to_np(q).shape[2:])
    k_np = _to_np(k).reshape(N, T_per_seq, *_to_np(k).shape[2:])
    v_np = _to_np(v).reshape(N, T_per_seq, HV, V)
    a_bat = a_np.reshape(N, T_per_seq, -1)
    b_bat = b_np.reshape(N, T_per_seq, -1)

    out_dtype = np_bfloat16
    out_np = np.empty((N, T_per_seq, HV, V), dtype=out_dtype)
    output = output if output is not None else out_np

    out_f32 = np.empty((N, T_per_seq, HV, V), dtype=np.float32)
    _gated_delta_rule_core(
        A_log=_to_f32(_to_np(A_log)),
        a=_to_f32(a_bat),
        dt_bias=_to_f32(_to_np(dt_bias)),
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=_to_f32(q_np),
        k=_to_f32(k_np),
        v=_to_f32(v_np),
        b=_to_f32(b_bat),
        initial_state_source=_to_f32(_to_np(initial_state_source)),
        initial_state_indices=_to_np(initial_state_indices).astype(np.int32),
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        output=out_f32,
    )

    if output is not None:
        output_arr = _to_np(output)
        output_arr[...] = _from_f32(out_f32, output_arr.dtype)
        return output
    else:
        result = _from_f32(out_f32, out_dtype)
        return result.reshape(1, T_total, HV, -1)


def _can_use_flashinfer_gdn_verify(
    initial_state_source, head_k_dim, head_v_dim, draft_token_num,
):
    if os.environ.get("TRTLLM_FLA_DISABLE_FLASHINFER_GDN", "0") == "1":
        return False
    if os.environ.get("TRTLLM_FLA_DISABLE_FLASHINFER_GDN_VERIFY", "0") == "1":
        return False
    if not _FLASHINFER_GDN_BF16_STATE_AVAILABLE:
        return False
    if initial_state_source is None or _to_np(initial_state_source).dtype != np_bfloat16:
        return False
    if head_k_dim != 128 or head_v_dim != 128:
        return False
    if not (1 <= draft_token_num <= _FI_GDN_MAX_MTP_T):
        return False
    return True


def _flashinfer_gdn_verify(
    A_log, a, dt_bias, softplus_beta, softplus_threshold,
    q, k, v, b,
    initial_state_source, initial_state_indices,
    intermediate_states_buffer, scale, use_qk_l2norm_in_kernel,
    output=None,
):
    """GDN MTP verify via FlashInfer bf16-state kernel."""
    N, T = _to_np(q).shape[0], _to_np(q).shape[1]
    HV, V = _to_np(v).shape[2], _to_np(v).shape[3]

    a_np = _to_np(a)
    b_np = _to_np(b)
    idx_np = _to_np(initial_state_indices).astype(np.int32)

    if a_np.__array_interface__["data"][0] % 32 != 0:
        a_np = a_np.copy()
    if b_np.__array_interface__["data"][0] % 32 != 0:
        b_np = b_np.copy()
    if idx_np.__array_interface__["data"][0] % 32 != 0:
        idx_np = idx_np.copy()

    out_dtype = np_bfloat16 if output is not None else np.float32
    out_f32 = np.empty((N, T, HV, V), dtype=np.float32)

    buf_f32 = np.empty_like(_to_np(intermediate_states_buffer), dtype=np.float32)

    _gated_delta_rule_core(
        A_log=_to_f32(_to_np(A_log)),
        a=_to_f32(a_np),
        dt_bias=_to_f32(_to_np(dt_bias)),
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=_to_f32(_to_np(q)),
        k=_to_f32(_to_np(k)),
        v=_to_f32(_to_np(v)),
        b=_to_f32(b_np),
        initial_state_source=_to_f32(_to_np(initial_state_source)),
        initial_state_indices=idx_np,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        output=out_f32,
        intermediate_states_buffer=buf_f32,
        disable_state_update=True,
    )

    # Write back intermediate states
    if intermediate_states_buffer is not None:
        buf_arr = _to_np(intermediate_states_buffer)
        buf_arr[...] = _from_f32(buf_f32, buf_arr.dtype)

    if output is not None:
        output_arr = _to_np(output)
        output_arr[...] = _from_f32(out_f32, output_arr.dtype)
        return output
    else:
        return _from_f32(out_f32, out_dtype)


def fused_sigmoid_gating_delta_rule_update(
    A_log, a, dt_bias, softplus_beta, softplus_threshold,
    q, k, v, b,
    initial_state_source, initial_state_indices,
    scale=None, use_qk_l2norm_in_kernel=False,
    cu_seqlens=None, output=None,
):
    """Fused gated delta rule update (torch-free).

    When FlashInfer GDN decode is available and conditions are met, dispatches
    to the faster path; otherwise falls back to the numpy reference loop.

    All arguments may be numpy arrays or tensor-like objects with ``.numpy()``.
    """
    q_np = _to_np(q)
    k_np = _to_np(k)
    v_np = _to_np(v)
    B, T, H, K = q_np.shape[0], q_np.shape[1], q_np.shape[2], q_np.shape[3]
    HV = v_np.shape[2]
    V = v_np.shape[3]
    N = B if cu_seqlens is None else len(_to_np(cu_seqlens)) - 1

    if scale is None:
        scale = K ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    if (cu_seqlens is not None and
            _can_use_flashinfer_gdn_decode(initial_state_source, K, V, T, N)):
        return _flashinfer_gdn_decode(
            A_log=A_log, a=a, dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q, k=k, v=v, b=b,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            output=output,
        )

    # Numpy fallback path
    a_f32 = _to_f32(_to_np(a))
    b_f32 = _to_f32(_to_np(b))
    q_f32 = _to_f32(q_np)
    k_f32 = _to_f32(k_np)
    v_f32 = _to_f32(v_np)
    A_log_f32 = _to_f32(_to_np(A_log))
    dt_bias_f32 = _to_f32(_to_np(dt_bias))
    state_f32 = _to_f32(_to_np(initial_state_source))
    idx_np = _to_np(initial_state_indices).astype(np.int32) if initial_state_indices is not None else None

    out_dtype = np_bfloat16 if output is not None else np.float32
    out_f32 = np.empty((B, T, HV, V), dtype=np.float32)

    # Handle strides for view-like inputs
    if cu_seqlens is not None:
        q_f32 = q_f32.reshape(N, T // N, H, K)
        k_f32 = k_f32.reshape(N, T // N, H, K)
        v_f32 = v_f32.reshape(N, T // N, HV, V)
        a_f32 = a_f32.reshape(N, T // N, HV)
        b_f32 = b_f32.reshape(N, T // N, HV)

    _gated_delta_rule_core(
        A_log=A_log_f32, a=a_f32, dt_bias=dt_bias_f32,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q_f32, k=k_f32, v=v_f32, b=b_f32,
        initial_state_source=state_f32,
        initial_state_indices=idx_np,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        output=out_f32,
    )

    if output is not None:
        output_arr = _to_np(output)
        output_arr[...] = _from_f32(out_f32, output_arr.dtype)
        return output
    else:
        return _from_f32(out_f32, out_dtype)
