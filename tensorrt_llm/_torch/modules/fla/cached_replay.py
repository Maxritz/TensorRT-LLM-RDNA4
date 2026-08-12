# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached replay for GDN verification — numpy implementation.

When TLLM_VULKAN_BACKEND=1, this module provides numpy implementations of the
cached replay kernels. The torch imports are available only for type annotations
and are not required at runtime.
"""

from typing import Optional

import numpy as np


_SMALL_GRID_HEAD_TILES = 512
_BV16_MAX_HEAD_TILES = 64
_BV32_MAX_HEAD_TILES = 128
_RATIO2_FINE_MAPPING_MAX_HEAD_TILES = 256
_EIGHT_WARP_COMMIT_HEAD_TILES = 1024
_PIPELINED_COMMIT_HEAD_TILES = 2048
_TWO_STAGE_REPLAY_HEAD_TILES = 4096
_L2_STREAMING_HEAD_TILES = 8192

CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE = 16


def _default_cached_replay_block_v(
    use_tuned_bf16_mapping: bool,
    head_tiles: int,
    value_dim: int,
) -> int:
    if use_tuned_bf16_mapping:
        if head_tiles <= _BV16_MAX_HEAD_TILES:
            return 16
        if head_tiles <= _BV32_MAX_HEAD_TILES:
            return 32
        if head_tiles <= _SMALL_GRID_HEAD_TILES:
            return 64
    p = 1
    while p < value_dim:
        p <<= 1
    return p


def _supports_fine_grained_replay_tiling(
    num_key_heads: int,
    num_value_heads: int,
    head_tiles: int,
) -> bool:
    return num_value_heads == 4 * num_key_heads or (
        num_value_heads == 2 * num_key_heads and head_tiles <= _RATIO2_FINE_MAPPING_MAX_HEAD_TILES
    )


def _exp(x):
    return np.exp(np.clip(x, -500, 500))


def _safe_exp(x):
    return np.where(x != x, 0.0, np.exp(np.clip(x, -500, 500)))


def fused_recurrent_gated_delta_rule_cached_replay_update(
    q, k, v, g, beta, ssm_states, state_indices,
    old_u, old_k, old_G, old_beta, cache_buf_idx,
    prev_num_accepted_tokens, history_size,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    A_log: Optional = None,
    dt_bias: Optional = None,
    launch_with_pdl: bool = False,
    replay_work_items: Optional = None,
    n_writes: Optional = None,
    block_v: Optional[int] = None,
    num_warps: Optional[int] = None,
    use_l2_state_cache: Optional[bool] = None,
    use_l2_streaming_inputs: Optional[bool] = None,
    use_l2_shared_inputs: Optional[bool] = None,
    main_num_stages: Optional[int] = None,
    packed_qkv: Optional = None,
    use_all_layer_commit: bool = False,
    output: Optional = None,
) -> np.ndarray:
    """GDN replay using cached causal updates rather than raw history.

    This is a simplified numpy implementation that processes each token sequentially,
    using the cached history vectors stored in old_u, old_k, old_G.
    """
    del old_beta
    N, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    assert q.shape == k.shape
    assert v.shape[:2] == (N, T)
    assert g.shape == (N, T, HV)
    assert beta.shape == (N, T, HV)

    use_packed_qkv = packed_qkv is not None
    if scale is None:
        scale = K ** -0.5

    fused_gating = A_log is not None
    if fused_gating:
        assert dt_bias is not None

    BK = K
    BT = T
    BH = history_size
    BV = V if block_v is None else block_v
    assert BV <= V

    if output is None:
        output = np.zeros((N, T, HV, V), dtype=np.float32)

    for b in range(N):
        for i_hv in range(HV):
            i_h = i_hv // (HV // H)
            
            # Load initial state
            if state_indices is not None:
                slot = state_indices[b]
            else:
                slot = -1
            
            # Get history from cached values
            if slot >= 0:
                b_pnat = prev_num_accepted_tokens[slot] if len(prev_num_accepted_tokens) > slot else 0
                b_buf = cache_buf_idx[slot] if len(cache_buf_idx) > slot else 0
                b_kh = old_k[0, slot * 2 + b_buf, :b_pnat, i_h, :K].astype(np.float32)
                b_uh = old_u[0, slot * 2 + b_buf, :b_pnat, i_hv, :V].astype(np.float32)
                b_Gh = old_G[0, slot * 2 + b_buf, :b_pnat, i_hv].astype(np.float32)
            else:
                b_kh = np.zeros((0, K), dtype=np.float32)
                b_uh = np.zeros((0, V), dtype=np.float32)
                b_Gh = np.zeros(0, dtype=np.float32)
            
            # Get current tokens
            if use_packed_qkv:
                qkv_width = 2 * H * K + HV * V
                kv_base = b * T
                q_row = qkv_row_base = (kv_base) * qkv_width
                b_q = q[b, :T, i_h, :K].astype(np.float32)
                b_k = packed_qkv[b, :T].reshape(T, qkv_width)[:, H*K:2*H*K].reshape(T, K) if packed_qkv is not None else q[b, :T, i_h, :K].astype(np.float32)
                b_v = packed_qkv[b, :T].reshape(T, qkv_width)[:, 2*H*K:].reshape(T, HV*V)[:, i_hv*V:(i_hv+1)*V] if packed_qkv is not None else v[b, :T, i_hv, :V].astype(np.float32)
            else:
                b_q = q[b, :T, i_h, :K].astype(np.float32)
                b_k = k[b, :T, i_h, :K].astype(np.float32)
                b_v = v[b, :T, i_hv, :V].astype(np.float32)
            
            b_g = g[b, :T, i_hv].astype(np.float32)
            b_beta = beta[b, :T, i_hv].astype(np.float32)

            if use_qk_l2norm_in_kernel:
                inv_k = 1.0 / (np.sqrt(np.sum(b_k * b_k, axis=-1)) + 1e-6)
                inv_q = scale / (np.sqrt(np.sum(b_q * b_q, axis=-1)) + 1e-6)
                b_kn = b_k * inv_k[:, None]
                b_qn = b_q * inv_q[:, None]
            else:
                b_kn = b_k
                b_qn = b_q * scale

            # Gating
            if fused_gating:
                g_A_exp = _exp(A_log[i_hv])
                g_dt_bias = dt_bias[i_hv]
                x = b_g + g_dt_bias
                softplus = np.where(x <= 20, 0.6931471805599453 * np.log(1 + np.exp(x)), x)
                b_g = np.where(np.arange(T) < T, -g_A_exp * softplus, 0.0)
                b_beta = 1.0 / (1.0 + np.exp(-b_beta))

            b_G_local = np.cumsum(b_g, axis=0)
            g_start = b_G_local[0] if len(b_G_local) > 0 else 0.0
            b_G = g_start + b_G_local

            # Load initial state from pool (state is (V, K), transpose to (K, V) for matmul)
            if slot >= 0:
                if ssm_states.ndim == 5:
                    b_h0 = ssm_states[0, slot, i_hv, :V, :K].astype(np.float32).T
                elif ssm_states.ndim == 4:
                    b_h0 = ssm_states[0, i_hv, :V, :K].astype(np.float32).T
                elif ssm_states.ndim == 3:
                    b_h0 = ssm_states[0, i_hv, :V, :K].astype(np.float32).T
                else:
                    b_h0 = np.zeros((K, V), dtype=np.float32)
            else:
                b_h0 = np.zeros((K, V), dtype=np.float32)

            hist_decay = _exp(b_G[:, None] - b_Gh[None, :] if len(b_Gh) > 0 else b_G[:, None])

            if len(b_kh) > 0 and len(b_Gh) > 0:
                b_kk_hist = b_kn @ b_kh.T
                b_qk_hist = b_qn @ b_kh.T
                b_k_hist_coeff = b_kk_hist * hist_decay
                b_q_hist_coeff = b_qk_hist * hist_decay
                b_k_hist = b_k_hist_coeff.astype(b_uh.dtype) @ b_uh
                b_q_hist = b_q_hist_coeff.astype(b_uh.dtype) @ b_uh
            else:
                b_k_hist = np.zeros((T, V), dtype=np.float32)
                b_q_hist = np.zeros((T, V), dtype=np.float32)

            b_qh0 = b_qn @ b_h0
            b_kh0 = b_kn @ b_h0

            b_rhs = b_beta[:, None] * (b_v - _exp(b_G)[:, None] * b_kh0 - b_k_hist)
            lower = np.arange(T)[:, None] > np.arange(T)[None, :]
            b_kk_new = b_qn @ b_kn.T
            new_decay = _exp(b_G[:, None] - b_G[None, :])
            b_A = np.where(lower, b_beta[:, None] * b_kk_new * new_decay, 0.0)

            b_U = np.zeros((T, BV), dtype=np.float32)
            for row in range(T):
                row_mask = np.arange(T) == row
                rhs_row = np.sum(np.where(row_mask[:, None], b_rhs[:, :BV], 0.0), axis=0)
                a_row = np.sum(np.where(row_mask[:, None], b_A, 0.0), axis=0)
                correction = np.sum(a_row[:, None] * b_U, axis=0)
                u_row = rhs_row - correction
                b_U[row] = u_row

            incl = np.arange(T)[:, None] >= np.arange(T)[None, :]
            b_qk_new = b_qn @ b_kn.T
            b_q_new_coeff = np.where(incl & lower, b_qk_new * new_decay, 0.0)
            b_q_new = np.zeros((T, BV), dtype=np.float32)
            for row in range(T):
                row_mask = np.arange(T) == row
                coeff_row = np.sum(np.where(row_mask[:, None], b_q_new_coeff, 0.0), axis=0)
                q_new_row = np.sum(coeff_row[:, None] * b_U, axis=0)
                b_q_new[row] = q_new_row

            b_o = _exp(b_G)[:, None] * b_qh0[:, :BV] + b_q_hist[:, :BV] + b_q_new
            output[b, :T, i_hv, :BV] = b_o

    return output


def commit_gdn_cached_replay_history_layers(
    *,
    ssm_states,
    ssm_state_descriptors: Optional = None,
    ssm_state_num_layers: Optional[int] = None,
    ssm_state_layer_stride: Optional[int] = None,
    ssm_state_slot_stride: Optional[int] = None,
    old_u,
    old_k,
    old_G,
    replay_work_items,
    n_writes,
    history_size: int,
    persistent_waves: int = 2,
    commit_block_v: Optional[int] = None,
    commit_num_warps: Optional[int] = None,
    commit_pipeline_stages: Optional[int] = None,
) -> None:
    """Advance all local GDN layer checkpoints from cached replay histories."""
    np_ssm_states = np.asarray(ssm_states, dtype=np.float32)
    np_old_u = np.asarray(old_u, dtype=np.float32)
    np_old_k = np.asarray(old_k, dtype=np.float32)
    np_old_G = np.asarray(old_G, dtype=np.float32)
    np_replay_work_items = np.asarray(replay_work_items, dtype=np.int32)
    
    K = np_old_k.shape[-1]
    V = np_old_u.shape[-1]
    HV = np_old_k.shape[-4] if np_old_k.ndim >= 4 else np_old_k.shape[-3]
    H = HV if np_old_k.ndim == 3 else np_old_k.shape[-4] // 2  # simplified assumption
    N = np_replay_work_items.shape[0]
    
    if N == 0:
        return
    
    for i in range(N):
        work = np_replay_work_items[i]
        slot = work[1]
        b_pnat = work[2]
        b_buf = work[3]
        
        for i_hv in range(HV):
            i_h = i_hv // (HV // H)
            
            b_h = np_ssm_states[0, i_hv, :V, :K].astype(np.float32)
            
            if b_pnat > 0:
                b_kh = np_old_k[0, slot * 2 + b_buf, :b_pnat, i_h, :K]
                b_uh = np_old_u[0, slot * 2 + b_buf, :b_pnat, i_hv, :V]
                b_Gh = np_old_G[0, slot * 2 + b_buf, :b_pnat, i_hv]
                
                if len(b_Gh) > 1:
                    g_start = b_Gh[-1]
                    commit_decay = _exp(g_start - b_Gh)
                    b_Uc = b_uh * np.where(np.arange(len(b_Gh)) < b_pnat, commit_decay, 0.0)[:, None]
                    # b_h is (V, K), b_kh.T @ b_Uc is (K, V) - need to match shapes
                    b_hc = b_h + (b_kh.T @ b_Uc).T
                else:
                    b_hc = b_h
            else:
                b_hc = b_h
            
            np_ssm_states[0, i_hv, :V, :K] = b_hc.astype(np_ssm_states.dtype)