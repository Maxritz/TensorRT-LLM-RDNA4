# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/wy_fast.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/wy_fast.py
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import numpy as np

from tensorrt_llm._torch.modules.fla.index import prepare_chunk_indices
from tensorrt_llm._torch.modules.fla.op import exp


def recompute_w_u_fwd(
    k: np.ndarray,
    v: np.ndarray,
    beta: np.ndarray,
    g_cumsum: np.ndarray,
    A: np.ndarray,
    cu_seqlens: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recompute w and u from k, v, beta, g_cumsum, and A.
    
    k: [B, T, Hg, K]
    v: [B, T, H, V]
    beta: [B, T, H]
    g_cumsum: [B, T, H]
    A: [B, T, H, BT]
    
    Returns:
    w: [B, T, H, K] — beta * k * exp(g)
    u: [B, T, H, V] — A @ (beta * v)
    """
    B, T, Hg, K = k.shape
    H = v.shape[-2]
    V = v.shape[-1]
    BT = A.shape[-1]

    chunk_indices = (prepare_chunk_indices(cu_seqlens, BT)
                     if cu_seqlens is not None else None)
    NT = (T + BT - 1) // BT if cu_seqlens is None else len(chunk_indices)
    BK = 64
    BV = 64
    u = np.empty_like(v, dtype=np.float32)
    w = np.empty((B, T, H, K), dtype=np.float32)

    if cu_seqlens is None:
        for i_t in range(NT):
            ts = i_t * BT
            te = min(ts + BT, T)
            cur_BT = te - ts
            
            for b in range(B):
                for h in range(H):
                    hg = h // (H // Hg) if H > Hg else h
                    
                    k_chunk = k[b, ts:te, hg, :].astype(np.float32)  # (cur_BT, K)
                    v_chunk = v[b, ts:te, h, :].astype(np.float32)  # (cur_BT, V)
                    beta_chunk = beta[b, ts:te, h].astype(np.float32)  # (cur_BT,)
                    g_chunk = g_cumsum[b, ts:te, h].astype(np.float32)  # (cur_BT,)
                    A_block = A[b, ts:te, h, :cur_BT].astype(np.float32)  # (cur_BT, cur_BT)
                    
                    # u = A @ (beta * v)
                    vb = v_chunk * beta_chunk[:, None]
                    u[b, ts:te, h, :] = A_block @ vb
                    
                    # w = beta * k * exp(g)
                    g_exp = exp(g_chunk)
                    w[b, ts:te, h, :] = k_chunk * beta_chunk[:, None] * g_exp[:, None]
    else:
        for i_n in range(len(cu_seqlens) - 1):
            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            T_seq = eos - bos
            nt = (T_seq + BT - 1) // BT
            
            for i_t in range(nt):
                ts = i_t * BT
                te = min(ts + BT, T_seq)
                cur_BT = te - ts
                
                for h in range(H):
                    hg = h // (H // Hg) if H > Hg else h
                    
                    k_chunk = k[0, bos + ts:bos + te, hg, :].astype(np.float32)
                    v_chunk = v[0, bos + ts:bos + te, h, :].astype(np.float32)
                    beta_chunk = beta[0, bos + ts:bos + te, h].astype(np.float32)
                    g_chunk = g_cumsum[0, bos + ts:bos + te, h].astype(np.float32)
                    A_block = A[0, bos + ts:bos + te, h, :cur_BT].astype(np.float32)
                    
                    vb = v_chunk * beta_chunk[:, None]
                    u[0, bos + ts:bos + te, h, :] = A_block @ vb
                    
                    g_exp = exp(g_chunk)
                    w[0, bos + ts:bos + te, h, :] = k_chunk * beta_chunk[:, None] * g_exp[:, None]

    return w, u


fwd_recompute_w_u = recompute_w_u_fwd