# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_o.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/chunk_o.py
# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np

from tensorrt_llm._torch.modules.fla.index import prepare_chunk_indices
from tensorrt_llm._torch.modules.fla.op import exp, safe_exp


def chunk_fwd_o(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    h: np.ndarray,
    g: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[np.ndarray] = None,
    chunk_size: int = 64,
    output: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute chunk-wise output: o = scale * q @ h + scale * (triangular_mask * (q @ k^T)) @ v
    
    q: [B, T, H_q, K]
    k: [B, T, H_q, K]
    v: [B, T, H_v, V]
    h: [B, NT, H, K, V] — intermediate state
    g: [B, T, H] — cumulative log decay
    """
    B, T, Hg, K = q.shape
    H = v.shape[-2]
    V = v.shape[-1]
    
    BT = min(chunk_size, max(16, 2 ** ((T - 1).bit_length() if T > 0 else 1)))
    if T > 0:
        BT_actual = min(chunk_size, T)
    else:
        BT_actual = chunk_size
    BT = min(chunk_size, 2 ** ((T - 1).bit_length() if T > 0 else 0)) if T > 0 else chunk_size
    BT = min(chunk_size, 2 ** max(4, (T - 1).bit_length())) if T > 0 else chunk_size
    # Simpler: just use chunk_size
    BT = chunk_size
    
    chunk_indices = (prepare_chunk_indices(cu_seqlens, BT)
                     if cu_seqlens is not None else None)
    NT = (T + BT - 1) // BT if cu_seqlens is None else len(chunk_indices)
    
    if scale is None:
        scale = K ** -0.5

    o = output if output is not None else np.empty_like(v, dtype=np.float32)
    o = o.astype(np.float32)

    if cu_seqlens is None:
        for i_t in range(NT):
            ts = i_t * BT
            te = min(ts + BT, T)
            cur_BT = te - ts
            
            for b in range(B):
                for hv in range(H):
                    h_idx = i_t  # chunk index for this sequence
                    h_block = h[b, h_idx, hv].astype(np.float32)  # (K, V)
                    
                    q_chunk = q[b, ts:te, hv if Hg <= H else 0, :].astype(np.float32)  # (cur_BT, K)
                    k_chunk = k[b, ts:te, hv if Hg <= H else 0, :].astype(np.float32)  # (cur_BT, K)
                    v_chunk = v[b, ts:te, hv, :].astype(np.float32)  # (cur_BT, V)
                    
                    # First term: q @ h -> (cur_BT, V)
                    o_term1 = q_chunk @ h_block  # (cur_BT, K) @ (K, V) = (cur_BT, V)
                    
                    # Second term: (triu(q @ k^T) * scale) @ v
                    A = q_chunk @ k_chunk.T  # (cur_BT, cur_BT)
                    mask = np.arange(cur_BT)[:, None] >= np.arange(cur_BT)[None, :]
                    A = np.where(mask, A, 0.0)
                    
                    if g is not None:
                        g_chunk = g[b, ts:te, hv].astype(np.float32)
                        g_diff = g_chunk[:, None] - g_chunk[None, :]
                        exp_diff = safe_exp(g_diff)
                        A = A * exp_diff
                    
                    o_term2 = A @ v_chunk  # (cur_BT, V)
                    
                    # b_g scaling
                    if g is not None:
                        g_chunk = g[b, ts:te, hv].astype(np.float32)
                        g_exp = exp(g_chunk)
                        o_term1 = o_term1 * g_exp[:, None]
                    
                    o[b, ts:te, hv, :] = (o_term1 + o_term2) * scale
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
                
                for hv in range(H):
                    h_idx = i_t
                    h_block = h[0, h_idx, hv].astype(np.float32)
                    
                    q_chunk = q[0, bos + ts:bos + te, hv if Hg <= H else 0, :].astype(np.float32)
                    k_chunk = k[0, bos + ts:bos + te, hv if Hg <= H else 0, :].astype(np.float32)
                    v_chunk = v[0, bos + ts:bos + te, hv, :].astype(np.float32)
                    
                    o_term1 = q_chunk @ h_block
                    A = q_chunk @ k_chunk.T
                    mask = np.arange(cur_BT)[:, None] >= np.arange(cur_BT)[None, :]
                    A = np.where(mask, A, 0.0)
                    
                    if g is not None:
                        g_chunk = g[0, bos + ts:bos + te, hv].astype(np.float32)
                        g_diff = g_chunk[:, None] - g_chunk[None, :]
                        exp_diff = safe_exp(g_diff)
                        A = A * exp_diff
                        g_chunk = g[0, bos + ts:bos + te, hv].astype(np.float32)
                        g_exp = exp(g_chunk)
                        o_term1 = o_term1 * g_exp[:, None]
                    
                    o_term2 = A @ v_chunk
                    o[0, bos + ts:bos + te, hv, :] = (o_term1 + o_term2) * scale
    
    if output is not None:
        np.copyto(output.astype(np.float32), o)
        return output
    return o