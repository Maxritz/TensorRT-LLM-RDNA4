# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_scaled_dot_kkt.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/chunk_scaled_dot_kkt.py
# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np

from tensorrt_llm._torch.modules.fla.index import prepare_chunk_indices


def chunk_scaled_dot_kkt_fwd(
    k: np.ndarray,
    beta: np.ndarray,
    g_cumsum: Optional[np.ndarray] = None,
    cu_seqlens: Optional[np.ndarray] = None,
    chunk_size: int = 64,
    output_dtype=np.float32,
) -> np.ndarray:
    r"""
    Compute beta * K * K^T for each chunk.
    
    Args:
        k: [B, T, Hg, K]
        beta: [B, T, H]
        g_cumsum: [B, T, H] — cumulative sum of gate (log space)
        cu_seqlens: [N+1] cumulative sequence lengths
        chunk_size: chunk size BT
        output_dtype: output dtype
    
    Returns:
        beta * K * K^T of shape [B, T, H, BT]
    """
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    chunk_indices = (prepare_chunk_indices(cu_seqlens, BT)
                     if cu_seqlens is not None else None)
    NT = (T + BT - 1) // BT if cu_seqlens is None else len(chunk_indices)
    
    A = np.empty((B, T, H, BT), dtype=np.float32)
    
    if cu_seqlens is None:
        for i_t in range(NT):
            ts = i_t * BT
            te = min(ts + BT, T)
            cur_BT = te - ts
            
            for b in range(B):
                for h in range(H):
                    # k[b, ts:te, h_or_hg, :] — shape (cur_BT, K)
                    hg = h // (H // Hg) if H > Hg else h
                    k_chunk = k[b, ts:te, hg, :].astype(np.float32)  # (cur_BT, K)
                    beta_chunk = beta[b, ts:te, h].astype(np.float32)  # (cur_BT,)
                    
                    # beta[:, None] * k @ k.T  ->  (cur_BT, K) * (K, cur_BT) = (cur_BT, cur_BT)
                    kb = k_chunk * beta_chunk[:, None]  # (cur_BT, K)
                    a_block = kb @ k_chunk.T  # (cur_BT, cur_BT)
                    
                    if g_cumsum is not None:
                        g_chunk = g_cumsum[b, ts:te, h].astype(np.float32)  # (cur_BT,)
                        g_diff = g_chunk[:, None] - g_chunk[None, :]
                        # safe_exp
                        exp_diff = np.where(g_diff > 0, np.exp(np.clip(g_diff, -500, 500)),
                                           np.exp(np.clip(g_diff, -500, 500)))
                        a_block = a_block * exp_diff
                    
                    # Zero out upper triangle (only keep lower triangular)
                    mask = np.arange(cur_BT)[:, None] > np.arange(cur_BT)[None, :]
                    a_block = np.where(mask, a_block, 0.0)
                    
                    # Write to full A matrix
                    A[b, ts:te, h, :cur_BT] = a_block
    else:
        # Variable length
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
                    beta_chunk = beta[0, bos + ts:bos + te, h].astype(np.float32)
                    
                    kb = k_chunk * beta_chunk[:, None]
                    a_block = kb @ k_chunk.T
                    
                    if g_cumsum is not None:
                        g_chunk = g_cumsum[0, bos + ts:bos + te, h].astype(np.float32)
                        g_diff = g_chunk[:, None] - g_chunk[None, :]
                        exp_diff = np.where(g_diff > 0, np.exp(np.clip(g_diff, -500, 500)),
                                           np.exp(np.clip(g_diff, -500, 500)))
                        a_block = a_block * exp_diff
                    
                    mask = np.arange(cur_BT)[:, None] > np.arange(cur_BT)[None, :]
                    a_block = np.where(mask, a_block, 0.0)
                    
                    A[0, bos + ts:bos + te, h, :cur_BT] = a_block
    
    if output_dtype is not None and output_dtype != np.float32:
        A = A.astype(output_dtype)
    
    return A