# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_delta_h.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/chunk_delta_h.py
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import numpy as np

from tensorrt_llm._torch.modules.fla.index import prepare_chunk_indices, prepare_chunk_offsets
from tensorrt_llm._torch.modules.fla.op import exp, safe_exp


def chunk_gated_delta_rule_fwd_h(
    k: np.ndarray,
    w: np.ndarray,
    u: np.ndarray,
    g: Optional[np.ndarray] = None,
    initial_state: Optional[np.ndarray] = None,
    initial_state_indices: Optional[np.ndarray] = None,
    output_final_state: bool = False,
    inplace_indexed_state_update: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Forward pass of the chunk-wise gated linear attention recurrence.
    
    Computes:
      - h[b, t, h, k, v]: intermediate state matrix
      - v_new: updated values
      - final_state: final state if output_final_state=True
    """
    B, T, Hg, K = k.shape
    H = u.shape[-2]
    V = u.shape[-1]
    BT = chunk_size

    chunk_indices = (prepare_chunk_indices(cu_seqlens, chunk_size)
                     if cu_seqlens is not None else None)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, (T + BT - 1) // BT, None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )

    h = np.zeros((B, NT, H, K, V), dtype=np.float32)
    use_indexed_state = initial_state is not None and initial_state_indices is not None
    if use_indexed_state and not inplace_indexed_state_update:
        raise ValueError(
            "Indexed chunk state updates require inplace_indexed_state_update=True."
        )
    store_final_state_in_kernel = output_final_state and not use_indexed_state
    final_state = (np.zeros((N, H, V, K), dtype=np.float32)
                   if store_final_state_in_kernel else None)

    v_new = np.empty_like(u, dtype=np.float32) if save_new_value else None

    # Process each sequence and chunk
    for b in range(B if cu_seqlens is None else 1):
        seq_b = b if cu_seqlens is None else 0
        T_seq = T if cu_seqlens is None else cu_seqlens[b + 1] - cu_seqlens[b] if cu_seqlens is not None else T
        bos = 0 if cu_seqlens is None else cu_seqlens[b]
        eos = T_seq if cu_seqlens is None else cu_seqlens[b + 1]
        
        if cu_seqlens is not None:
            nt_seq = (T_seq + BT - 1) // BT
        else:
            nt_seq = NT

        for i_h in range(H):
            hg = i_h // (H // Hg) if H > Hg else i_h
            
            # Initial state
            b_h = np.zeros((K, V), dtype=np.float32)
            if initial_state is not None:
                if use_indexed_state:
                    slot = initial_state_indices[seq_b]
                    if store_final_state_in_kernel:
                        b_h = initial_state[slot, i_h].astype(np.float32)
                    else:
                        b_h = initial_state[seq_b, i_h].astype(np.float32)
                else:
                    b_h = initial_state[seq_b, i_h].astype(np.float32)

            for i_t in range(nt_seq):
                ts = i_t * BT
                te = min(ts + BT, T_seq) if cu_seqlens is None else min(bos + ts + BT, eos)
                cur_T = te - ts if cu_seqlens is None else min(ts + BT, T_seq)
                
                # Store current h
                h[seq_b, i_t, i_h] = b_h

                # Load v, w, k for this chunk
                v_chunk = u[seq_b, ts:ts + cur_T if cu_seqlens is None else bos + ts:bos + te, i_h, :].astype(np.float32)
                w_chunk = w[seq_b, ts:ts + cur_T if cu_seqlens is None else bos + ts:bos + te, i_h, :].astype(np.float32)
                k_chunk = k[seq_b, ts:ts + cur_T if cu_seqlens is None else bos + ts:bos + te, hg, :].astype(np.float32)
                if g is not None:
                    g_chunk = g[seq_b, ts:ts + cur_T if cu_seqlens is None else bos + ts:bos + te, i_h].astype(np.float32)

                # Compute v_new
                b_v_new = v_chunk - w_chunk @ b_h  # (cur_T, V) - (cur_T, K) @ (K, V)
                
                if save_new_value:
                    # v_new shape: (B, T, H, V)
                    v_new[seq_b, ts if cu_seqlens is None else bos + ts:ts if cu_seqlens is None else bos + te, i_h, :] = b_v_new

                # Compute new h
                beta_chunk = 1.0  # beta is absorbed into w already
                
                # h_new = b_h + k^T @ v_new
                b_h = b_h + k_chunk.T @ b_v_new

                if g is not None:
                    g_last = g_chunk[-1]
                    g_last_exp = exp(g_last)
                    b_h = b_h * g_last_exp
                    
                    # v_new also gets scaled
                    g_decay = np.exp(g_chunk[-1] - g_chunk)
                    b_v_new = b_v_new * g_decay[:, None]

    if output_final_state and use_indexed_state:
        final_state = np.zeros((N, H, V, K), dtype=np.float32)
        for i in range(N):
            slot = initial_state_indices[i]
            final_state[i] = initial_state[slot].astype(np.float32)
    
    return h, v_new, final_state