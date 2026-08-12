# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/cumsum.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/cumsum.py
# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np

from tensorrt_llm._torch.modules.fla.index import prepare_chunk_indices


def chunk_local_cumsum_scalar(
    g: np.ndarray,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[np.ndarray] = None,
    head_first: bool = False,
    output_dtype: Optional[np.dtype] = np.float32,
) -> np.ndarray:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2**(chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    BT = chunk_size
    chunk_indices = (prepare_chunk_indices(cu_seqlens, BT)
                     if cu_seqlens is not None else None)
    NT = (T + BT - 1) // BT if cu_seqlens is None else len(chunk_indices)
    
    g_org = g.astype(np.float32 if output_dtype is not None else g.dtype)
    g_out = np.empty_like(g_org)
    
    if cu_seqlens is None:
        for i_t in range(NT):
            s = i_t * BT
            e = min(s + BT, T)
            if head_first:
                for i_bh in range(B * H):
                    i_b, i_h = i_bh // H, i_bh % H
                    b_s = g_org[i_b, i_h, s:e]
                    b_o = np.cumsum(b_s, axis=0)
                    if reverse:
                        b_z = np.sum(b_s, axis=0)
                        b_o = -b_o + b_z + b_s
                    if scale is not None:
                        b_o *= scale
                    g_out[i_b, i_h, s:e] = b_o.astype(g_out.dtype)
            else:
                for i_bh in range(B * H):
                    i_b, i_h = i_bh // H, i_bh % H
                    b_s = g_org[i_b, s:e, i_h]
                    b_o = np.cumsum(b_s, axis=0)
                    if reverse:
                        b_z = np.sum(b_s, axis=0)
                        b_o = -b_o + b_z + b_s
                    if scale is not None:
                        b_o *= scale
                    g_out[i_b, s:e, i_h] = b_o.astype(g_out.dtype)
    else:
        for i_n in range(len(cu_seqlens) - 1):
            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            T_seq = eos - bos
            nt = (T_seq + BT - 1) // BT
            indices = chunk_indices[i_n * 2: i_n * 2 + 2] if len(chunk_indices) > 0 else None
            seq_chunks = []
            for i_t in range(nt):
                ts = i_t * BT
                te = min(ts + BT, T_seq)
                if head_first:
                    for i_h in range(H):
                        b_s = g_org[bos:bos + T_seq, i_h] if False else g_org[0, i_h, bos:te]
                        b_o = np.cumsum(b_s, axis=0)
                        if reverse:
                            b_z = np.sum(b_s, axis=0)
                            b_o = -b_o + b_z + b_s
                        if scale is not None:
                            b_o *= scale
                        g_out[0, i_h, bos:te] = b_o.astype(g_out.dtype)
                else:
                    for i_h in range(H):
                        b_s = g_org[0, bos:te, i_h]
                        b_o = np.cumsum(b_s, axis=0)
                        if reverse:
                            b_z = np.sum(b_s, axis=0)
                            b_o = -b_o + b_z + b_s
                        if scale is not None:
                            b_o *= scale
                        g_out[0, bos:te, i_h] = b_o.astype(g_out.dtype)
    
    return g_out


def chunk_local_cumsum_vector(
    g: np.ndarray,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[np.ndarray] = None,
    head_first: bool = False,
    output_dtype: Optional[np.dtype] = np.float32,
) -> np.ndarray:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    chunk_indices = (prepare_chunk_indices(cu_seqlens, chunk_size)
                     if cu_seqlens is not None else None)
    NT = (T + BT - 1) // BT if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2**(chunk_size.bit_length() - 1), "chunk_size must be a power of 2"

    g_org = g.astype(np.float32 if output_dtype is not None else g.dtype)
    g_out = np.empty_like(g_org)

    if cu_seqlens is None:
        for i_t in range(NT):
            s = i_t * BT
            e = min(s + BT, T)
            if head_first:
                b_s = g_org[:, :, s:e, :]
                b_o = np.cumsum(b_s, axis=-2)
            else:
                b_s = g_org[:, s:e, :, :]
                b_o = np.cumsum(b_s, axis=1)
            if reverse:
                b_z = np.sum(b_s, axis=(-2 if head_first else 1), keepdims=True)
                b_o = -b_o + b_z + b_s
            if scale is not None:
                b_o *= scale
            if head_first:
                g_out[:, :, s:e, :] = b_o.astype(g_out.dtype)
            else:
                g_out[:, s:e, :, :] = b_o.astype(g_out.dtype)
    else:
        for i_n in range(len(cu_seqlens) - 1):
            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            T_seq = eos - bos
            nt = (T_seq + BT - 1) // BT
            for i_t in range(nt):
                ts = i_t * BT
                te = min(ts + BT, T_seq)
                if head_first:
                    b_s = g_org[0, :, bos:te, :]
                    b_o = np.cumsum(b_s, axis=-2)
                else:
                    b_s = g_org[0, bos:te, :, :]
                    b_o = np.cumsum(b_s, axis=0)
                if reverse:
                    b_z = np.sum(b_s, axis=(-2 if head_first else 0), keepdims=True)
                    b_o = -b_o + b_z + b_s
                if scale is not None:
                    b_o *= scale
                if head_first:
                    g_out[0, :, bos:te, :] = b_o.astype(g_out.dtype)
                else:
                    g_out[0, bos:te, :, :] = b_o.astype(g_out.dtype)
    
    return g_out


def chunk_local_cumsum(
    g: np.ndarray,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[np.ndarray] = None,
    head_first: bool = False,
    output_dtype: Optional[np.dtype] = np.float32,
    **kwargs,
) -> np.ndarray:
    if cu_seqlens is not None:
        assert (g.shape[0] == 1), "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    elif len(g.shape) == 4:
        return chunk_local_cumsum_vector(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    else:
        raise ValueError(f"Unsupported input shape {g.shape}, "
                         f"which should be (B, T, H, D) if `head_first=False` "
                         f"or (B, H, T, D) otherwise")