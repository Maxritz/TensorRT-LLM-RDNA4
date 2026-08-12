# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/solve_tril.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/solve_tril.py
# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np


def solve_tril_16x16_numpy(A_block):
    """Solve (I + A)^-1 for a single 16x16 strictly-lower-triangular block.
    
    A_block: (16, 16) array, strictly lower triangular (A.triu() == 0).
    Returns: (16, 16) lower-triangular array = (I + A)^-1.
    """
    I = np.eye(16, dtype=np.float32)
    return np.linalg.inv(I + A_block)


def solve_tril_numpy(A: np.ndarray, cu_seqlens: Optional[np.ndarray] = None,
                     output_dtype=None) -> np.ndarray:
    """Pure-numpy replacement for solve_tril.
    
    A: [B, T, H, BT] — the strictly-lower-triangular block matrix.
    Returns: (I + A)^-1 with the same shape.
    """
    assert A.shape[-1] in [16, 32, 64], f"Unsupported BT={A.shape[-1]}, only 16/32/64 supported"
    
    B, T, H, BT = A.shape
    dtype = output_dtype if output_dtype is not None else A.dtype
    
    if BT == 16:
        # Direct inversion for 16x16 blocks
        result = np.empty_like(A, dtype=np.float32)
        for b in range(B):
            for t in range(T):
                for h in range(H):
                    result[b, t, h] = solve_tril_16x16_numpy(A[b, t, h].astype(np.float32))
        return result.astype(dtype)
    
    # For 32 and 64, use block inversion
    Ai = np.empty_like(A, dtype=np.float32)
    
    for b in range(B):
        for t in range(T):
            for h in range(H):
                A_block = A[b, t, h].astype(np.float32)
                
                if BT == 32:
                    # Split into four 16x16 blocks
                    A11 = A_block[:16, :16]
                    A12 = A_block[:16, 16:]
                    A21 = A_block[16:, :16]
                    A22 = A_block[16:, 16:]
                    
                    I16 = np.eye(16, dtype=np.float32)
                    inv_A11 = solve_tril_16x16_numpy(A11)
                    # Schur complement: S = A22 - A21 @ inv_A11 @ A12
                    S = A22 - A21 @ inv_A11 @ A12
                    inv_S = solve_tril_16x16_numpy(S)
                    
                    result_block = np.zeros((32, 32), dtype=np.float32)
                    result_block[:16, :16] = inv_A11 + inv_A11 @ A12 @ inv_S @ A21 @ inv_A11
                    result_block[:16, 16:] = -inv_A11 @ A12 @ inv_S
                    result_block[16:, :16] = -inv_S @ A21 @ inv_A11
                    result_block[16:, 16:] = inv_S
                    
                    Ai[b, t, h] = result_block
                
                elif BT == 64:
                    # Split into four 32x32 blocks, each handled as above
                    blocks = []
                    for i in range(4):
                        for j in range(4):
                            sub = A_block[i*16:i*16+16, j*16:j*16+16]
                            blocks.append(sub)
                        # Actually handle as 2x2 of 32x32
                    # Simpler: just use full numpy inversion on the 64x64
                    Ai[b, t, h] = np.linalg.inv(np.eye(64, dtype=np.float32) + A_block)
    
    return Ai.astype(dtype)


def solve_tril(
    A: np.ndarray,
    cu_seqlens: Optional[np.ndarray] = None,
    output_dtype=None,
) -> np.ndarray:
    """
    Compute the inverse of the lower triangular matrix.
    A should be strictly lower triangular, i.e., A.triu() == 0.
    """
    return solve_tril_numpy(A, cu_seqlens, output_dtype)


def solve_tril_torch(A, cu_seqlens=None, output_dtype=None):
    """Torch-compatible wrapper for solve_tril — delegates to numpy."""
    np_A = np.asarray(A)
    if hasattr(A, 'cpu'):
        np_A = A.cpu().numpy()
    out = solve_tril_numpy(np_A, 
                          cu_seqlens=None if cu_seqlens is None else (cu_seqlens.cpu().numpy() if hasattr(cu_seqlens, 'cpu') else np.asarray(cu_seqlens)),
                          output_dtype=None)
    if hasattr(A, 'device'):
        import torch as _t
        if output_dtype is not None:
            _dt = output_dtype
            if hasattr(_dt, 'numpy'):
                _dt = np.dtype(_dt)
            if hasattr(_dt, 'is_floating_point'):
                if _dt.is_floating_point:
                    out = out.astype(np.float16 if _dt == _t.float16 else np.float32)
        return _t.from_numpy(out).to(A.device)
    return out