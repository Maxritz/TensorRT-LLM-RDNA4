# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/chunk.py
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import numpy as np

from tensorrt_llm._torch.modules.fla.chunk_delta_h import \
    chunk_gated_delta_rule_fwd_h
from tensorrt_llm._torch.modules.fla.chunk_o import chunk_fwd_o
from tensorrt_llm._torch.modules.fla.chunk_scaled_dot_kkt import \
    chunk_scaled_dot_kkt_fwd
from tensorrt_llm._torch.modules.fla.cumsum import chunk_local_cumsum
from tensorrt_llm._torch.modules.fla.l2norm import l2norm_fwd
from tensorrt_llm._torch.modules.fla.solve_tril import solve_tril
from tensorrt_llm._torch.modules.fla.utils import SUPPRESS_LEVEL
from tensorrt_llm._torch.modules.fla.wy_fast import recompute_w_u_fwd


def chunk_gated_delta_rule_fwd(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    scale: float,
    initial_state: np.ndarray,
    initial_state_indices: Optional[np.ndarray],
    inplace_indexed_state_update: bool,
    output_final_state: bool,
    cu_seqlens: Optional[np.ndarray] = None,
    output: Optional[np.ndarray] = None,
) -> Tuple:
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    A = chunk_scaled_dot_kkt_fwd(k=k,
                                 beta=beta,
                                 g_cumsum=g,
                                 cu_seqlens=cu_seqlens,
                                 output_dtype=np.float32)
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        output_final_state=output_final_state,
        inplace_indexed_state_update=inplace_indexed_state_update,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        output=output,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


def chunk_gated_delta_rule(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    scale: float = None,
    initial_state: np.ndarray = None,
    initial_state_indices: Optional[np.ndarray] = None,
    inplace_indexed_state_update: bool = False,
    output_final_state: bool = False,
    cu_seqlens: Optional[np.ndarray] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    r"""
    Args:
        q: [B, T, H, K] if head_first=False else [B, H, T, K].
        k: [B, T, H, K] if head_first=False else [B, H, T, K].
        v: [B, T, H, V] if head_first=False else [B, H, T, V].
        g: [B, T, H] if head_first=False else [B, H, T]. (forget) gating tensor (in log space!)
        beta: [B, T, H] if head_first=False else [B, H, T].
        scale: Scale factor for the RetNet attention scores.
        initial_state: [N, H, V, K] for N input sequences.
        initial_state_indices: [N] state-pool indices.
        inplace_indexed_state_update: opt-in for writing indexed final states back.
        output_final_state: Whether to output the final state of shape [N, H, V, K].
        cu_seqlens: [N+1] cumulative sequence lengths.
        head_first: Deprecated; use head_first=False.
        use_qk_l2norm_in_kernel: Apply L2 norm to q/k before processing.
        output: Pre-allocated output buffer.

    Returns:
        o: [B, T, H, V] if head_first=False else [B, H, T, V].
        final_state: [N, H, V, K] if output_final_state=True else None.
    """
    if head_first:
        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))
        beta = np.transpose(beta, (0, 2, 1))
        g = np.transpose(g, (0, 2, 1))

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.")
        num_sequences = len(cu_seqlens) - 1
        if initial_state_indices is not None:
            if initial_state_indices.shape[0] != num_sequences:
                raise ValueError(
                    f"The number of initial-state indices is expected to be equal to the number of input "
                    f"sequences, i.e., {num_sequences} rather than {initial_state_indices.shape[0]}."
                )
        elif initial_state is not None and initial_state.shape[0] != num_sequences:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {num_sequences} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    o, final_state = _chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        inplace_indexed_state_update=inplace_indexed_state_update,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        output=output,
    )

    if head_first:
        o = np.transpose(o, (0, 2, 1, 3))
    
    return o, final_state


def _chunk_gated_delta_rule(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    scale: float,
    initial_state: np.ndarray,
    initial_state_indices: Optional[np.ndarray],
    inplace_indexed_state_update: bool,
    output_final_state: bool,
    cu_seqlens: Optional[np.ndarray],
    output: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Internal entry that calls chunk_gated_delta_rule_fwd."""
    g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        inplace_indexed_state_update=inplace_indexed_state_update,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        output=output,
    )
    return o, final_state