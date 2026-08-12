# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/fused_recurrent.py
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import numpy as np

from tensorrt_llm._torch.modules.fla.op import exp


def fused_recurrent_gated_delta_rule_fwd(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    scale: float,
    initial_state: Optional[np.ndarray],
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[np.ndarray] = None,
    output: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    o = output if output is not None else np.empty((N, *v.shape), dtype=q.dtype)
    if output_final_state:
        final_state = np.zeros((N, HV, V, K), dtype=np.float32)
    else:
        final_state = None

    for n in range(N):
        if cu_seqlens is not None:
            bos = int(cu_seqlens[n])
            eos = int(cu_seqlens[n + 1])
        else:
            bos = n * T
            eos = bos + T

        seq_T = eos - bos
        h = np.zeros((HV, V, K), dtype=np.float32)

        if initial_state is not None:
            h = initial_state[n].astype(np.float32)

        for t in range(seq_T):
            for hv in range(HV):
                h[hv] *= exp(g[bos + t, hv])
                v_t = v[bos + t, hv]
                k_t = k[bos + t, hv % H]
                beta_t = beta[bos + t, hv]

                v_t = v_t - np.sum(h[hv] * k_t[:, None], axis=0)
                v_t = v_t * beta_t
                h[hv] += k_t[:, None] * v_t[None, :]

                q_t = q[bos + t, hv % H]
                if use_qk_l2norm_in_kernel:
                    q_t = q_t / (np.linalg.norm(q_t) + 1e-6)
                    k_t = k_t / (np.linalg.norm(k_t) + 1e-6)
                q_t = q_t * scale
                o[bos + t, hv] = np.sum(h[hv] * q_t[:, None], axis=0)

        if output_final_state:
            final_state[n] = h

    o = o.squeeze(0) if o.shape[0] == 1 else o
    return o, final_state


def fused_recurrent_gated_delta_rule_update_fwd(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    scale: float,
    initial_state_source: np.ndarray,
    initial_state_indices: np.ndarray,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[np.ndarray] = None,
    disable_state_update: bool = False,
    disable_output_calculation: bool = False,
    intermediate_states_buffer: Optional[np.ndarray] = None,
    cache_steps: Optional[int] = None,
    output: Optional[np.ndarray] = None,
) -> np.ndarray:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    if disable_output_calculation:
        o = np.zeros((1, 1, 1, 1), dtype=q.dtype)
    else:
        o = output if output is not None else np.empty((N, *v.shape), dtype=q.dtype)

    final_state = None

    for n in range(N):
        if cu_seqlens is not None:
            bos = int(cu_seqlens[n])
            eos = int(cu_seqlens[n + 1])
        else:
            bos = n * T
            eos = bos + T

        seq_T = eos - bos
        idx = int(initial_state_indices[n]) if initial_state_indices is not None else 0

        if idx >= 0 and initial_state_source is not None:
            h = initial_state_source[idx].astype(np.float32)
        else:
            h = np.zeros((HV, V, K), dtype=np.float32)

        for t in range(seq_T):
            for hv in range(HV):
                h[hv] *= exp(g[bos + t, hv])
                v_t = v[bos + t, hv]
                k_t = k[bos + t, hv % H]
                beta_t = beta[bos + t, hv]

                v_t = v_t - np.sum(h[hv] * k_t[:, None], axis=0)
                v_t = v_t * beta_t
                h[hv] += k_t[:, None] * v_t[None, :]

                if not disable_output_calculation:
                    q_t = q[bos + t, hv % H]
                    if use_qk_l2norm_in_kernel:
                        q_t = q_t / (np.linalg.norm(q_t) + 1e-6)
                    q_t = q_t * scale
                    o[bos + t, hv] = np.sum(h[hv] * q_t[:, None], axis=0)

        if not disable_state_update and idx >= 0:
            initial_state_source[idx] = h

    o = o.squeeze(0) if o.shape[0] == 1 else o
    return o


def fused_recurrent_gated_delta_rule(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray = None,
    scale: float = None,
    initial_state: np.ndarray = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[np.ndarray] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1]**-0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = np.ones(q.shape[:-1], dtype=q.dtype)

    o, final_state = fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        output=output,
    )
    return o, final_state


def fused_recurrent_gated_delta_rule_update(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray = None,
    scale: float = None,
    initial_state_source: np.ndarray = None,
    initial_state_indices: np.ndarray = None,
    cu_seqlens: Optional[np.ndarray] = None,
    use_qk_l2norm_in_kernel: bool = False,
    disable_state_update: bool = False,
    disable_output_calculation: bool = False,
    intermediate_states_buffer: Optional[np.ndarray] = None,
    cache_steps: Optional[int] = None,
    output: Optional[np.ndarray] = None,
) -> np.ndarray:
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.")
        if (initial_state_source is not None
                and initial_state_indices.shape[0] != len(cu_seqlens) - 1):
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state_indices.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1]**-0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = np.ones(q.shape[:-1], dtype=q.dtype)

    o = fused_recurrent_gated_delta_rule_update_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        disable_state_update=disable_state_update,
        disable_output_calculation=disable_output_calculation,
        intermediate_states_buffer=intermediate_states_buffer,
        cache_steps=cache_steps,
        output=output,
    )
    return o
