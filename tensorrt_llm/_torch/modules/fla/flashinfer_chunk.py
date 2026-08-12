# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer GDN prefill adapter for ``Qwen3NextGatedDeltaNet.forward_extend``.

Exposes ``chunk_gated_delta_rule`` with a signature call-compatible with the
vendored Triton ``tensorrt_llm._torch.modules.fla.chunk.chunk_gated_delta_rule``.

When ``TLLM_VULKAN_BACKEND=1``, this module uses the pure-numpy flashinfer
package and numpy-based helper functions. No torch import is required at runtime.
"""

from typing import Optional, Tuple

import numpy as np

from tensorrt_llm._torch.modules.fla.fused_state_io import (
    cast_scatter,
    gather_cast,
)
from tensorrt_llm._torch.modules.fla.l2norm import l2norm_fwd


def chunk_gated_delta_rule(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    scale: Optional[float] = None,
    initial_state: Optional[np.ndarray] = None,
    initial_state_indices: Optional[np.ndarray] = None,
    inplace_indexed_state_update: bool = False,
    output_final_state: bool = False,
    cu_seqlens: Optional[np.ndarray] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Adapter for FlashInfer's chunk_gated_delta_rule.

    When torch is not available, delegates to numpy-based flashinfer package.
    """
    import flashinfer

    # Step 1: pre-flight asserts
    assert head_first is False, "head_first=True is not supported by this wrapper"
    assert q.ndim == 4 and q.shape[0] == 1, f"q must be [1, T, H_q, D_k], got {q.shape}"
    assert k.shape[2] == q.shape[2], (
        f"num_q_heads ({q.shape[2]}) must equal num_k_heads ({k.shape[2]})"
    )
    assert g.dtype == np.float32, f"g must be fp32, got {g.dtype}"
    assert cu_seqlens is not None, "cu_seqlens is required (varlen mode)"
    assert initial_state is not None, "initial_state is required"
    if inplace_indexed_state_update:
        assert initial_state_indices is not None, (
            "inplace_indexed_state_update=True requires initial_state_indices"
        )

    # Step 2: layout [1, T, H, D] -> [T, H, D]
    q3 = np.ascontiguousarray(q[0])
    k3 = np.ascontiguousarray(k[0])
    v3 = np.ascontiguousarray(v[0])
    # Convert g from log-space to linear-space alpha (exp)
    g2 = np.exp(np.ascontiguousarray(g[0]))
    beta2 = np.ascontiguousarray(beta[0]).astype(np.float32)

    # Step 3: emulate use_qk_l2norm_in_kernel
    if use_qk_l2norm_in_kernel:
        q3 = l2norm_fwd(q3)
        k3 = l2norm_fwd(k3)

    # Step 4: gather initial state
    state_dtype = initial_state.dtype
    gathered_init = gather_cast(initial_state, initial_state_indices, out_dtype=state_dtype)

    # Step 5: call flashinfer
    total_seq_len = q3.shape[0]
    num_o_heads = max(q3.shape[1], v3.shape[1])
    head_size = q3.shape[2]
    need_state = inplace_indexed_state_update or output_final_state
    
    if output is not None:
        output_buf = np.ascontiguousarray(output[0])
    else:
        output_buf = np.empty((total_seq_len, num_o_heads, head_size), dtype=q3.dtype)

    if need_state:
        num_seqs = cu_seqlens.shape[0] - 1
        state_buf = np.empty((num_seqs, num_o_heads, head_size, head_size), dtype=state_dtype)
        out_packed, out_state = flashinfer.chunk_gated_delta_rule(
            q=q3,
            k=k3,
            v=v3,
            g=g2,
            beta=beta2,
            scale=scale,
            initial_state=gathered_init,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            output=output_buf,
            output_state=state_buf,
        )
    else:
        out_packed = flashinfer.chunk_gated_delta_rule(
            q=q3,
            k=k3,
            v=v3,
            g=g2,
            beta=beta2,
            scale=scale,
            initial_state=gathered_init,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            output=output_buf,
        )
        out_state = None

    # Step 7: cast state back, scatter / return
    final_to_return = None
    if inplace_indexed_state_update:
        cast_scatter(out_state, initial_state, initial_state_indices)
    elif output_final_state:
        num_seqs_out, num_h_out, v_out, k_out = out_state.shape
        final_to_return = np.empty(
            (num_seqs_out, num_h_out, v_out, k_out),
            dtype=initial_state.dtype,
        )
        cast_scatter(out_state, final_to_return, None)

    # Step 8: restore output layout
    out = out_packed[np.newaxis, ...]

    return out, final_to_return