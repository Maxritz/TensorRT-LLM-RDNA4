# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Equivalence test for trtllm.fla_gated_delta_rule_fwd (thop) vs numpy reference.

Validates:
  1. CPU path (CPU tensors) matches _gated_delta_rule_core numpy reference.
  2. Vulkan path (if TLLM_VULKAN_BACKEND=1 and GPU tensors) matches numpy reference.
"""

import os
import pytest
import torch
import numpy as np

from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)

_VK = os.environ.get("TLLM_VULKAN_BACKEND", "0") == "1"


@pytest.mark.parametrize("N,T,H,HV,K,V", [
    (2, 4, 4, 8, 64, 64),
    (1, 8, 2, 4, 128, 128),
    (3, 16, 4, 8, 64, 64),
])
@pytest.mark.parametrize("use_l2norm", [True, False])
def test_cpu_thop_matches_numpy(N, T, H, HV, K, V, use_l2norm):
    torch.manual_seed(42)

    q = torch.randn(N, T, H, K, dtype=torch.float32)
    k = torch.randn(N, T, H, K, dtype=torch.float32)
    v = torch.randn(N, T, HV, V, dtype=torch.float32) * 0.5
    a = torch.randn(N, T, HV, dtype=torch.float32)
    b = torch.randn(N, T, HV, dtype=torch.float32)
    A_log = torch.randn(HV, dtype=torch.float32) * 0.5
    dt_bias = torch.randn(HV, dtype=torch.float32) * 0.5

    scale = K ** -0.5
    softplus_beta = 1.0
    softplus_threshold = 20.0

    # Numpy reference
    out_np = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q, k=k, v=v, b=b,
        initial_state_source=None,
        initial_state_indices=None,
        scale=scale,
        use_qk_l2norm_in_kernel=use_l2norm,
        cu_seqlens=None,
        output=None,
    )

    # Thop (CPU fallback path)
    out_thop = torch.ops.trtllm.fla_gated_delta_rule_fwd(
        q, k, v, a, b, A_log, dt_bias,
        initial_state=None,
        initial_state_indices=None,
        output=None,
        N=N, T=T, H=H, HV=HV, V=V, K=K,
        scale=scale,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        use_qk_l2norm_in_kernel=use_l2norm,
        disable_state_update=False,
    )

    assert out_thop.shape == (N, T, HV, V)
    assert out_thop.dtype == torch.float32
    np.testing.assert_allclose(out_thop.numpy(), out_np, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("N,T,H,HV,K,V", [
    (2, 4, 4, 8, 64, 64),
    (1, 8, 2, 4, 128, 128),
])
@pytest.mark.parametrize("use_l2norm", [True, False])
def test_thop_with_initial_state(N, T, H, HV, K, V, use_l2norm):
    torch.manual_seed(7)

    q = torch.randn(N, T, H, K, dtype=torch.float32)
    k = torch.randn(N, T, H, K, dtype=torch.float32)
    v = torch.randn(N, T, HV, V, dtype=torch.float32) * 0.5
    a = torch.randn(N, T, HV, dtype=torch.float32)
    b = torch.randn(N, T, HV, dtype=torch.float32)
    A_log = torch.randn(HV, dtype=torch.float32) * 0.5
    dt_bias = torch.randn(HV, dtype=torch.float32) * 0.5

    init_state = torch.randn(N, HV, V, K, dtype=torch.float32) * 0.3
    init_indices = torch.arange(N, dtype=torch.int32)
    scale = K ** -0.5

    # Numpy reference (with state writeback — pass a copy)
    init_state_np = init_state.clone()
    out_np = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, dt_bias=dt_bias,
        softplus_beta=1.0, softplus_threshold=20.0,
        q=q, k=k, v=v, b=b,
        initial_state_source=init_state_np,
        initial_state_indices=init_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_l2norm,
        cu_seqlens=None,
        output=None,
    )

    # Thop (CPU fallback, disable_state_update=False so state is written back)
    init_state_thop = init_state.clone()
    out_thop = torch.ops.trtllm.fla_gated_delta_rule_fwd(
        q, k, v, a, b, A_log, dt_bias,
        initial_state=init_state_thop,
        initial_state_indices=init_indices,
        output=None,
        N=N, T=T, H=H, HV=HV, V=V, K=K,
        scale=scale,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        use_qk_l2norm_in_kernel=use_l2norm,
        disable_state_update=False,
    )

    np.testing.assert_allclose(out_thop.numpy(), out_np, atol=1e-5, rtol=1e-5)
    # Final state should match (h_row after T steps written back to init_state)
    np.testing.assert_allclose(
        init_state_thop.numpy(), init_state_np.numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _VK, reason="TLLM_VULKAN_BACKEND not set")
@pytest.mark.parametrize("N,T,H,HV,K,V", [(2, 4, 4, 8, 64, 64)])
def test_vulkan_thop_matches_numpy(N, T, H, HV, K, V):
    torch.manual_seed(99)
    device = "cuda"

    q = torch.randn(N, T, H, K, device=device, dtype=torch.float32)
    k = torch.randn(N, T, H, K, device=device, dtype=torch.float32)
    v = torch.randn(N, T, HV, V, device=device, dtype=torch.float32) * 0.5
    a = torch.randn(N, T, HV, device=device, dtype=torch.float32)
    b = torch.randn(N, T, HV, device=device, dtype=torch.float32)
    A_log = torch.randn(HV, device=device, dtype=torch.float32) * 0.5
    dt_bias = torch.randn(HV, device=device, dtype=torch.float32) * 0.5
    scale = K ** -0.5

    init_state = torch.randn(N, HV, V, K, device=device, dtype=torch.float32) * 0.3
    init_indices = torch.arange(N, device=device, dtype=torch.int32)

    out_np = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log.cpu(), a=a.cpu(), dt_bias=dt_bias.cpu(),
        softplus_beta=1.0, softplus_threshold=20.0,
        q=q.cpu(), k=k.cpu(), v=v.cpu(), b=b.cpu(),
        initial_state_source=init_state.cpu().clone(),
        initial_state_indices=init_indices.cpu(),
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
        output=None,
    )

    out_vk = torch.ops.trtllm.fla_gated_delta_rule_fwd(
        q, k, v, a, b, A_log, dt_bias,
        initial_state=init_state.clone(),
        initial_state_indices=init_indices,
        output=None,
        N=N, T=T, H=H, HV=HV, V=V, K=K,
        scale=scale,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=False,
    )

    np.testing.assert_allclose(out_vk.cpu().numpy(), out_np, atol=1e-3, rtol=1e-3)
