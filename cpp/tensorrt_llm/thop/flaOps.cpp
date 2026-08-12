/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/vulkanBackend.h"
#include "tensorrt_llm/kernels/vulkanKernelRegistry.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>

namespace
{

constexpr int MAX_K = 128;

// CPU fallback: matches _gated_delta_rule_core in
// tensorrt_llm/_torch/modules/fla/fused_sigmoid_gating_recurrent.py
void gatedDeltaRuleCpuKernel(
    const float* q, const float* k, const float* v,
    const float* a, const float* b,
    const float* aLog, const float* dtBias,
    const float* initState, const int* initIndices,
    float* output,
    int32_t N, int32_t T, int32_t H, int32_t HV, int32_t V, int32_t K,
    float scale, float softplusBeta, float softplusThreshold,
    bool useL2Norm, bool disableStateUpdate, bool hasInitialState)
{
    int ratio = (HV > H) ? (HV / H) : 1;

    for (int n = 0; n < N; ++n)
    {
        int initIdx = hasInitialState && initIndices ? initIndices[n] : -1;

        for (int t = 0; t < T; ++t)
        {
            int abBase = n * T * HV + t * HV;
            int qkBase = n * T * H * K + t * H * K;
            int vBase  = n * T * HV * V + t * HV * V;
            int oBase  = n * T * HV * V + t * HV * V;

            for (int hv = 0; hv < HV; ++hv)
            {
                int iH   = hv / ratio;
                int iH_K = iH * K;

                // softplus(a + dt_bias)
                float x = a[abBase + hv] + dtBias[hv];
                float sp;
                if (x > softplusThreshold)
                    sp = x;
                else if (x < -softplusThreshold)
                    sp = std::log1p(std::exp(softplusBeta * x)) / softplusBeta;
                else
                    sp = std::log1p(std::exp(softplusBeta * x)) / softplusBeta;

                float g      = -std::exp(aLog[hv]) * sp;
                float beta   = 1.0f / (1.0f + std::exp(-(b[abBase + hv])));
                float expG   = std::exp(g);

                const float* qRow = &q[qkBase + iH_K];
                const float* kRow = &k[qkBase + iH_K];

                // Pre-compute qh, kh (same across V)
                float qh[MAX_K], kh[MAX_K];
                for (int i = 0; i < K; ++i)
                {
                    qh[i] = qRow[i];
                    kh[i] = kRow[i];
                }

                if (useL2Norm)
                {
                    float qn = 0.f, kn = 0.f;
                    for (int i = 0; i < K; ++i)
                    {
                        qn += qh[i] * qh[i];
                        kn += kh[i] * kh[i];
                    }
                    qn = std::sqrt(qn) + 1e-6f;
                    kn = std::sqrt(kn) + 1e-6f;
                    for (int i = 0; i < K; ++i)
                    {
                        qh[i] /= qn;
                        kh[i] /= kn;
                    }
                }

                for (int i = 0; i < K; ++i)
                    qh[i] *= scale;

                for (int vv = 0; vv < V; ++vv)
                {
                    float h[MAX_K];

                    if (hasInitialState && initIdx >= 0)
                    {
                        const float* hPtr = &initState[initIdx * HV * V * K + hv * V * K + vv * K];
                        for (int i = 0; i < K; ++i)
                            h[i] = hPtr[i];
                    }
                    else
                    {
                        for (int i = 0; i < K; ++i)
                            h[i] = 0.f;
                    }

                    // h *= exp(g)
                    for (int i = 0; i < K; ++i)
                        h[i] *= expG;

                    // v_row -= dot(h, kh)
                    float dotH = 0.f;
                    for (int i = 0; i < K; ++i)
                        dotH += h[i] * kh[i];
                    float vRow = v[vBase + hv * V + vv] - dotH;
                    vRow *= beta;

                    // h += kh * vRow
                    for (int i = 0; i < K; ++i)
                        h[i] += kh[i] * vRow;

                    // output = dot(h, qh)
                    float oVal = 0.f;
                    for (int i = 0; i < K; ++i)
                        oVal += h[i] * qh[i];
                    output[oBase + hv * V + vv] = oVal;

                    // Write back state
                    if (hasInitialState && initIdx >= 0 && !disableStateUpdate)
                    {
                        float* hPtr = &initState[initIdx * HV * V * K + hv * V * K + vv * K];
                        for (int i = 0; i < K; ++i)
                            hPtr[i] = h[i];
                    }
                }
            }
        }
    }
}

} // namespace

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fla_gated_delta_rule_fwd("
        "Tensor q, Tensor k, Tensor v, "
        "Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
        "Tensor? initial_state, Tensor? initial_state_indices, "
        "Tensor? output, "
        "int N, int T, int H, int HV, int V, int K, "
        "float scale, float softplus_beta, float softplus_threshold, "
        "bool use_qk_l2norm_in_kernel, bool disable_state_update) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fla_gated_delta_rule_fwd", TORCH_FN(
        [](torch::Tensor q, torch::Tensor k, torch::Tensor v,
           torch::Tensor a, torch::Tensor b, torch::Tensor A_log, torch::Tensor dt_bias,
           c10::optional<torch::Tensor> initial_state,
           c10::optional<torch::Tensor> initial_state_indices,
           c10::optional<torch::Tensor> output_opt,
           int64_t N, int64_t T, int64_t H, int64_t HV, int64_t V, int64_t K,
           double scale, double softplus_beta, double softplus_threshold,
           bool useL2Norm, bool disableStateUpdate)
        -> torch::Tensor {
            auto opts = q.options();
            auto output = output_opt.has_value() ? output_opt.value()
                                                : torch::empty({N, T, HV, V}, opts);

            TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                        "q, k, v must be contiguous");
            TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                        "a, b must be contiguous");
            TORCH_CHECK(A_log.is_contiguous() && dt_bias.is_contiguous(),
                        "A_log, dt_bias must be contiguous");

#ifdef USE_VULKAN_BACKEND
            if (TLLM_VULKAN_BACKEND_ACTIVE())
            {
                tensorrt_llm::common::VulkanBackend::launchGatedDeltaRule(
                    A_log.data_ptr(), dt_bias.data_ptr(),
                    a.data_ptr(), b.data_ptr(),
                    q.data_ptr(), k.data_ptr(), v.data_ptr(),
                    initial_state.has_value() ? initial_state->data_ptr() : nullptr,
                    initial_state_indices.has_value() ? initial_state_indices->data_ptr() : nullptr,
                    output.data_ptr(),
                    static_cast<uint32_t>(N), static_cast<uint32_t>(T),
                    static_cast<uint32_t>(H), static_cast<uint32_t>(HV),
                    static_cast<uint32_t>(V), static_cast<uint32_t>(K),
                    initial_state.has_value(),
                    disableStateUpdate,
                    static_cast<float>(scale),
                    static_cast<float>(softplus_beta),
                    static_cast<float>(softplus_threshold),
                    useL2Norm);
                tensorrt_llm::common::VulkanBackend::streamSynchronize();
                return output;
            }
#endif
            // CPU fallback
            gatedDeltaRuleCpuKernel(
                q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                a.data_ptr<float>(), b.data_ptr<float>(),
                A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
                initial_state.has_value() ? initial_state->data_ptr<float>() : nullptr,
                initial_state_indices.has_value() ? initial_state_indices->data_ptr<int32_t>() : nullptr,
                output.data_ptr<float>(),
                static_cast<int32_t>(N), static_cast<int32_t>(T),
                static_cast<int32_t>(H), static_cast<int32_t>(HV),
                static_cast<int32_t>(V), static_cast<int32_t>(K),
                static_cast<float>(scale),
                static_cast<float>(softplus_beta),
                static_cast<float>(softplus_threshold),
                useL2Norm, disableStateUpdate,
                initial_state.has_value());
            return output;
        }));
}
