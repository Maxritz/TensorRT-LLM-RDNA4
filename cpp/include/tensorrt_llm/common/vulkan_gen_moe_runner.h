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

#ifndef VULKAN_GEN_MOE_RUNNER_H
#define VULKAN_GEN_MOE_RUNNER_H

#include "tensorrt_llm/common/vulkanBackend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tensorrt_llm::common {

// GenMoERunner: Vulkan-backed block-scale MoE runner for trtllm-gen custom ops.
// Handles FP4, FP8, and MXFP4 block-scale quantization formats used by NVFP4/MXFP4 MoE.
class GenMoERunner
{
public:
    enum class ActivationType
    {
        Sigmoid = 0,
        Relu = 1,
        Gelu = 2,
        Silu = 3,
        Swiglu = 4,
        Relu2 = 5,
        GeluTanh = 6,
    };

    struct MoEConfig
    {
        uint32_t num_experts = 8;
        uint32_t top_k = 2;
        uint32_t hidden_size = 4096;
        uint32_t intermediate_size = 14336;
        ActivationType activation = ActivationType::Swiglu;
        bool has_fc1_bias = false;
        bool has_fc2_bias = false;
        bool has_routing_bias = false;
        bool normalize_scales = false;
        uint32_t n_group = 0;
        uint32_t topk_group = 0;
        uint32_t routing_method_type = 0;
        uint32_t local_expert_offset = 0;
        uint32_t local_num_experts = 0;
        float routed_scaling_factor = 1.0f;
        bool do_finalize = true;
    };

    explicit GenMoERunner(MoEConfig config);

    // Full MoE forward pass with block-scale quantization.
    // Mirrors the Python _VulkanGenMoERunner.run_moe interface.
    void run_moe(
        const float* hidden_states,        // [num_tokens, hidden_size] host fp32
        const float* routing_logits,        // [num_tokens, num_experts] host fp32 (nullable)
        const float* routing_bias,         // [num_experts] host fp32 (nullable)
        const void* hidden_states_quant,   // quantized hidden states (nullable)
        const float* hidden_states_scale,  // hidden scale (nullable)
        const void* const* gemm1_weights,  // [num_experts] quantized weights
        const float* const* gemm1_weights_scale, // [num_experts] weight scales
        const float* const* gemm1_bias,    // [num_experts] bias (nullable)
        const float* gemm1_alpha,          // nullable
        const float* gemm1_beta,           // nullable
        const float* gemm1_clamp_limit,    // nullable
        const void* const* gemm2_weights,  // [num_experts] quantized weights
        const float* const* gemm2_weights_scale, // [num_experts] weight scales
        const float* const* gemm2_bias,    // [num_experts] bias (nullable)
        const float* output1_scale_scalar, // nullable
        const float* output1_scale_gate_scalar, // nullable
        const float* output2_scale_scalar, // nullable
        uint32_t num_tokens,
        const uint32_t* topk_ids,          // [num_tokens, top_k] (nullable if routing_logits given)
        const float* topk_weights,         // [num_tokens, top_k] (nullable if routing_logits given)
        float* output);                    // [num_tokens, hidden_size] host fp32 (caller-allocated)

    // Single GEMM profile (no routing, all experts processed separately)
    void run_gemm_profile(
        const float* x,                     // [num_tokens, hidden_size]
        const void* const* fc1_expert_weights,  // [num_experts]
        const float* const* fc1_expert_biases,   // [num_experts] or nullptr
        const void* const* fc2_expert_weights,  // [num_experts]
        const float* const* fc2_expert_biases,   // [num_experts] or nullptr
        uint32_t num_tokens,
        float* output);                    // [num_tokens, hidden_size]

    // Config queries for autotuner
    int get_num_configs();
    std::vector<int> get_valid_configs();
    int get_tactic_num(int gemm_idx);

    // Workspace management
    void clear_workspaces();
    void clear_cache();

private:
    MoEConfig mConfig;
    VulkanBackend* mBackend;
    void* mWorkspace = nullptr;

    // Staging helpers
    void* stage_input(const void* data, size_t count);
    void* stage_input_fp4(const void* data, size_t count);
    void unstage_output(float* out, void* buf, size_t count);
    void free_buffer(void* ptr);

    // Compute wrappers (fp32, staging through Vulkan)
    void vulkan_gemm_wrapper(const float* a, const float* b, float* out,
                             uint32_t M, uint32_t N, uint32_t K);
    void vulkan_softmax_wrapper(const float* in, float* out,
                                uint32_t rows, uint32_t cols);
    void vulkan_topk_wrapper(const float* in, uint32_t* out_idx, float* out_val,
                             uint32_t rows, uint32_t cols, uint32_t topk);
    void vulkan_sigmoid_mul_wrapper(const float* a, const float* b, float* out, size_t n);
    void vulkan_elementwise_add_wrapper(const float* a, const float* b, float* out, size_t n);
    void vulkan_scale_rows_wrapper(const float* in, const float* scale, float* out,
                                   uint32_t rows, uint32_t cols);
    void vulkan_index_add_wrapper(float* out, const uint32_t* indices,
                                  const float* values, uint32_t out_rows,
                                  uint32_t val_rows, uint32_t cols);
    void vulkan_topk_general_wrapper(const float* in, uint32_t* out_idx, float* out_val,
                                     uint32_t rows, uint32_t cols, uint32_t topk);

    // Block-scale dequantization + GEMM
    void run_block_scale_gemv_wrapper(
        const void* weight_data, const float* weight_scale,
        const float* activation, uint32_t M, uint32_t N, uint32_t K,
        float* output);
};

} // namespace tensorrt_llm::common

#endif // VULKAN_GEN_MOE_RUNNER_H
