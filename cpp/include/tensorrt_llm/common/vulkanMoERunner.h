// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Standalone C++ MoE runner using Vulkan compute backend (no Python/torch dependency).

#ifndef VULKAN_MOE_RUNNER_H
#define VULKAN_MOE_RUNNER_H

#include "tensorrt_llm/common/vulkanBackend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tensorrt_llm::common {

class VulkanMoERunner
{
public:
    struct MoEConfig
    {
        uint32_t num_experts = 8;
        uint32_t top_k = 2;
        uint32_t hidden_size = 4096;
        uint32_t intermediate_size = 14336;
        bool swiglu_activation = true;
        bool has_fc1_bias = false;
        bool has_fc2_bias = false;
    };

    explicit VulkanMoERunner(MoEConfig config);

    // Run the full MoE forward pass.
    // All buffers are host-side float32. The runner handles staging to Vulkan
    // device internally and returns the result in *output*.
    //
    // Parameters:
    //   hidden_states: [num_tokens, hidden_size]
    //   routing_logits: [num_tokens, num_experts]
    //   gemm1_weights: [num_experts, intermediate_size*2, hidden_size] (flattened per expert)
    //   gemm2_weights: [num_experts, hidden_size, intermediate_size] (flattened per expert)
    //   fc1_biases: [num_experts, intermediate_size*2] (optional, nullptr if not used)
    //   fc2_biases: [num_experts, hidden_size] (optional, nullptr if not used)
    //   output: [num_tokens, hidden_size] (caller-allocated)
    void run_moe(const float* hidden_states,
                 const float* routing_logits,
                 const float* const* gemm1_weights,
                 const float* const* gemm2_weights,
                 const float* const* fc1_biases,  // nullptr if no bias
                 const float* const* fc2_biases,  // nullptr if no bias
                 uint32_t num_tokens,
                 float* output);

    // Run single GEMM profile (non-routing path)
    void run_gemm_profile(const float* x,
                          const float* const* fc1_expert_weights,
                          const float* const* fc1_expert_biases,
                          const float* const* fc2_expert_weights,
                          const float* const* fc2_expert_biases,
                          uint32_t num_tokens,
                          float* output);

private:
    MoEConfig mConfig;
    VulkanBackend* mBackend;

    // Internal staging helpers (mirror torch_bridge.py pattern)
    void* stage_input(const float* data, size_t count);
    void* stage_indices(const uint32_t* data, size_t count);
    void unstage_output(float* out, void* buf, size_t count);
    void free_buffer(void* ptr);

    // Compute helpers using VulkanBackend static methods
    void vulkan_gemm_wrapper(const float* a, const float* b, float* out,
                             uint32_t M, uint32_t N, uint32_t K);
    void vulkan_softmax_wrapper(const float* in, float* out,
                                uint32_t rows, uint32_t cols);
    void vulkan_topk_wrapper(const float* in, uint32_t* out_idx, float* out_val,
                             uint32_t rows, uint32_t cols, uint32_t topk);
    void vulkan_sigmoid_mul_wrapper(const float* a, const float* b, float* out,
                                    size_t n);
    void vulkan_elementwise_add_wrapper(const float* a, const float* b, float* out,
                                        size_t n);
    void vulkan_scale_rows_wrapper(const float* in, const float* scale, float* out,
                                   uint32_t rows, uint32_t cols);
    void vulkan_cast_wrapper(const float* in, float* out, size_t n, int32_t dtype);
    void vulkan_index_add_wrapper(float* out, const uint32_t* indices,
                                  const float* values, uint32_t out_rows,
                                  uint32_t val_rows, uint32_t cols);
    void vulkan_fill_wrapper(float* buf, float value, size_t n);
    void vulkan_gather_wrapper(const float* src, const uint32_t* indices,
                               float* out, size_t num_indices, size_t elem_size);
    void vulkan_compare_eq_wrapper(const float* input, uint32_t* output,
                                   float threshold, size_t n);
};

} // namespace tensorrt_llm::common

#endif // VULKAN_MOE_RUNNER_H
