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

#ifndef VULKAN_GEM_RUNNER_H
#define VULKAN_GEM_RUNNER_H

#include "tensorrt_llm/common/vulkanBackend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tensorrt_llm::common {

// VulkanGemmRunner: Standalone C++ GEMM runner using Vulkan compute backend.
// Supports FP32, FP16, FP8, FP4 (block-scale and weight-only) via Vulkan shaders.
class VulkanGemmRunner
{
public:
    enum class GemmType
    {
        FP32,
        FP16,
        FP8_BLOCK_SCALE,
        FP8_WEIGHT_ONLY,
        FP4_BLOCK_SCALE,
        FP4_WEIGHT_ONLY,
        W4_WEIGHT_ONLY,
        W8_WEIGHT_ONLY,
        MXFP8,
        MXFP4,
        FUSED_FP8,
    };

    struct GemmConfig
    {
        GemmType type = GemmType::FP32;
        uint32_t output_dtype = 0; // 0=fp32, 1=fp16, 2=bf16
        bool has_bias = false;
        bool has_act = false;       // SiLU/SwGLU activation
        bool has_residual = false;
    };

    explicit VulkanGemmRunner(GemmConfig config);

    // Simple GEMM: C = A * B (+ optional bias)
    // A: [M, K], B: [K, N], C: [M, N] — all host fp32
    void run_gemm(const float* a, const float* b, const float* bias,
                  uint32_t M, uint32_t N, uint32_t K, float* output);

    // Block-scale GEMM: dequantize A/B using per-block scales, then multiply.
    // A: [M, K] fp8, A_scale: [M, K/block_size] fp8
    // B: [K, N] fp8, B_scale: [K/block_size, N] fp8
    // Output: [M, N] fp32
    void run_block_scale_gemm(
        const void* a_data, const void* a_scale,
        const void* b_data, const void* b_scale,
        const float* bias,
        uint32_t M, uint32_t N, uint32_t K,
        uint32_t block_size,
        float* output);

    // Weight-only GEMM: dequantize weights from int8/float8, multiply with fp16/fp32 input.
    // weight: quantized, weight_scale: fp32 scale per column
    // act: [M, K] (fp16 or fp32), output: [M, N] fp32
    void run_weight_only_gemm(
        const void* act, const void* weight, const float* weight_scale,
        const float* bias,
        uint32_t M, uint32_t N, uint32_t K,
        float* output);

    // Fused GEMM + activation: C = act(A*B + bias), where act is SiLU/SwGLU
    void run_fused_gemm_act(
        const void* a, const void* b, const float* bias,
        uint32_t M, uint32_t N, uint32_t K,
        float* output);

    // Profile a single GEMM (used by trtllm-gen autotuner)
    void run_gemm_profile(
        const void* a, const void* b, const float* bias,
        const void* a_scale, const void* b_scale,
        uint32_t M, uint32_t N, uint32_t K,
        uint32_t block_size,
        float* output);

    // Get valid configs for autotuning
    std::vector<int> get_valid_configs(uint32_t M, uint32_t N, uint32_t K);
    int get_num_configs();
    int get_num_heuristic_algos(uint32_t M, uint32_t N, uint32_t K);

    // Workspace management (no-ops for Vulkan backend)
    void clear_workspaces() {}
    int get_tactic_num(int gemm_idx) { return 0; }
    void clear_cache() {}

private:
    GemmConfig mConfig;
    VulkanBackend* mBackend;

    // Staging helpers (same pattern as VulkanMoERunner)
    void* stage_input(const void* data, size_t count);
    void unstage_output(float* out, void* buf, size_t count);
    void free_buffer(void* ptr);

    // Internal compute wrappers
    void vulkan_gemm_wrapper(const float* a, const float* b, float* out,
                             uint32_t M, uint32_t N, uint32_t K);
    void vulkan_elementwise_add_wrapper(const float* a, const float* b, float* out,
                                        size_t n);
    void vulkan_silu_wrapper(const float* in, float* out, size_t n);
    void vulkan_sigmoid_wrapper(const float* in, float* out, size_t n);
};

} // namespace tensorrt_llm::common

#endif // VULKAN_GEM_RUNNER_H
