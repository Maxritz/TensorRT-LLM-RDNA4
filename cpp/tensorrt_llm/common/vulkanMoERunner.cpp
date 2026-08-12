// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Standalone C++ MoE runner using Vulkan compute backend (no Python/torch dependency).

#include "tensorrt_llm/common/vulkanMoERunner.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace tensorrt_llm::common {

VulkanMoERunner::VulkanMoERunner(MoEConfig config)
    : mConfig(config)
    , mBackend(nullptr)
{
}

void* VulkanMoERunner::stage_input(const float* data, size_t count)
{
    size_t bytes = count * sizeof(float);
    void* ptr = VulkanBackend::malloc(bytes);
    if (!ptr) return nullptr;
    VulkanBackend::memcpyHostToDevice(ptr, data, bytes);
    return ptr;
}

void* VulkanMoERunner::stage_indices(const uint32_t* data, size_t count)
{
    size_t bytes = count * sizeof(uint32_t);
    void* ptr = VulkanBackend::malloc(bytes);
    if (!ptr) return nullptr;
    VulkanBackend::memcpyHostToDevice(ptr, data, bytes);
    return ptr;
}

void VulkanMoERunner::unstage_output(float* out, void* buf, size_t count)
{
    size_t bytes = count * sizeof(float);
    VulkanBackend::memcpyDeviceToHost(out, buf, bytes);
}

void VulkanMoERunner::free_buffer(void* ptr)
{
    if (ptr) VulkanBackend::free(ptr);
}

void VulkanMoERunner::vulkan_gemm_wrapper(const float* a, const float* b, float* out,
                                          uint32_t M, uint32_t N, uint32_t K)
{
    void* pa = stage_input(a, M * K);
    void* pb = stage_input(b, K * N);
    void* pout = VulkanBackend::malloc(M * N * sizeof(float));
    VulkanBackend::launchFp16Gemm(pa, pb, pout, M, N, K);
    unstage_output(out, pout, M * N);
    free_buffer(pa);
    free_buffer(pb);
    free_buffer(pout);
}

void VulkanMoERunner::vulkan_softmax_wrapper(const float* in, float* out,
                                             uint32_t rows, uint32_t cols)
{
    // Use elementwise approach: softmax per row
    void* pin = stage_input(in, rows * cols);
    void* pout = VulkanBackend::malloc(rows * cols * sizeof(float));
    VulkanBackend::launchSoftmax(pin, pout, rows, 1, cols);
    unstage_output(out, pout, rows * cols);
    free_buffer(pin);
    free_buffer(pout);
}

void VulkanMoERunner::vulkan_topk_wrapper(const float* in, uint32_t* out_idx, float* out_val,
                                          uint32_t rows, uint32_t cols, uint32_t topk)
{
    void* pin = stage_input(in, rows * cols);
    void* pidx = VulkanBackend::malloc(rows * topk * sizeof(uint32_t));
    void* pval = VulkanBackend::malloc(rows * topk * sizeof(float));
    VulkanBackend::launchTopKGeneral(pin, pidx, pval, rows, cols, topk);
    VulkanBackend::memcpyDeviceToHost(out_idx, pidx, rows * topk * sizeof(uint32_t));
    unstage_output(out_val, pval, rows * topk);
    free_buffer(pin);
    free_buffer(pidx);
    free_buffer(pval);
}

void VulkanMoERunner::vulkan_sigmoid_mul_wrapper(const float* a, const float* b, float* out,
                                                 size_t n)
{
    void* pa = stage_input(a, n);
    void* pb = stage_input(b, n);
    void* pout = VulkanBackend::malloc(n * sizeof(float));
    VulkanBackend::launchSigmoidMul(pa, pb, pout, n);
    unstage_output(out, pout, n);
    free_buffer(pa);
    free_buffer(pb);
    free_buffer(pout);
}

void VulkanMoERunner::vulkan_elementwise_add_wrapper(const float* a, const float* b, float* out,
                                                     size_t n)
{
    void* pa = stage_input(a, n);
    void* pb = stage_input(b, n);
    void* pout = VulkanBackend::malloc(n * sizeof(float));
    VulkanBackend::launchElementwiseAdd(pa, pb, pout, n);
    unstage_output(out, pout, n);
    free_buffer(pa);
    free_buffer(pb);
    free_buffer(pout);
}

void VulkanMoERunner::vulkan_scale_rows_wrapper(const float* in, const float* scale, float* out,
                                                uint32_t rows, uint32_t cols)
{
    void* pin = stage_input(in, rows * cols);
    void* pscale = stage_input(scale, rows);
    void* pout = VulkanBackend::malloc(rows * cols * sizeof(float));
    VulkanBackend::launchScaleRows(pin, pscale, pout, rows, cols);
    unstage_output(out, pout, rows * cols);
    free_buffer(pin);
    free_buffer(pscale);
    free_buffer(pout);
}

void VulkanMoERunner::vulkan_cast_wrapper(const float* in, float* out, size_t n, int32_t dtype)
{
    // dtype 0 = fp32 (no-op), 1 = fp16, 2 = bf16
    if (dtype == 0) {
        std::memcpy(out, in, n * sizeof(float));
        return;
    }
    void* pin = stage_input(in, n);
    void* pout = VulkanBackend::malloc(n * sizeof(float));
    VulkanBackend::launchCast(pin, pout, n, dtype);
    unstage_output(out, pout, n);
    free_buffer(pin);
    free_buffer(pout);
}

void VulkanMoERunner::vulkan_index_add_wrapper(float* out, const uint32_t* indices,
                                               const float* values, uint32_t out_rows,
                                               uint32_t val_rows, uint32_t cols)
{
    void* pout = stage_input(out, out_rows * cols);
    void* pidx = stage_input(reinterpret_cast<const float*>(indices), val_rows);
    void* pval = stage_input(values, val_rows * cols);
    VulkanBackend::launchIndexAdd(pout, pidx, pval, out_rows, val_rows, cols);
    unstage_output(out, pout, out_rows * cols);
    free_buffer(pout);
    free_buffer(pidx);
    free_buffer(pval);
}

void VulkanMoERunner::vulkan_fill_wrapper(float* buf, float value, size_t n)
{
    void* pbuf = VulkanBackend::malloc(n * sizeof(float));
    VulkanBackend::launchFill(pbuf, value, n);
    unstage_output(buf, pbuf, n);
    free_buffer(pbuf);
}

void VulkanMoERunner::vulkan_gather_wrapper(const float* src, const uint32_t* indices,
                                            float* out, size_t num_indices, size_t elem_size)
{
    void* psrc = stage_input(src, elem_size);
    void* pidx = stage_input(reinterpret_cast<const float*>(indices), num_indices);
    void* pout = VulkanBackend::malloc(num_indices * sizeof(float));
    VulkanBackend::launchGather(psrc, pidx, pout, static_cast<uint32_t>(num_indices));
    unstage_output(out, pout, num_indices);
    free_buffer(psrc);
    free_buffer(pidx);
    free_buffer(pout);
}

void VulkanMoERunner::vulkan_compare_eq_wrapper(const float* input, uint32_t* output,
                                                float threshold, size_t n)
{
    void* pin = stage_input(input, n);
    void* pout = VulkanBackend::malloc(n * sizeof(uint32_t));
    VulkanBackend::launchCompareEq(pin, pout, threshold, n);
    VulkanBackend::memcpyDeviceToHost(output, pout, n * sizeof(uint32_t));
    free_buffer(pin);
    free_buffer(pout);
}

void VulkanMoERunner::run_gemm_profile(
    const float* x,
    const float* const* fc1_expert_weights,
    const float* const* fc1_expert_biases,
    const float* const* fc2_expert_weights,
    const float* const* fc2_expert_biases,
    uint32_t num_tokens,
    float* output)
{
    uint32_t hidden = mConfig.hidden_size;
    uint32_t intermediate = mConfig.intermediate_size;
    uint32_t num_experts = mConfig.num_experts;

    // Allocate results and fill with zeros
    std::vector<float> results(num_tokens * hidden, 0.0f);

    for (uint32_t e = 0; e < num_experts; ++e) {
        // FC1: [num_tokens, hidden] x [hidden, 2*intermediate] -> [num_tokens, 2*intermediate]
        std::vector<float> fc1_out(num_tokens * 2 * intermediate);
        const float* w1 = fc1_expert_weights[e];
        vulkan_gemm_wrapper(x, w1, fc1_out.data(), num_tokens, 2 * intermediate, hidden);

        // Add bias
        if (fc1_expert_biases && fc1_expert_biases[e]) {
            // Broadcast bias: [2*intermediate] -> [num_tokens, 2*intermediate]
            std::vector<float> bias_expanded(num_tokens * 2 * intermediate);
            for (uint32_t t = 0; t < num_tokens; ++t) {
                std::memcpy(&bias_expanded[t * 2 * intermediate],
                           fc1_expert_biases[e], 2 * intermediate * sizeof(float));
            }
            vulkan_elementwise_add_wrapper(fc1_out.data(), bias_expanded.data(),
                                           fc1_out.data(), num_tokens * 2 * intermediate);
        }

        // Activation: SwiGLU (split + sigmoid_mul) or ReLU
        std::vector<float> act_out(num_tokens * intermediate);
        if (mConfig.swiglu_activation) {
            // Chunk: split fc1_out [num_tokens, 2*intermediate] into up [num_tokens, intermediate]
            // and gate [num_tokens, intermediate]
            std::vector<float> up(num_tokens * intermediate);
            std::vector<float> gate(num_tokens * intermediate);
            for (uint32_t t = 0; t < num_tokens; ++t) {
                std::memcpy(&up[t * intermediate],
                           &fc1_out[t * 2 * intermediate],
                           intermediate * sizeof(float));
                std::memcpy(&gate[t * intermediate],
                           &fc1_out[t * 2 * intermediate + intermediate],
                           intermediate * sizeof(float));
            }
            vulkan_sigmoid_mul_wrapper(up.data(), gate.data(), act_out.data(),
                                       num_tokens * intermediate);
        } else {
            // ReLU
            std::vector<float> relu_out(num_tokens * 2 * intermediate);
            // Use vulkan_relu on the full tensor, then take first half
            void* pin = stage_input(fc1_out.data(), num_tokens * 2 * intermediate);
            void* pout = VulkanBackend::malloc(num_tokens * 2 * intermediate * sizeof(float));
            VulkanBackend::launchRelu(pin, pout, num_tokens * 2 * intermediate);
            unstage_output(relu_out.data(), pout, num_tokens * 2 * intermediate);
            free_buffer(pin);
            free_buffer(pout);
            std::memcpy(act_out.data(), relu_out.data(), num_tokens * intermediate * sizeof(float));
        }

        // FC2: [num_tokens, intermediate] x [intermediate, hidden] -> [num_tokens, hidden]
        std::vector<float> fc2_out(num_tokens * hidden);
        const float* w2 = fc2_expert_weights[e];
        vulkan_gemm_wrapper(act_out.data(), w2, fc2_out.data(), num_tokens, hidden, intermediate);

        // Add bias
        if (fc2_expert_biases && fc2_expert_biases[e]) {
            std::vector<float> bias_expanded(num_tokens * hidden);
            for (uint32_t t = 0; t < num_tokens; ++t) {
                std::memcpy(&bias_expanded[t * hidden], fc2_expert_biases[e], hidden * sizeof(float));
            }
            vulkan_elementwise_add_wrapper(fc2_out.data(), bias_expanded.data(),
                                           fc2_out.data(), num_tokens * hidden);
        }

        // Accumulate
        vulkan_elementwise_add_wrapper(results.data(), fc2_out.data(), results.data(),
                                       num_tokens * hidden);
    }

    std::memcpy(output, results.data(), num_tokens * hidden * sizeof(float));
}

void VulkanMoERunner::run_moe(
    const float* hidden_states,
    const float* routing_logits,
    const float* const* gemm1_weights,
    const float* const* gemm2_weights,
    const float* const* gemm1_biases,
    const float* const* gemm2_biases,
    uint32_t num_tokens,
    float* output)
{
    uint32_t hidden = mConfig.hidden_size;
    uint32_t num_experts = mConfig.num_experts;
    uint32_t top_k = mConfig.top_k;
    uint32_t intermediate = mConfig.intermediate_size;

    // Allocate output and fill with zeros
    std::fill(output, output + num_tokens * hidden, 0.0f);

    // Step 1: Routing — GEMM + softmax + topk
    // routing_logits: [num_tokens, num_experts]
    // gemm1_weights[0]: [num_experts, hidden] -> routing_scores: [num_tokens, num_experts]

    // If routing_logits is provided, run routing GEMM
    std::vector<float> routing_scores(num_tokens * num_experts);
    if (routing_logits) {
        // routing_scores = routing_logits * gemm1_weights[0].T
        // gemm1_weights[0]: [num_experts, hidden] -> weight for GEMM is [hidden, num_experts]^T
        // Output: [num_tokens, num_experts]
        vulkan_gemm_wrapper(routing_logits, gemm1_weights[0],
                           routing_scores.data(), num_tokens, num_experts, hidden);
    } else {
        // No routing logits, copy directly
        std::memcpy(routing_scores.data(), routing_logits, num_tokens * num_experts * sizeof(float));
    }

    // Apply routing bias if present (gemm1_biases[0] is used as routing bias)
    if (gemm1_biases && gemm1_biases[0]) {
        std::vector<float> bias_expanded(num_tokens * num_experts);
        for (uint32_t t = 0; t < num_tokens; ++t) {
            std::memcpy(&bias_expanded[t * num_experts], gemm1_biases[0], num_experts * sizeof(float));
        }
        vulkan_elementwise_add_wrapper(routing_scores.data(), bias_expanded.data(),
                                       routing_scores.data(), num_tokens * num_experts);
    }

    // Softmax over experts dimension
    vulkan_softmax_wrapper(routing_scores.data(), routing_scores.data(),
                           num_tokens, num_experts);

    // Top-K routing
    std::vector<uint32_t> topk_ids(num_tokens * top_k);
    std::vector<float> topk_weights(num_tokens * top_k);
    vulkan_topk_wrapper(routing_scores.data(), topk_ids.data(), topk_weights.data(),
                        num_tokens, num_experts, top_k);

    // Step 2: For each expert, process its assigned tokens
    for (uint32_t e = 0; e < num_experts; ++e) {
        // Find tokens assigned to this expert
        // mask[i] = (topk_ids[i] == e) — but topk_ids is [num_tokens, top_k]
        // Need to find which (token, k) pairs have expert == e

        // Create mask: [num_tokens * top_k] of uint32
        std::vector<uint32_t> mask(num_tokens * top_k);
        vulkan_compare_eq_wrapper(
            reinterpret_cast<const float*>(topk_ids.data()),
            mask.data(),
            static_cast<float>(e),
            num_tokens * top_k);

        // Count active tokens for this expert
        uint32_t active_count = 0;
        std::vector<uint32_t> token_row_indices;
        std::vector<uint32_t> token_k_indices;
        for (uint32_t i = 0; i < num_tokens * top_k; ++i) {
            if (mask[i]) {
                active_count++;
                token_row_indices.push_back(i / top_k);
                token_k_indices.push_back(i % top_k);
            }
        }

        if (active_count == 0) continue;

        // Gather expert tokens: hidden_states[token_row_indices]
        std::vector<float> expert_tokens(active_count * hidden);
        for (uint32_t i = 0; i < active_count; ++i) {
            std::memcpy(&expert_tokens[i * hidden],
                       &hidden_states[token_row_indices[i] * hidden],
                       hidden * sizeof(float));
        }

        // Gather scales: topk_weights[token_row_indices, token_k_indices]
        std::vector<float> scales(active_count);
        for (uint32_t i = 0; i < active_count; ++i) {
            scales[i] = topk_weights[token_row_indices[i] * top_k + token_k_indices[i]];
        }

        // FC1: expert_tokens * gemm1_weights[e+offset]
        // gemm1_weights[e]: [2*intermediate, hidden] -> output [active_count, 2*intermediate]
        std::vector<float> fc1_out(active_count * 2 * intermediate);
        const float* w1 = gemm1_weights[e];
        vulkan_gemm_wrapper(expert_tokens.data(), w1, fc1_out.data(),
                           active_count, 2 * intermediate, hidden);

        // Add bias
        if (gemm1_biases && gemm1_biases[e]) {
            std::vector<float> bias_expanded(active_count * 2 * intermediate);
            for (uint32_t t = 0; t < active_count; ++t) {
                std::memcpy(&bias_expanded[t * 2 * intermediate],
                           gemm1_biases[e], 2 * intermediate * sizeof(float));
            }
            vulkan_elementwise_add_wrapper(fc1_out.data(), bias_expanded.data(),
                                           fc1_out.data(), active_count * 2 * intermediate);
        }

        // Split + sigmoid_mul (SwiGLU)
        std::vector<float> act_out(active_count * intermediate);
        {
            std::vector<float> up(active_count * intermediate);
            std::vector<float> gate(active_count * intermediate);
            for (uint32_t t = 0; t < active_count; ++t) {
                std::memcpy(&up[t * intermediate],
                           &fc1_out[t * 2 * intermediate],
                           intermediate * sizeof(float));
                std::memcpy(&gate[t * intermediate],
                           &fc1_out[t * 2 * intermediate + intermediate],
                           intermediate * sizeof(float));
            }
            vulkan_sigmoid_mul_wrapper(up.data(), gate.data(), act_out.data(),
                                       active_count * intermediate);
        }

        // FC2: act_out * gemm2_weights[e+offset]
        // gemm2_weights[e]: [hidden, intermediate] -> output [active_count, hidden]
        std::vector<float> fc2_out(active_count * hidden);
        const float* w2 = gemm2_weights[e];
        vulkan_gemm_wrapper(act_out.data(), w2, fc2_out.data(),
                           active_count, hidden, intermediate);

        // Add bias
        if (gemm2_biases && gemm2_biases[e]) {
            std::vector<float> bias_expanded(active_count * hidden);
            for (uint32_t t = 0; t < active_count; ++t) {
                std::memcpy(&bias_expanded[t * hidden], gemm2_biases[e], hidden * sizeof(float));
            }
            vulkan_elementwise_add_wrapper(fc2_out.data(), bias_expanded.data(),
                                           fc2_out.data(), active_count * hidden);
        }

        // Scale rows: fc2_out * scales.unsqueeze(-1)
        std::vector<float> scaled(active_count * hidden);
        vulkan_scale_rows_wrapper(fc2_out.data(), scales.data(), scaled.data(),
                                  active_count, hidden);

        // IndexAdd: output[token_row_indices] += scaled
        vulkan_index_add_wrapper(output, token_row_indices.data(), scaled.data(),
                                num_tokens, active_count, hidden);
    }
}

} // namespace tensorrt_llm::common
