/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/vulkanBackend.h"
#include "tensorrt_llm/common/vulkanMoERunner.h"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>

using namespace tensorrt_llm::common;

static std::vector<float> makeRandom(uint32_t n, float scale = 1.0f)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(gen);
    return v;
}

static bool approxEqual(const std::vector<float>& a, const std::vector<float>& b,
                        float eps = 0.5f)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > eps) return false;
    }
    return true;
}

// Simple CPU reference MoE implementation for validation
static void cpuMoe(
    const std::vector<float>& hidden_states,
    const std::vector<float>& routing_logits,
    const std::vector<const float*>& gemm1_weights,
    const std::vector<const float*>& gemm2_weights,
    const std::vector<const float*>& fc1_biases,
    const std::vector<const float*>& fc2_biases,
    uint32_t num_tokens, uint32_t num_experts, uint32_t top_k,
    uint32_t hidden, uint32_t intermediate,
    std::vector<float>& output)
{
    output.assign(num_tokens * hidden, 0.0f);

    // Softmax + topk for routing
    for (uint32_t t = 0; t < num_tokens; ++t) {
        // Softmax over experts
        std::vector<float> scores(num_experts);
        float max_val = -1e30f;
        for (uint32_t e = 0; e < num_experts; ++e)
            max_val = std::max(max_val, routing_logits[t * num_experts + e]);
        float sum = 0;
        for (uint32_t e = 0; e < num_experts; ++e) {
            scores[e] = std::exp(routing_logits[t * num_experts + e] - max_val);
            sum += scores[e];
        }
        for (uint32_t e = 0; e < num_experts; ++e)
            scores[e] /= sum;

        // Top-k
        std::vector<std::pair<float, uint32_t>> ranked;
        for (uint32_t e = 0; e < num_experts; ++e)
            ranked.emplace_back(scores[e], e);
        std::sort(ranked.rbegin(), ranked.rend());
        ranked.resize(top_k);

        for (const auto& [score, e] : ranked) {
            // FC1
            std::vector<float> fc1(2 * intermediate, 0);
            for (uint32_t j = 0; j < 2 * intermediate; ++j) {
                float s = 0;
                for (uint32_t h = 0; h < hidden; ++h) {
                    s += hidden_states[t * hidden + h] * gemm1_weights[e][j * hidden + h];
                }
                if (fc1_biases[e]) s += fc1_biases[e][j];
                fc1[j] = s;
            }

            // SwiGLU
            std::vector<float> act(intermediate);
            for (uint32_t j = 0; j < intermediate; ++j) {
                float up = fc1[j];
                float gate = fc1[intermediate + j];
                act[j] = up * (1.0f / (1.0f + std::exp(-gate)));
            }

            // FC2
            for (uint32_t h = 0; h < hidden; ++h) {
                float s = 0;
                for (uint32_t j = 0; j < intermediate; ++j) {
                    s += act[j] * gemm2_weights[e][h * intermediate + j];
                }
                if (fc2_biases[e]) s += fc2_biases[e][h];
                output[t * hidden + h] += s * score;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    // Initialize Vulkan backend
    auto backend = VulkanBackend::getInstance();
    if (!backend->initialize(0)) {
        std::cerr << "Failed to initialize Vulkan backend: " << backend->getLastError() << std::endl;
        return 1;
    }
    std::cout << "Vulkan backend initialized successfully." << std::endl;

    // Configuration
    uint32_t num_tokens = 4;
    uint32_t num_experts = 4;
    uint32_t top_k = 2;
    uint32_t hidden = 8;
    uint32_t intermediate = 6;

    // Create runner
    VulkanMoERunner runner({
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden,
        .intermediate_size = intermediate,
        .swiglu_activation = true,
        .has_fc1_bias = true,
        .has_fc2_bias = true,
    });

    // Create test data
    auto hidden_states = makeRandom(num_tokens * hidden, 0.5f);
    auto routing_logits = makeRandom(num_tokens * num_experts, 1.0f);

    std::vector<const float*> w1_ptrs, w2_ptrs, b1_ptrs, b2_ptrs;
    std::vector<std::vector<float>> w1_data, w2_data, b1_data, b2_data;
    for (uint32_t e = 0; e < num_experts; ++e) {
        w1_data.emplace_back(makeRandom(2 * intermediate * hidden, 0.1f));
        w2_data.emplace_back(makeRandom(hidden * intermediate, 0.1f));
        b1_data.emplace_back(makeRandom(2 * intermediate, 0.0f));
        b2_data.emplace_back(makeRandom(hidden, 0.0f));
        w1_ptrs.push_back(w1_data.back().data());
        w2_ptrs.push_back(w2_data.back().data());
        b1_ptrs.push_back(b1_data.back().data());
        b2_ptrs.push_back(b2_data.back().data());
    }

    // Run Vulkan MoE
    std::vector<float> vulkan_output(num_tokens * hidden);
    runner.run_moe(
        hidden_states.data(),
        routing_logits.data(),
        w1_ptrs.data(), w2_ptrs.data(),
        b1_ptrs.data(), b2_ptrs.data(),
        num_tokens,
        vulkan_output.data());

    // Run CPU reference
    std::vector<float> cpu_output;
    cpuMoe(hidden_states, routing_logits,
           w1_ptrs, w2_ptrs, b1_ptrs, b2_ptrs,
           num_tokens, num_experts, top_k, hidden, intermediate,
           cpu_output);

    // Compare
    bool pass = approxEqual(cpu_output, vulkan_output, 1.0f);
    std::cout << "Vulkan output: ";
    for (uint32_t i = 0; i < std::min(num_tokens * hidden, 8u); ++i)
        std::cout << vulkan_output[i] << " ";
    std::cout << std::endl;
    std::cout << "CPU output:    ";
    for (uint32_t i = 0; i < std::min(num_tokens * hidden, 8u); ++i)
        std::cout << cpu_output[i] << " ";
    std::cout << std::endl;
    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;

    return pass ? 0 : 1;
}
