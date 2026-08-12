// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tensorrt_llm/common/gguf_loader.h"
#include "tensorrt_llm/common/vulkan_tokenizer.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>

namespace tllm::vk
{

// ============================================================
// CPU helpers (mirror vulkan_ops.py)
// ============================================================

// Precompute RoPE frequency table
void precompute_rope_freqs(
    int seq_len, int head_dim, float theta,
    std::vector<float>& cos_out, std::vector<float>& sin_out);

// Apply RoPE to Q/K: rotate half and combine with cos/sin
// q, k are (tokens, heads, head_dim), cos/sin are (seq, head_dim)
void apply_rope(
    std::vector<float>& q, std::vector<float>& k,
    const std::vector<float>& cos, const std::vector<float>& sin,
    int tokens, int heads, int head_dim, int position);

// CPU softmax
std::vector<float> softmax_cpu(const std::vector<float>& logits, int batch);

// CPU top-k sampling
int sample_top_k_top_p(
    const std::vector<float>& probs,
    int vocab_size, int top_k, float top_p,
    std::mt19937& rng);

// Helper: get tensor by name (const pointer)
const LoadedTensor* getTensor_ptr(const std::vector<LoadedTensor>& tensors, const std::string& name);

// Helper: transpose weight from (rows, cols) to (cols, rows)
std::vector<float> transposeWeight(const std::vector<float>& weight, int rows, int cols);

// ============================================================
// Qwen2 Inference Engine
// ============================================================

class Qwen2VulkanInference
{
public:
    Qwen2VulkanInference();
    ~Qwen2VulkanInference();

    // Load model from GGUF or safetensors
    bool loadModel(const std::string& model_path, bool is_gguf);

    // Generate tokens from prompt
    std::vector<int32_t> generate(
        const std::string& prompt,
        int max_tokens = 128,
        float temperature = 0.8f,
        int top_k = 50,
        float top_p = 0.9f,
        uint32_t seed = 42);

    // Get model info
    int vocabSize() const;

private:
    // Initialize / deinitialize Vulkan
    bool initVulkan();

    // Transformer forward for a single batch of tokens
    std::vector<float> forward(
        const std::vector<int32_t>& tokens,
        int position, bool is_prefill);

    // Single transformer block
    void transformerBlock(
        int layer_idx,
        const std::vector<float>& hidden_states,
        int tokens, int position,
        std::vector<float>& out);

    // Upload a host buffer to Vulkan device
    void* upload(const std::vector<float>& data);

    // Download from Vulkan device to host
    std::vector<float> download(void* ptr, size_t num_floats);

    // Free a device buffer
    void freeDevice(void* ptr);

    // Run GEMM: C[M,N] = A[M,K] * B[K,N]
    std::vector<float> runGemm(void* a_dev, void* b_dev, int M, int N, int K);

    // Run GEMM with a LoadedTensor weight (transposes weight from (N,K) to (K,N))
    std::vector<float> runGemmWithWeight(void* a_dev, const LoadedTensor* weight, int M, int N, int K);

    // RMS norm helper
    void rmsNorm(const std::vector<float>& x, const std::vector<float>& weight,
                 float eps, int token_count, int hidden_dim,
                 std::vector<float>& out);

    // Debug helper
    void dbgDump(const std::string& label, const std::vector<float>& vals, int n = 10);

    // Model metadata
    int m_n_layers = 0;
    int m_hidden_dim = 0;
    int m_n_heads = 0;
    int m_n_kv_heads = 0;
    int m_head_dim = 0;
    int m_intermediate_dim = 0;
    int m_vocab_size = 0;
    float m_norm_eps = 1e-6f;
    float m_rope_theta = 10000.0f;
    int m_n_rep = 1;

    // RoPE tables
    std::vector<float> m_cos_table;
    std::vector<float> m_sin_table;
    int m_max_seq_len = 2048;

    // KV cache (host-side)
    std::vector<std::vector<float>> m_kv_cache_k;
    std::vector<std::vector<float>> m_kv_cache_v;
    int m_seq_len = 0;

    // Tokenizer
    std::unique_ptr<GgufTokenizer> m_tokenizer;

    // Random number generator
    std::mt19937 m_rng;

    // Model weights and state
    std::vector<LoadedTensor> m_weights;
    bool m_is_gguf = false;

    // GPU weight cache: maps weight name to device pointer (avoids re-upload per token)
    std::unordered_map<std::string, void*> m_weightCache;

    // Cached transposed lm_head weights (CPU-side, avoids re-transpose per token)
    std::vector<float> m_lmHeadT;
};

} // namespace tllm::vk