// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorrt_llm/common/vulkan_inference.h"
#include "tensorrt_llm/common/vulkanBackend.h"
#include "tensorrt_llm/common/safetensors_loader.h"
#include "tensorrt_llm/common/vulkanCommon.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>

namespace tllm::vk
{

using namespace tensorrt_llm::common;

// ============================================================
// CPU helpers
// ============================================================

void precompute_rope_freqs(
    int seq_len, int head_dim, float theta,
    std::vector<float>& cos_out, std::vector<float>& sin_out)
{
    cos_out.resize(static_cast<size_t>(seq_len) * head_dim);
    sin_out.resize(static_cast<size_t>(seq_len) * head_dim);

    // Generate head_dim frequencies (standard: 1/theta^(d/head_dim) for even d)
    std::vector<float> freqs(head_dim);
    for (int i = 0; i < head_dim; i += 2)
    {
        float freq = 1.0f / std::pow(theta, static_cast<float>(i) / head_dim);
        freqs[i]     = freq;
        freqs[i + 1] = freq;
    }

    for (int t = 0; t < seq_len; t++)
    {
        float t_f = static_cast<float>(t);
        for (int d = 0; d < head_dim; d++)
        {
            float angle = t_f * freqs[d];
            cos_out[static_cast<size_t>(t) * head_dim + d] = std::cos(angle);
            sin_out[static_cast<size_t>(t) * head_dim + d] = std::sin(angle);
        }
    }
}

void apply_rope(
    std::vector<float>& q, std::vector<float>& k,
    const std::vector<float>& cos, const std::vector<float>& sin,
    int tokens, int heads, int head_dim, int position)
{
    int half = head_dim / 2;
    for (int t = 0; t < tokens; t++)
    {
        int pos = position + t;
        for (int h = 0; h < heads; h++)
        {
            float* q_row = &q[(static_cast<size_t>(t) * heads + h) * head_dim];
            float* k_row = &k[(static_cast<size_t>(t) * heads + h) * head_dim];
            const float* c = &cos[static_cast<size_t>(pos) * head_dim];
            const float* s = &sin[static_cast<size_t>(pos) * head_dim];

            for (int d = 0; d < half; d++)
            {
                float q1 = q_row[d];
                float q2 = q_row[half + d];
                float k1 = k_row[d];
                float k2 = k_row[half + d];

                q_row[d]         = q1 * c[d] - q2 * s[d];
                q_row[half + d]  = q1 * s[d] + q2 * c[d];
                k_row[d]         = k1 * c[d] - k2 * s[d];
                k_row[half + d]  = k1 * s[d] + k2 * c[d];
            }
        }
    }
}

std::vector<float> softmax_cpu(const std::vector<float>& logits, int batch)
{
    std::vector<float> probs(logits.size());
    size_t row_size = logits.size() / batch;
    for (int b = 0; b < batch; b++)
    {
        const float* row = &logits[static_cast<size_t>(b) * row_size];
        float max_val = row[0];
        for (size_t i = 1; i < row_size; i++)
            max_val = std::max(max_val, row[i]);
        float sum = 0.0f;
        for (size_t i = 0; i < row_size; i++)
        {
            float e = std::exp(row[i] - max_val);
            probs[static_cast<size_t>(b) * row_size + i] = e;
            sum += e;
        }
        for (size_t i = 0; i < row_size; i++)
            probs[static_cast<size_t>(b) * row_size + i] /= sum;
    }
    return probs;
}

int sample_top_k_top_p(
    const std::vector<float>& probs,
    int vocab_size, int top_k, float top_p,
    std::mt19937& rng)
{
    std::vector<int> indices(vocab_size);
    for (int i = 0; i < vocab_size; i++) indices[i] = i;

    // top_k <= 0: disable filtering (sample from full vocab)
    int k = top_k <= 0 ? vocab_size : std::min(top_k, vocab_size);
    if (k < 1) k = 1;

    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b) { return probs[a] > probs[b]; });

    std::vector<int> selected(indices.begin(), indices.begin() + k);

    // Top-p filtering
    if (top_p > 0.0f && top_p < 1.0f)
    {
        float cum = 0.0f;
        std::vector<int> filtered;
        for (int idx : selected)
        {
            cum += probs[idx];
            filtered.push_back(idx);
            if (cum >= top_p) break;
        }
        selected = filtered;
    }

    // Renormalize and apply temperature
    std::vector<float> weights(selected.size());
    float sum = 0.0f;
    for (size_t i = 0; i < selected.size(); i++)
    {
        weights[i] = probs[selected[i]];
        sum += weights[i];
    }
    for (auto& w : weights) w /= sum;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float acc = 0.0f;
    for (size_t i = 0; i < selected.size(); i++)
    {
        acc += weights[i];
        if (r <= acc) return selected[i];
    }
    return selected.back();
}

const LoadedTensor* getTensor_ptr(const std::vector<LoadedTensor>& tensors, const std::string& name)
{
    for (const auto& t : tensors)
        if (t.name == name) return &t;
    return nullptr;
}

std::vector<float> transposeWeight(const std::vector<float>& weight, int rows, int cols)
{
    std::vector<float> result(static_cast<size_t>(cols) * rows);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            result[static_cast<size_t>(c) * rows + r] = weight[static_cast<size_t>(r) * cols + c];
    return result;
}

// ============================================================
// Qwen2VulkanInference
// ============================================================

Qwen2VulkanInference::Qwen2VulkanInference()
    : m_rng(42)
{
}

Qwen2VulkanInference::~Qwen2VulkanInference()
{
    for (auto& w : m_weights)
    {
        if (w.buffer)
            VulkanBackend::free(w.buffer);
    }
}

bool Qwen2VulkanInference::initVulkan()
{
    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive())
        return backend->initialize(0);
    return true;
}

bool Qwen2VulkanInference::loadModel(const std::string& model_path, bool is_gguf)
{
    if (!initVulkan())
    {
        std::cerr << "[ERROR] Failed to initialize Vulkan backend\n";
        return false;
    }

    m_is_gguf = is_gguf;

    if (is_gguf)
    {
        GgufModelLoader loader;
        loader.setVulkanRuntime(VulkanRuntime::getInstance().get());
        if (!loader.load(model_path))
        {
            std::cerr << "[ERROR] Failed to load GGUF model: " << model_path << "\n";
            return false;
        }

        const auto& meta = loader.getMeta();
        m_n_layers = meta.n_layers;
        m_hidden_dim = meta.hidden_dim;
        m_n_heads = meta.n_heads;
        m_n_kv_heads = meta.n_kv_heads;
        m_head_dim = meta.head_dim;
        m_intermediate_dim = meta.intermediate_dim;
        m_vocab_size = meta.vocab_size;
        m_norm_eps = meta.norm_eps;
        m_rope_theta = meta.rope_theta;

    m_tokenizer = std::make_unique<GgufTokenizer>();
    m_tokenizer->load(model_path);

        m_weights = loader.getTensors();
    }
    else
    {
        SafetensorsModelLoader loader;
        loader.setVulkanRuntime(VulkanRuntime::getInstance().get());
        if (!loader.load(model_path))
        {
            std::cerr << "[ERROR] Failed to load safetensors model: " << model_path << "\n";
            return false;
        }

        const auto& meta = loader.getMeta();
        m_n_layers = meta.n_layers;
        m_hidden_dim = meta.hidden_dim;
        m_n_heads = meta.n_heads;
        m_n_kv_heads = meta.n_kv_heads;
        m_head_dim = meta.head_dim;
        m_intermediate_dim = meta.intermediate_dim;
        m_vocab_size = meta.vocab_size;
        m_norm_eps = meta.norm_eps;
        m_rope_theta = meta.rope_theta;

        m_weights = loader.getTensors();
    }

    // Check for NaN/Inf in weights
    int nan_count = 0;
    int inf_count = 0;
    for (const auto& w : m_weights)
    {
        int local_nan = 0, local_inf = 0;
        for (float val : w.data)
        {
            if (std::isnan(val)) local_nan++;
            if (std::isinf(val)) local_inf++;
        }
        if (local_nan > 0 || local_inf > 0)
        {
            std::cerr << "[WARN] Weight '" << w.name << "': NaN=" << local_nan << " Inf=" << local_inf << " size=" << w.data.size() << "\n";
            nan_count += local_nan;
            inf_count += local_inf;
        }
    }
    if (nan_count > 0 || inf_count > 0)
    {
        std::cerr << "[WARN] Total: NaN=" << nan_count << " Inf=" << inf_count << " across all weights\n";
    }

    if (m_n_kv_heads > 0)
        m_n_rep = m_n_heads / m_n_kv_heads;
    else
        m_n_rep = 1;

    // Infer vocab_size from token_embd.weight if not set from metadata
    if (m_vocab_size == 0)
    {
        const LoadedTensor* emb = getTensor_ptr(m_weights,
            m_is_gguf ? "token_embd.weight" : "model.embed_tokens.weight");
        if (emb && emb->shape.size() >= 2)
        {
            if (m_is_gguf)
                // GGUF shape is [hidden, vocab] (ggml reversed)
                m_vocab_size = static_cast<int>(emb->shape[emb->shape.size() - 1]);
            else
                // Safetensors shape is [vocab, hidden]
                m_vocab_size = static_cast<int>(emb->shape[0]);
        }
    }

    m_max_seq_len = 4096;
    precompute_rope_freqs(m_max_seq_len, m_head_dim, m_rope_theta, m_cos_table, m_sin_table);

    // Allocate KV cache
    int kv_dim = m_n_kv_heads * m_head_dim;
    m_kv_cache_k.resize(m_n_layers, std::vector<float>(static_cast<size_t>(m_max_seq_len) * kv_dim, 0.0f));
    m_kv_cache_v.resize(m_n_layers, std::vector<float>(static_cast<size_t>(m_max_seq_len) * kv_dim, 0.0f));

    std::cout << "[INFO] Model loaded: " << m_n_layers << " layers, hidden=" << m_hidden_dim
              << ", heads=" << m_n_heads << ", kv_heads=" << m_n_kv_heads
              << ", head_dim=" << m_head_dim << ", vocab=" << m_vocab_size
              << ", rope_theta=" << m_rope_theta << "\n";

    return true;
}

int Qwen2VulkanInference::vocabSize() const { return m_vocab_size; }

void* Qwen2VulkanInference::upload(const std::vector<float>& data)
{
    size_t bytes = data.size() * sizeof(float);
    void* ptr = VulkanBackend::malloc(bytes);
    if (!ptr)
    {
        std::cerr << "[ERROR] Failed to allocate " << bytes << " bytes on Vulkan device\n";
        return nullptr;
    }
    VulkanBackend::memcpyHostToDevice(ptr, data.data(), bytes);
    return ptr;
}

std::vector<float> Qwen2VulkanInference::download(void* ptr, size_t num_floats)
{
    std::vector<float> result(num_floats);
    VulkanBackend::memcpyDeviceToHost(result.data(), ptr, num_floats * sizeof(float));
    return result;
}

void Qwen2VulkanInference::freeDevice(void* ptr)
{
    if (ptr) VulkanBackend::free(ptr);
}

std::vector<float> Qwen2VulkanInference::runGemm(void* a_dev, void* b_dev, int M, int N, int K)
{
    void* out_dev = VulkanBackend::malloc(static_cast<size_t>(M) * N * sizeof(float));
    VulkanBackend::launchFp16Gemm(a_dev, b_dev, out_dev,
        static_cast<uint32_t>(M), static_cast<uint32_t>(N), static_cast<uint32_t>(K));
    std::vector<float> result = download(out_dev, static_cast<size_t>(M) * N);
    freeDevice(out_dev);
    return result;
}

std::vector<float> Qwen2VulkanInference::runGemmWithWeight(void* a_dev, const LoadedTensor* weight, int M, int N, int K)
{
    if (weight == nullptr)
    {
        std::cerr << "[ERROR] null weight in GEMM\n";
        return {};
    }

    // Both GGUF and Safetensors store weights in PyTorch row-major (N, K) layout.
    // Transpose to (K, N) for the GEMM shader which reads b[k*N + c].
    void* b_dev;
    std::string cacheKey = weight->name + ":t";
    auto it = m_weightCache.find(cacheKey);
    if (it != m_weightCache.end())
    {
        b_dev = it->second;
    }
    else
    {
        std::vector<float> weight_t = transposeWeight(weight->data, N, K);
        b_dev = upload(weight_t);
        m_weightCache[cacheKey] = b_dev;
    }

    std::vector<float> result = runGemm(a_dev, b_dev, M, N, K);
    return result;
}

void Qwen2VulkanInference::rmsNorm(
    const std::vector<float>& x, const std::vector<float>& weight,
    float eps, int token_count, int hidden_dim,
    std::vector<float>& out)
{
    out.resize(x.size());
    for (int t = 0; t < token_count; t++)
    {
        const float* row = &x[static_cast<size_t>(t) * hidden_dim];
        float* out_row = &out[static_cast<size_t>(t) * hidden_dim];

        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_dim; i++)
            sum_sq += row[i] * row[i];
        float rms = std::sqrt(sum_sq / hidden_dim + eps);
        float inv_rms = 1.0f / rms;

        for (int i = 0; i < hidden_dim; i++)
            out_row[i] = row[i] * inv_rms * weight[i];
    }
}

void Qwen2VulkanInference::dbgDump(const std::string& label, const std::vector<float>& vals, int n)
{
    std::cout << "[DBG] " << label << " first " << std::min(n, (int)vals.size()) << ": ";
    for (int i = 0; i < std::min(n, (int)vals.size()); i++)
        std::cout << vals[i] << " ";
    std::cout << "\n";
}

void Qwen2VulkanInference::transformerBlock(
    int layer_idx,
    const std::vector<float>& hidden_states,
    int tokens, int position,
    std::vector<float>& out)
{
    std::string prefix = m_is_gguf
        ? ("blk." + std::to_string(layer_idx))
        : ("model.layers." + std::to_string(layer_idx));

    // --- Input RMS Norm ---
    std::string norm_key = m_is_gguf ? (prefix + ".attn_norm.weight")
                                     : (prefix + ".input_layernorm.weight");
    const LoadedTensor* norm_weight = getTensor_ptr(m_weights, norm_key);
    if (!norm_weight)
    {
        std::cerr << "[ERROR] Missing " << norm_key << "\n";
        out = hidden_states;
        return;
    }

    std::vector<float> x_norm;
    rmsNorm(hidden_states, norm_weight->data, m_norm_eps, tokens, m_hidden_dim, x_norm);

    if (layer_idx == 0) {
        dbgDump("hidden_states[0]", hidden_states, 10);
        dbgDump("x_norm[0]", x_norm, 10);
    }

    // --- QKV Projections ---
    const LoadedTensor* q_weight = getTensor_ptr(m_weights, prefix + ".attn_q.weight");
    const LoadedTensor* k_weight = getTensor_ptr(m_weights, prefix + ".attn_k.weight");
    const LoadedTensor* v_weight = getTensor_ptr(m_weights, prefix + ".attn_v.weight");

    std::string q_key = m_is_gguf ? (prefix + ".attn_q.weight") : (prefix + ".self_attention.q_proj.weight");
    std::string k_key = m_is_gguf ? (prefix + ".attn_k.weight") : (prefix + ".self_attention.k_proj.weight");
    std::string v_key = m_is_gguf ? (prefix + ".attn_v.weight") : (prefix + ".self_attention.v_proj.weight");

    q_weight = getTensor_ptr(m_weights, q_key);
    k_weight = getTensor_ptr(m_weights, k_key);
    v_weight = getTensor_ptr(m_weights, v_key);

    void* x_dev = upload(x_norm);

    std::vector<float> q_data, k_data, v_data;

    if (q_weight && k_weight && v_weight)
    {
        q_data = runGemmWithWeight(x_dev, q_weight, tokens, m_n_heads * m_head_dim, m_hidden_dim);
        k_data = runGemmWithWeight(x_dev, k_weight, tokens, m_n_kv_heads * m_head_dim, m_hidden_dim);
        v_data = runGemmWithWeight(x_dev, v_weight, tokens, m_n_kv_heads * m_head_dim, m_hidden_dim);

        // Add QKV biases if present
        std::string q_bias_key = prefix + ".attn_q.bias";
        std::string k_bias_key = prefix + ".attn_k.bias";
        std::string v_bias_key = prefix + ".attn_v.bias";
        const LoadedTensor* q_bias = getTensor_ptr(m_weights, q_bias_key);
        const LoadedTensor* k_bias = getTensor_ptr(m_weights, k_bias_key);
        const LoadedTensor* v_bias = getTensor_ptr(m_weights, v_bias_key);
        if (q_bias && q_bias->data.size() == static_cast<size_t>(m_n_heads * m_head_dim))
        {
            for (int t = 0; t < tokens; t++)
                for (int h = 0; h < m_n_heads * m_head_dim; h++)
                    q_data[t * m_n_heads * m_head_dim + h] += q_bias->data[h];
        }
        if (k_bias && k_bias->data.size() == static_cast<size_t>(m_n_kv_heads * m_head_dim))
        {
            for (int t = 0; t < tokens; t++)
                for (int h = 0; h < m_n_kv_heads * m_head_dim; h++)
                    k_data[t * m_n_kv_heads * m_head_dim + h] += k_bias->data[h];
        }
        if (v_bias && v_bias->data.size() == static_cast<size_t>(m_n_kv_heads * m_head_dim))
        {
            for (int t = 0; t < tokens; t++)
                for (int h = 0; h < m_n_kv_heads * m_head_dim; h++)
                    v_data[t * m_n_kv_heads * m_head_dim + h] += v_bias->data[h];
        }

        if (layer_idx == 0) {
            dbgDump("q_data[0] after bias", q_data, 10);
            dbgDump("k_data[0] after bias", k_data, 10);
            dbgDump("v_data[0] after bias", v_data, 10);
        }
    }
    else
    {
        // Try fused QKV
        std::string qkv_key = m_is_gguf ? (prefix + ".attn_qkv.weight") : (prefix + ".self_attention.query_key_value.weight");
        const LoadedTensor* qkv_weight = getTensor_ptr(m_weights, qkv_key);
        if (qkv_weight)
        {
            int q_size = m_n_heads * m_head_dim;
            int kv_size = m_n_kv_heads * m_head_dim;
            int total = q_size + 2 * kv_size;

            // Weight is (total, hidden_dim) — split by rows then transpose each part
            std::vector<float> q_part(qkv_weight->data.begin(),
                qkv_weight->data.begin() + static_cast<size_t>(q_size) * m_hidden_dim);
            LoadedTensor q_tensor{".q", {q_size, m_hidden_dim}, q_part, 0};
            q_data = runGemmWithWeight(x_dev, &q_tensor, tokens, q_size, m_hidden_dim);

            std::vector<float> k_part(qkv_weight->data.begin() + static_cast<size_t>(q_size) * m_hidden_dim,
                qkv_weight->data.begin() + static_cast<size_t>(q_size + kv_size) * m_hidden_dim);
            LoadedTensor k_tensor{".k", {kv_size, m_hidden_dim}, k_part, 0};
            k_data = runGemmWithWeight(x_dev, &k_tensor, tokens, kv_size, m_hidden_dim);

            std::vector<float> v_part(qkv_weight->data.begin() + static_cast<size_t>(q_size + kv_size) * m_hidden_dim,
                qkv_weight->data.begin() + static_cast<size_t>(total) * m_hidden_dim);
            LoadedTensor v_tensor{".v", {kv_size, m_hidden_dim}, v_part, 0};
            v_data = runGemmWithWeight(x_dev, &v_tensor, tokens, kv_size, m_hidden_dim);
        }
        else
        {
            std::cerr << "[ERROR] No QKV weights found for layer " << layer_idx << "\n";
            out = hidden_states;
            return;
        }
    }

    freeDevice(x_dev);

    // --- RoPE ---
    apply_rope(q_data, q_data, m_cos_table, m_sin_table,
        tokens, m_n_heads, m_head_dim, position);
    apply_rope(k_data, k_data, m_cos_table, m_sin_table,
        tokens, m_n_kv_heads, m_head_dim, position);

    // --- KV Cache update ---
    int kv_dim = m_n_kv_heads * m_head_dim;
    for (int t = 0; t < tokens; t++)
    {
        int pos = position + t;
        if (pos >= m_max_seq_len) pos = m_max_seq_len - 1;
        std::memcpy(&m_kv_cache_k[layer_idx][static_cast<size_t>(pos) * kv_dim],
            &k_data[static_cast<size_t>(t) * kv_dim],
            static_cast<size_t>(kv_dim) * sizeof(float));
        std::memcpy(&m_kv_cache_v[layer_idx][static_cast<size_t>(pos) * kv_dim],
            &v_data[static_cast<size_t>(t) * kv_dim],
            static_cast<size_t>(kv_dim) * sizeof(float));
    }

    int current_seq = position + tokens;
    if (current_seq > m_max_seq_len) current_seq = m_max_seq_len;

    // --- GQA expand K/V ---
    // Layout: [head][seq][head_dim] (head-major) to match attention shader
    std::vector<float> k_full(static_cast<size_t>(m_n_heads) * current_seq * m_head_dim);
    std::vector<float> v_full(static_cast<size_t>(m_n_heads) * current_seq * m_head_dim);
    for (int h = 0; h < m_n_heads; h++)
    {
        int src_h = h / m_n_rep;
        for (int t = 0; t < current_seq; t++)
        {
            std::memcpy(&k_full[(static_cast<size_t>(h) * current_seq + t) * m_head_dim],
                &m_kv_cache_k[layer_idx][(static_cast<size_t>(t) * m_n_kv_heads + src_h) * m_head_dim],
                static_cast<size_t>(m_head_dim) * sizeof(float));
            std::memcpy(&v_full[(static_cast<size_t>(h) * current_seq + t) * m_head_dim],
                &m_kv_cache_v[layer_idx][(static_cast<size_t>(t) * m_n_kv_heads + src_h) * m_head_dim],
                static_cast<size_t>(m_head_dim) * sizeof(float));
        }
    }

    // --- Attention ---
    // Rearrange Q from [tokens][heads][dim] to [heads][tokens][dim] for shader
    std::vector<float> q_rearranged(static_cast<size_t>(m_n_heads) * tokens * m_head_dim);
    for (int h = 0; h < m_n_heads; h++)
        for (int t = 0; t < tokens; t++)
            std::memcpy(&q_rearranged[(static_cast<size_t>(h) * tokens + t) * m_head_dim],
                &q_data[(static_cast<size_t>(t) * m_n_heads + h) * m_head_dim],
                static_cast<size_t>(m_head_dim) * sizeof(float));

    void* q_dev = upload(q_rearranged);
    void* k_dev = upload(k_full);
    void* v_dev = upload(v_full);
    void* attn_out_dev = VulkanBackend::malloc(
        static_cast<size_t>(m_n_heads) * tokens * m_head_dim * sizeof(float));

    VulkanBackend::launchAttention(
        q_dev, k_dev, v_dev, attn_out_dev,
        1, static_cast<uint32_t>(m_n_heads),
        static_cast<uint32_t>(tokens),
        static_cast<uint32_t>(current_seq),
        static_cast<uint32_t>(m_head_dim), true);

    // Download attention output and rearrange from [heads][tokens][dim] back to [tokens][heads*dim]
    std::vector<float> attn_out_raw = download(attn_out_dev,
        static_cast<size_t>(m_n_heads) * tokens * m_head_dim);
    std::vector<float> attn_out(static_cast<size_t>(tokens) * m_n_heads * m_head_dim);
    for (int h = 0; h < m_n_heads; h++)
        for (int t = 0; t < tokens; t++)
            std::memcpy(&attn_out[(static_cast<size_t>(t) * m_n_heads + h) * m_head_dim],
                &attn_out_raw[(static_cast<size_t>(h) * tokens + t) * m_head_dim],
                static_cast<size_t>(m_head_dim) * sizeof(float));

    freeDevice(q_dev);
    freeDevice(k_dev);
    freeDevice(v_dev);
    freeDevice(attn_out_dev);

    // --- Output projection ---
    std::string o_key = m_is_gguf ? (prefix + ".attn_output.weight") : (prefix + ".self_attention.dense.weight");
    const LoadedTensor* o_weight = getTensor_ptr(m_weights, o_key);

    void* attn_dev = upload(attn_out);
    std::vector<float> attn_proj = runGemmWithWeight(attn_dev, o_weight, tokens, m_hidden_dim, m_hidden_dim);
    freeDevice(attn_dev);

    if (layer_idx == 0) {
        dbgDump("layer0_attn_proj[1]", attn_proj, 10);
    }

    // --- Residual ---
    out.resize(tokens * m_hidden_dim);
    for (size_t i = 0; i < out.size(); i++)
        out[i] = hidden_states[i] + attn_proj[i];

    // --- MLP ---
    std::string post_norm_key = m_is_gguf ? (prefix + ".ffn_norm.weight")
                                           : (prefix + ".post_attention_layernorm.weight");
    const LoadedTensor* post_norm = getTensor_ptr(m_weights, post_norm_key);

    std::vector<float> mlp_norm;
    rmsNorm(out, post_norm->data, m_norm_eps, tokens, m_hidden_dim, mlp_norm);

    std::string gate_key = m_is_gguf ? (prefix + ".ffn_gate.weight") : (prefix + ".mlp.gate_proj.weight");
    std::string up_key   = m_is_gguf ? (prefix + ".ffn_up.weight") : (prefix + ".mlp.up_proj.weight");
    std::string down_key = m_is_gguf ? (prefix + ".ffn_down.weight") : (prefix + ".mlp.down_proj.weight");

    const LoadedTensor* gate_weight = getTensor_ptr(m_weights, gate_key);
    const LoadedTensor* up_weight   = getTensor_ptr(m_weights, up_key);
    const LoadedTensor* down_weight = getTensor_ptr(m_weights, down_key);

    void* mlp_norm_dev = upload(mlp_norm);

    std::vector<float> gate_out = runGemmWithWeight(mlp_norm_dev, gate_weight, tokens, m_intermediate_dim, m_hidden_dim);
    std::vector<float> up_out   = runGemmWithWeight(mlp_norm_dev, up_weight, tokens, m_intermediate_dim, m_hidden_dim);
    freeDevice(mlp_norm_dev);

    // SwiGLU: out[i] = up[i] * silu(gate[i])
    void* gate_dev = upload(gate_out);
    void* silu_out_dev = VulkanBackend::malloc(gate_out.size() * sizeof(float));
    VulkanBackend::launchSilu(gate_dev, silu_out_dev, gate_out.size());
    std::vector<float> gate_silu = download(silu_out_dev, gate_out.size());
    freeDevice(gate_dev);
    freeDevice(silu_out_dev);

    std::vector<float> swish_out(gate_out.size());
    for (size_t i = 0; i < swish_out.size(); i++)
        swish_out[i] = up_out[i] * gate_silu[i];

    if (layer_idx == 0) {
        dbgDump("layer0_swish[1]", swish_out, 10);
        dbgDump("layer0_down_wt first 10", down_weight->data, 10);
    }

    void* swish_dev = upload(swish_out);
    std::vector<float> mlp_result = runGemmWithWeight(swish_dev, down_weight, tokens, m_hidden_dim, m_intermediate_dim);
    freeDevice(swish_dev);

    if (layer_idx == 0) {
        dbgDump("layer0_gate_out[1]", gate_out, 10);
        dbgDump("layer0_up_out[1]", up_out, 10);
        dbgDump("layer0_mlp_result[1]", mlp_result, 10);
    }

for (size_t i = 0; i < out.size(); i++)
        out[i] = out[i] + mlp_result[i];

    if (layer_idx == 0) {
        dbgDump("layer0_out[1] (after attn+mlp)", out, 10);
    }
}

std::vector<float> Qwen2VulkanInference::forward(
    const std::vector<int32_t>& tokens,
    int position, bool is_prefill)
{
    int num_tokens = static_cast<int>(tokens.size());

    // Token embedding
    const LoadedTensor* emb = getTensor_ptr(m_weights,
        m_is_gguf ? "token_embd.weight" : "model.embed_tokens.weight");
    if (!emb)
    {
        std::cerr << "[ERROR] Missing token embedding\n";
        return {};
    }

    std::vector<float> hidden_states(static_cast<size_t>(num_tokens) * m_hidden_dim);
    for (int t = 0; t < num_tokens; t++)
    {
        int32_t tid = tokens[t];
        if (tid < 0 || tid >= m_vocab_size) tid = 0;
        std::memcpy(&hidden_states[static_cast<size_t>(t) * m_hidden_dim],
            &emb->data[static_cast<size_t>(tid) * m_hidden_dim],
            static_cast<size_t>(m_hidden_dim) * sizeof(float));
    }

    // Transformer blocks
    for (int i = 0; i < m_n_layers; i++)
    {
        std::vector<float> block_out;
        transformerBlock(i, hidden_states, num_tokens, position, block_out);
        hidden_states = std::move(block_out);
    }

    // Final RMS Norm
    std::string final_norm_key = m_is_gguf ? "output_norm.weight" : "model.norm.weight";
    const LoadedTensor* final_norm = getTensor_ptr(m_weights, final_norm_key);
    if (!final_norm)
    {
        std::cerr << "[ERROR] Missing final norm\n";
        return {};
    }

    std::vector<float> normed;
    rmsNorm(hidden_states, final_norm->data, m_norm_eps, num_tokens, m_hidden_dim, normed);

    // LM head logits (last token)
    const LoadedTensor* lm_head = getTensor_ptr(m_weights, "output.weight");
    if (!lm_head)
    {
        if (m_is_gguf)
            lm_head = getTensor_ptr(m_weights, "token_embd.weight");
        else
            lm_head = getTensor_ptr(m_weights, "lm_head.weight");
    }
    if (!lm_head)
        lm_head = getTensor_ptr(m_weights,
            m_is_gguf ? "token_embd.weight" : "model.embed_tokens.weight");

    if (m_vocab_size == 0 && lm_head && lm_head->shape.size() >= 2)
    {
        // GGUF: shape is reversed [hidden, vocab] -> last element is vocab
        // Safetensors: shape is [vocab, hidden] -> last element is hidden, need shape[0]
        if (m_is_gguf)
            m_vocab_size = static_cast<int>(lm_head->shape[lm_head->shape.size() - 1]);
        else
            m_vocab_size = static_cast<int>(lm_head->shape[0]);
    }

    int vocab = m_vocab_size;
    if (vocab == 0) vocab = 32000;

    // Get last token's hidden state
    std::vector<float> last_hidden(m_hidden_dim);
    std::memcpy(last_hidden.data(),
        &normed[static_cast<size_t>(num_tokens - 1) * m_hidden_dim],
        static_cast<size_t>(m_hidden_dim) * sizeof(float));

    // GEMM: (1, hidden) x (hidden, vocab) -> (1, vocab)
    void* last_dev = upload(last_hidden);

    std::vector<float> logits(vocab);

    // Split lm_head GEMM into chunks to avoid AMD GPU TDR timeout on large
    // (>200MB) dispatch. Each chunk processes at most 16K output columns,
    // reading ~57MB from B per GEMM — well within GPU timeout limits.
    const int CHUNK_SIZE = 16384;

    // Cache transposed weights to avoid re-transposing on every forward pass
    if (m_lmHeadT.empty())
    {
        m_lmHeadT = transposeWeight(lm_head->data, vocab, m_hidden_dim);
    }
    std::vector<float>& lm_head_t = m_lmHeadT;

    int offset = 0;
    while (offset < vocab) {
        int chunk = std::min(CHUNK_SIZE, vocab - offset);

        // Extract B chunk: columns [offset, offset+chunk) of transposed weights
        // lm_head_t is (hidden, vocab) row-major, so column c is at indices [k*vocab + c]
        std::vector<float> b_chunk(static_cast<size_t>(m_hidden_dim) * chunk);
        for (int k = 0; k < m_hidden_dim; k++) {
            std::memcpy(&b_chunk[static_cast<size_t>(k) * chunk],
                        &lm_head_t[static_cast<size_t>(k) * vocab + offset],
                        static_cast<size_t>(chunk) * sizeof(float));
        }

        void* b_dev = upload(b_chunk);
        void* out_dev = VulkanBackend::malloc(static_cast<size_t>(chunk) * sizeof(float));
        if (b_dev && out_dev) {
            VulkanBackend::launchFp16Gemm(last_dev, b_dev, out_dev, 1, chunk, m_hidden_dim);
            std::vector<float> chunk_logits = download(out_dev, static_cast<size_t>(chunk));
            std::memcpy(&logits[offset], chunk_logits.data(), static_cast<size_t>(chunk) * sizeof(float));
        }
        freeDevice(b_dev);
        freeDevice(out_dev);

        offset += chunk;
    }

    freeDevice(last_dev);

    return logits;
}

std::vector<int32_t> Qwen2VulkanInference::generate(
    const std::string& prompt,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p,
    uint32_t seed)
{
    m_rng.seed(seed);

    std::vector<int32_t> prompt_tokens = m_tokenizer ? m_tokenizer->encode(prompt) : std::vector<int32_t>();
    if (prompt_tokens.empty())
    {
        std::cerr << "[ERROR] No tokens from prompt\n";
        return {};
    }

    // Add BOS if needed
    if (prompt_tokens[0] != m_tokenizer->bos_token_id())
        prompt_tokens.insert(prompt_tokens.begin(), m_tokenizer->bos_token_id());

    std::vector<int32_t> generated;
    int position = 0;

    std::cout << "[INFO] Prefill: " << prompt_tokens.size() << " tokens\n";
    float safe_temp = temperature > 0.0f ? temperature : 1.0f;

    // Prefill
    auto logits = forward(prompt_tokens, position, true);
    position += static_cast<int>(prompt_tokens.size());

    // Apply temperature to logits, then softmax, then sample (no temperature in sampler)
    for (auto& l : logits) l /= safe_temp;
    auto probs = softmax_cpu(logits, 1);

    // Debug: print top-5 logits
    {
        std::vector<std::pair<float, int>> pairs;
        for (int i = 0; i < m_vocab_size; i++) pairs.emplace_back(logits[i], i);
        std::partial_sort(pairs.begin(), pairs.begin() + 5, pairs.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        std::cout << "  [DEBUG] Top-5 logits: ";
        for (int i = 0; i < 5; i++)
            std::cout << "(" << pairs[i].second << "," << pairs[i].first << ") ";
        std::cout << "\n";
    }

    int next_token = sample_top_k_top_p(probs, m_vocab_size, top_k, top_p, m_rng);
    generated.push_back(next_token);

    // Print the decoded token
    std::string decoded1 = m_tokenizer->decode({next_token});
    std::cout << decoded1 << std::flush;

    // Decode loop
    for (int step = 0; step < max_tokens - 1; step++)
    {
        auto dec_logits = forward({next_token}, position, false);

        for (auto& l : dec_logits) l /= safe_temp;
        auto dec_probs = softmax_cpu(dec_logits, 1);
        next_token = sample_top_k_top_p(dec_probs, m_vocab_size, top_k, top_p, m_rng);
        generated.push_back(next_token);

        std::string decoded2 = m_tokenizer->decode({next_token});
        std::cout << decoded2 << std::flush;

        position++;

        if (next_token == m_tokenizer->eos_token_id())
            break;
    }

    std::cout << "\n";
    return generated;
}

} // namespace tllm::vk