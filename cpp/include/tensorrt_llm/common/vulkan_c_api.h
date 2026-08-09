/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

/*
 * C ABI bridge for the Vulkan backend.
 *
 * Exposes `extern "C"` entry points that can be loaded via ctypes/cffi from
 * any language with a C-FFI.  Each function takes raw device-pointer values
 * and primitive scalars so Python code can pass `tensor.data_ptr()` directly.
 *
 * Data-path strategy: host staging.  PyTorch GPU tensor → host → Vulkan
 * device buffer → compute → host → PyTorch GPU tensor.  This is correct but
 * not optimal; a future optimisation can add VK external-memory interop to
 * share buffers directly.
 */

#include "tensorrt_llm/common/vulkanBackend.h"
#include "tensorrt_llm/kernels/vulkanKernelRegistry.h"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

using namespace tensorrt_llm::common;

namespace {

VulkanBackend* getBackend()
{
    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive()) {
        if (!backend->initialize(0)) {
            throw std::runtime_error(
                std::string("VulkanBackend failed to initialize. ") +
                backend->getLastError());
        }
    }
    return backend.get();
}

} // namespace

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

extern "C" int32_t tllm_vulkan_init(uint32_t gpu_id)
{
    try {
        auto backend = VulkanBackend::getInstance();
        return backend->initialize(gpu_id) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

extern "C" int32_t tllm_vulkan_is_active()
{
    return VulkanBackend::getInstance()->isActive() ? 1 : 0;
}

extern "C" void tllm_vulkan_device_synchronize()
{
    VulkanBackend::deviceSynchronize();
}

/* ------------------------------------------------------------------ */
/* Memory helpers                                                     */
/* ------------------------------------------------------------------ */

extern "C" void tllm_vulkan_malloc(void** out_ptr, size_t byte_count)
{
    *out_ptr = VulkanBackend::malloc(byte_count);
}

extern "C" void tllm_vulkan_free(void* ptr)
{
    VulkanBackend::free(ptr);
}

extern "C" void tllm_vulkan_memcpy_h2d(void* dst_device, const void* src_host, size_t byte_count)
{
    VulkanBackend::memcpyHostToDevice(dst_device, src_host, byte_count);
}

extern "C" void tllm_vulkan_memcpy_d2h(void* dst_host, const void* src_device, size_t byte_count)
{
    VulkanBackend::memcpyDeviceToHost(dst_host, src_device, byte_count);
}

/* ------------------------------------------------------------------ */
/* Softmax                                                             */
/* ------------------------------------------------------------------ */

extern "C" int32_t tllm_vulkan_softmax(
    void* input_ptr, void* output_ptr,
    uint32_t batch_size, uint32_t num_heads, uint32_t seq_len,
    uint32_t dtype)
{
    try {
        auto backend = getBackend();
        backend->launchSoftmax(input_ptr, output_ptr, batch_size, num_heads, seq_len);
        backend->streamSynchronize();
        return 1;
    } catch (...) {
        return 0;
    }
}

/* ------------------------------------------------------------------ */
/* Core compute ops (GEMM, RMS norm, elementwise)                      */
/* ------------------------------------------------------------------ */

// FP16/FP32 GEMM: C[M,N] = A[M,K] * B[K,N], row-major, non-transposed.
// (This variant stores operands/output in fp32 for deterministic
//  verification — see fp16_gemm.comp; a folded float16 variant is a
//  follow-up.) Mirrors launchFp16Gemm which hard-codes aT=bT=false.
extern "C" int32_t tllm_vulkan_gemm(
    void* a, void* b, void* output,
    uint32_t M, uint32_t N, uint32_t K)
{
    try {
        auto backend = getBackend();
        backend->launchFp16Gemm(a, b, output, M, N, K);
        backend->streamSynchronize();
        return 1;
    } catch (...) {
        return 0;
    }
}

// RMSNorm: out = (in / sqrt(mean(in^2) + eps)) * gamma + beta,
// feature-normalized per token over hiddenDim (row-major
// tokenCount * hiddenDim) — matches the rms_norm.comp reference.
extern "C" int32_t tllm_vulkan_rms_norm(
    void* input, void* gamma, void* beta, void* output,
    float eps, size_t hiddenDim, size_t tokenCount)
{
    try {
        auto backend = getBackend();
        backend->launchRmsNorm(input, gamma, beta, output, eps, hiddenDim, tokenCount);
        backend->streamSynchronize();
        return 1;
    } catch (...) {
        return 0;
    }
}

// Elementwise add: output = a + b (fp32), matches elementwise_add.comp.
extern "C" int32_t tllm_vulkan_elementwise_add(
    void* a, void* b, void* output, size_t elementCount)
{
    try {
        auto backend = getBackend();
        backend->launchElementwiseAdd(a, b, output, elementCount);
        backend->streamSynchronize();
        return 1;
    } catch (...) {
        return 0;
    }
}

/* ------------------------------------------------------------------ */
/* Attention (scaled dot-product, fp32)                                 */
/* ------------------------------------------------------------------ */

// O = softmax((Q K^T) / sqrt(headDim)) V, row-major [batch, numHeads, seq, headDim].
// `causal` masks keys j > i (set to -inf). Mirrors attention.comp, which is
// one thread per query, deterministic 3-pass (max/sum/out) reduction.
extern "C" int32_t tllm_vulkan_attention(
    void* q, void* k, void* v, void* output,
    uint32_t batchSize, uint32_t numHeads,
    uint32_t seqLenQ, uint32_t seqLenK, uint32_t headDim,
    uint32_t causal)
{
    try {
        auto backend = getBackend();
        backend->launchAttention(q, k, v, output,
            batchSize, numHeads, seqLenQ, seqLenK, headDim, causal != 0u);
        backend->streamSynchronize();
        return 1;
    } catch (...) {
        return 0;
    }
}

/* ------------------------------------------------------------------ */
/* Top-K (sparse attention token selection, fp32)                        */
/* ------------------------------------------------------------------ */

// Top-k per (batch, head) row from `scores`: emits up to `topk` row-local
// token offsets (descending value, first-max wins ties). input_offsets and
// output_offsets are [batchSize+1] uint exclusive scans; topkIndices is
// [numHeads * totalOutputTokens] int. Mirrors topk.comp.
extern "C" int32_t tllm_vulkan_topk(
    void* scores, void* inputOffsets, void* outputOffsets,
    void* topkIndices,
    uint32_t topk, uint32_t numHeads, uint32_t batchSize,
    uint32_t totalTokens, uint32_t totalOutputTokens)
{
    try {
        auto backend = getBackend();
        backend->launchTopk(scores, inputOffsets, outputOffsets, topkIndices,
            topk, numHeads, batchSize, totalTokens, totalOutputTokens);
        backend->streamSynchronize();
        return 1;
    } catch (...) {
        return 0;
    }
}

/* ------------------------------------------------------------------ */
/* Q8_0 (block-quantized) GEMM                                         */
/* ------------------------------------------------------------------ */

// Dequantized matmul: C[M,N] = A[M,K] * W_dequant[N,K]^T, one thread per
// output element. Weights use the GGML block_q8_0 layout: each 36-byte block
// is one fp32 scale followed by 32 signed int8 weights (shared scale per
// block). K must be a multiple of 32; blocksPerRow = K / 32. Mirrors
// q8_0_gemm.comp and VulkanBackend::launchQ8_0Gemm.
extern "C" int32_t tllm_vulkan_q8_0_gemm(
    void* weight, void* activation, void* output,
    uint32_t M, uint32_t N, uint32_t K, uint32_t blocksPerRow)
{
    try {
        auto backend = getBackend();
        backend->launchQ8_0Gemm(weight, activation, output, M, N, K, blocksPerRow);
        backend->streamSynchronize();
        return 1;
    } catch (...) {
        return 0;
    }
}

/* ------------------------------------------------------------------ */
/* KV Cache Update (2D) — post speculative decode                      */
/* ------------------------------------------------------------------ */

extern "C" int32_t tllm_vulkan_kv_cache_update_2d(
    void* kv_cache_k, void* kv_cache_v,
    void* accepted_indices, void* num_accepted_tokens,
    void* past_key_value_lens,
    void* rewind_adjustments,
    void* seq_slot_remapping,
    uint32_t batch_size, uint32_t num_kv_heads,
    uint32_t max_kv_cache_len, uint32_t head_dim,
    uint32_t max_draft_len, int32_t rewind_draft_token_common_count,
    uint32_t layer_count)
{
    try {
        auto backend = getBackend();
        backend->launchKVCacheUpdate2D(
            kv_cache_k, kv_cache_v,
            accepted_indices, num_accepted_tokens,
            past_key_value_lens,
            rewind_adjustments,
            seq_slot_remapping,
            batch_size, num_kv_heads, max_kv_cache_len,
            head_dim, max_draft_len, rewind_draft_token_common_count,
            layer_count);
        backend->streamSynchronize();
        return 1;
    } catch (...) {
        return 0;
    }
}

/* ------------------------------------------------------------------ */
/* Tree spec decode                                                    */
/* ------------------------------------------------------------------ */

extern "C" int32_t tllm_vulkan_tree_spec_build(
    void* parent_list, void* selected_index,
    void* tree_mask, void* positions,
    void* retrieve_index, void* retrieve_next_token,
    void* retrieve_next_sibling,
    uint32_t batch_size, uint32_t draft_token_num,
    uint32_t top_k, uint32_t depth,
    uint32_t num_int32_per_row)
{
    try {
        auto backend = getBackend();
        backend->launchTreeSpecBuild(
            parent_list, selected_index, tree_mask, positions,
            retrieve_index, retrieve_next_token, retrieve_next_sibling,
            batch_size, draft_token_num, top_k, depth, num_int32_per_row);
        backend->streamSynchronize();
        return 1;
    } catch (...) {
        return 0;
    }
}

extern "C" int32_t tllm_vulkan_tree_spec_rejection(
    void* accept_index, void* accept_token_num, void* accept_token,
    void* draft_tokens, void* target_probs,
    void* retrieve_next_token, void* retrieve_next_sibling,
    void* tree_valid, void* rng_samples,
    uint32_t batch_size, uint32_t num_spec_tokens,
    uint32_t num_draft_tokens, uint32_t vocab_size,
    uint32_t k_max_tried_per_level)
{
    try {
        auto backend = getBackend();
        backend->launchTreeSpecRejection(
            accept_index, accept_token_num, accept_token,
            draft_tokens, target_probs,
            retrieve_next_token, retrieve_next_sibling,
            tree_valid, rng_samples,
            batch_size, num_spec_tokens, num_draft_tokens,
            vocab_size, k_max_tried_per_level);
        backend->streamSynchronize();
        return 1;
    } catch (...) {
        return 0;
    }
}

/* ------------------------------------------------------------------ */
/* Spec-decode accept                                                  */
/* ------------------------------------------------------------------ */

extern "C" int32_t tllm_vulkan_spec_accept(
    void* target_logits, void* draft_logits, void* uniform_rng,
    void* draft_tokens, void* accept_count, void* accepted_tokens,
    void* resample_probs,
    uint32_t batch_size, uint32_t draft_len, uint32_t vocab_size,
    float temperature, float accept_prob_floor)
{
    try {
        auto backend = getBackend();
        backend->launchSpecDecodeAccept(
            target_logits, draft_logits, uniform_rng, draft_tokens,
            accept_count, accepted_tokens, resample_probs,
            batch_size, draft_len, vocab_size,
            temperature, accept_prob_floor);
        backend->streamSynchronize();
        return 1;
    } catch (...) {
        return 0;
    }
}
