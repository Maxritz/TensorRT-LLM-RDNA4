/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/compressorKernels/compressorKernels.h"
#include "tensorrt_llm/common/vulkanBackend.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace tk = tensorrt_llm::kernels::compressor;

namespace
{

// ===== Original three ops (kept for backward compat / direct thop callers) =====

// Decode kernel: write tokens to paged cache + conditional compression
void compressorPagedKvCompressOp(torch::Tensor kv_score, // [m, 2*state_dim] bf16/fp32
    torch::Tensor ape,                                   // [compress_ratio, state_dim] fp32
    torch::Tensor paged_kv,                              // [num_blocks, page_size, state_dim] bf16/fp32
    torch::Tensor paged_score,                           // [num_blocks, page_size, state_dim] bf16/fp32
    torch::Tensor block_table_kv,                        // [bsz, max_blocks] int32
    torch::Tensor block_table_score,                     // [bsz, max_blocks] int32
    torch::Tensor output,                                // [total_outputs, head_dim] bf16
    torch::Tensor kv_lens,                               // [bsz] int32
    torch::Tensor cu_seq_lens,                           // [bsz+1] int32
    torch::Tensor cu_kv_comp,                            // [bsz+1] int32
    int64_t batch_size, int64_t page_size, int64_t head_dim, int64_t compress_ratio, int64_t next_n)
{
    constexpr int64_t kMinNextN = 1;
    constexpr int64_t kMaxNextN = 8;
    TORCH_CHECK(next_n >= kMinNextN && next_n <= kMaxNextN, "next_n must be in [1, 8], got ", next_n);

    int kv_score_eb = static_cast<int>(kv_score.element_size());
    int state_eb = static_cast<int>(paged_kv.element_size());
    int out_eb = static_cast<int>(output.element_size());
    TORCH_CHECK(
        paged_score.element_size() == paged_kv.element_size(), "paged_kv and paged_score must use the same dtype");

    void* stream = nullptr;
#ifdef USE_VULKAN_BACKEND
    if (TLLM_VULKAN_BACKEND_ACTIVE())
    {
        tensorrt_llm::common::VulkanBackend::launchCompressDecode(
            kv_score.data_ptr(), ape.data_ptr<float>(), paged_kv.data_ptr(), paged_score.data_ptr(),
            block_table_kv.data_ptr<int32_t>(), block_table_score.data_ptr<int32_t>(), output.data_ptr(),
            kv_lens.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(), cu_kv_comp.data_ptr<int32_t>(),
            static_cast<uint32_t>(batch_size), static_cast<uint32_t>(page_size),
            static_cast<int>(block_table_kv.size(1)), static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(compress_ratio), static_cast<uint32_t>(next_n),
            kv_score_eb, state_eb, out_eb, stream);
        return;
    }
#endif
    auto cudaStream = at::cuda::getCurrentCUDAStream();
    tk::pagedKvCompressLaunch(kv_score.data_ptr(), ape.data_ptr<float>(), paged_kv.data_ptr(), paged_score.data_ptr(),
        block_table_kv.data_ptr<int32_t>(), block_table_score.data_ptr<int32_t>(), output.data_ptr(),
        kv_lens.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(), cu_kv_comp.data_ptr<int32_t>(),
        static_cast<int>(batch_size), static_cast<int>(page_size), static_cast<int>(block_table_kv.size(1)),
        static_cast<int>(head_dim), static_cast<int>(compress_ratio), static_cast<int>(next_n), kv_score_eb, state_eb,
        out_eb, cudaStream);
}

// Prefill kernel: bulk compression with state update
void compressorPrefillReductionOp(torch::Tensor kv_score, torch::Tensor ape, torch::Tensor paged_kv,
    torch::Tensor paged_score, torch::Tensor block_table_kv, torch::Tensor block_table_score, torch::Tensor output,
    torch::Tensor kv_lens, torch::Tensor start_pos, torch::Tensor cu_seq_lens, torch::Tensor cu_kv_comp,
    int64_t batch_size, int64_t page_size, int64_t head_dim, int64_t compress_ratio, int64_t max_outputs)
{
    int kv_score_eb = static_cast<int>(kv_score.element_size());
    int state_eb = static_cast<int>(paged_kv.element_size());
    int out_eb = static_cast<int>(output.element_size());
    TORCH_CHECK(
        paged_score.element_size() == paged_kv.element_size(), "paged_kv and paged_score must use the same dtype");

    void* stream = nullptr;
#ifdef USE_VULKAN_BACKEND
    if (TLLM_VULKAN_BACKEND_ACTIVE())
    {
        tensorrt_llm::common::VulkanBackend::launchCompressPrefill(
            kv_score.data_ptr(), ape.data_ptr<float>(), paged_kv.data_ptr(), paged_score.data_ptr(),
            block_table_kv.data_ptr<int32_t>(), block_table_score.data_ptr<int32_t>(), output.data_ptr(),
            kv_lens.data_ptr<int32_t>(), start_pos.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(),
            cu_kv_comp.data_ptr<int32_t>(),
            static_cast<uint32_t>(batch_size), static_cast<uint32_t>(page_size),
            static_cast<int>(block_table_kv.size(1)), static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(compress_ratio), static_cast<uint32_t>(max_outputs),
            kv_score_eb, state_eb, out_eb, stream);
        return;
    }
#endif
    auto cudaStream = at::cuda::getCurrentCUDAStream();
    tk::prefillReductionLaunch(kv_score.data_ptr(), ape.data_ptr<float>(), paged_kv.data_ptr(), paged_score.data_ptr(),
        block_table_kv.data_ptr<int32_t>(), block_table_score.data_ptr<int32_t>(), output.data_ptr(),
        kv_lens.data_ptr<int32_t>(), start_pos.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(),
        cu_kv_comp.data_ptr<int32_t>(), static_cast<int>(batch_size), static_cast<int>(page_size),
        static_cast<int>(block_table_kv.size(1)), static_cast<int>(head_dim), static_cast<int>(compress_ratio),
        static_cast<int>(max_outputs), kv_score_eb, state_eb, out_eb, cudaStream);
}

// Fused postprocess + scatter: RMSNorm + RoPE + Hadamard + paged scatter in one kernel
void compressorPostProcessScatterOp(torch::Tensor kv_comp, // [total_tokens, head_dim] input
    std::optional<torch::Tensor> kv_out,                   // [total_tokens, head_dim] output (optional)
    torch::Tensor rms_weight,                              // [head_dim]
    double rms_eps,
    torch::Tensor cos_sin_table,                           // [max_pos, 2, rope_dim/2]
    torch::Tensor position_ids,                            // [total_tokens]
    int64_t nope_dim, int64_t rope_dim,
    torch::Tensor kv_cache,                                // paged cache buffer
    torch::Tensor num_outputs,                             // [bsz] int32
    torch::Tensor cu_kv_comp,                              // [bsz+1] int32
    torch::Tensor start_pos,                               // [bsz] int32
    torch::Tensor block_offsets,                           // [bsz, max_blocks] int32
    torch::Tensor compressed_mask,                         // [total_tokens] bool — per-token mask
    int64_t tokens_per_block, int64_t cache_scale_type, bool rotate_activation,
    std::optional<torch::Tensor> quant_output, std::optional<torch::Tensor> scale_output)
{
    TORCH_CHECK(
        cos_sin_table.scalar_type() == at::kFloat, "cos_sin_table must be float32, got ", cos_sin_table.scalar_type());
    TORCH_CHECK(cos_sin_table.is_contiguous(), "cos_sin_table must be contiguous");
    TORCH_CHECK(position_ids.scalar_type() == at::kInt, "position_ids must be int32, got ", position_ids.scalar_type());
    TORCH_CHECK(position_ids.is_contiguous(), "position_ids must be contiguous");
    TORCH_CHECK(compressed_mask.scalar_type() == at::kBool, "compressed_mask must be bool, got ",
        compressed_mask.scalar_type());

    void* stream = nullptr;
#ifdef USE_VULKAN_BACKEND
    if (TLLM_VULKAN_BACKEND_ACTIVE())
    {
        tensorrt_llm::common::VulkanBackend::launchCompressPostproc(
            kv_comp.data_ptr(), rms_weight.data_ptr(), static_cast<float>(rms_eps),
            cos_sin_table.data_ptr<float>(), position_ids.data_ptr<int32_t>(),
            static_cast<int32_t>(nope_dim), static_cast<int32_t>(rope_dim), kv_cache.data_ptr(),
            num_outputs.data_ptr<int32_t>(), cu_kv_comp.data_ptr<int32_t>(), start_pos.data_ptr<int32_t>(),
            block_offsets.data_ptr<int32_t>(),
            reinterpret_cast<uint8_t const*>(compressed_mask.data_ptr()),
            static_cast<uint32_t>(num_outputs.size(0)), // batch_size
            static_cast<uint32_t>(tokens_per_block),
            static_cast<uint32_t>(kv_comp.size(1)),     // head_dim
            static_cast<uint32_t>(block_offsets.size(1)), // max_blocks
            static_cast<uint32_t>(kv_comp.element_size()), // elem_bytes
            static_cast<uint32_t>(kv_comp.size(0)),     // total_tokens
            static_cast<int32_t>(cache_scale_type), rotate_activation,
            quant_output.has_value() ? quant_output->data_ptr() : nullptr,
            scale_output.has_value() ? scale_output->data_ptr() : nullptr,
            stream);
        return;
    }
#endif
    auto cudaStream = at::cuda::getCurrentCUDAStream();
    tk::postProcessScatterLaunch(kv_comp.data_ptr(), kv_out.has_value() ? kv_out->data_ptr() : nullptr,
        rms_weight.data_ptr(), static_cast<float>(rms_eps), cos_sin_table.data_ptr<float>(),
        position_ids.data_ptr<int32_t>(), static_cast<int>(nope_dim), static_cast<int>(rope_dim), kv_cache.data_ptr(),
        num_outputs.data_ptr<int32_t>(), cu_kv_comp.data_ptr<int32_t>(), start_pos.data_ptr<int32_t>(),
        block_offsets.data_ptr<int32_t>(), reinterpret_cast<bool const*>(compressed_mask.data_ptr()),
        static_cast<int>(num_outputs.size(0)),    // batch_size
        static_cast<int>(tokens_per_block),
        static_cast<int>(kv_comp.size(1)),        // head_dim
        static_cast<int>(block_offsets.size(1)),  // max_blocks
        static_cast<int>(kv_comp.element_size()), // elem_bytes
        static_cast<int>(kv_comp.size(0)),        // total_tokens
        static_cast<int>(cache_scale_type), rotate_activation,
        quant_output.has_value() ? quant_output->data_ptr() : nullptr,
        scale_output.has_value() ? scale_output->data_ptr() : nullptr, cudaStream);
}

// ===== Unified compressor forward — replaces Python Compressor nn.Module =====
//
// Does: F.linear(x, wkv_gate_weight) → prefill/decode compress → postprocess_scatter
// Returns tuple: (primary_output, scale_or_None)
// - indexer NONE: (kv_out, None)
// - indexer FP8_BLOCKWISE: (quant_output, scale_output)
// - indexer MXFP4: (quant_output, scale_output)
// - main compressor (non-indexer): (kv_comp, None)

std::tuple<c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
compressorForwardOp(
    // Input
    torch::Tensor x,                                  // [num_tokens, dim]
    // Weights
    torch::Tensor wkv_gate_weight,                    // [2*state_dim, dim] — stored as [out, in]
    torch::Tensor rms_weight,                         // [head_dim]
    float rms_eps,
    torch::Tensor ape,                                // [compress_ratio, state_dim] fp32
    torch::Tensor cos_sin_table,                      // [max_pos, 2, rope_dim/2] fp32
    // State buffers (mutable)
    torch::Tensor paged_kv_state,                     // [num_blocks, page_size, state_dim]
    torch::Tensor paged_score_state,                  // [num_blocks, page_size, state_dim]
    // Block tables
    torch::Tensor block_table_kv_state,               // [bsz, max_blocks] int32
    torch::Tensor block_table_score_state,            // [bsz, max_blocks] int32
    torch::Tensor block_table,                        // [bsz, max_blocks] int32 (for scatter)
    torch::Tensor kv_cache,                           // paged KV cache buffer (for scatter)
    // Compression metadata
    torch::Tensor kv_lens,                            // [bsz] int32
    torch::Tensor cu_seq_lens,                        // [bsz+1] int32
    torch::Tensor cu_new_comp_kv,                     // [bsz+1] int32
    torch::Tensor cached_token_lens,                  // [bsz] int32 (for prefill)
    torch::Tensor start_pos,                          // [bsz] int32 (for scatter)
    torch::Tensor num_outputs,                        // [bsz] int32 (for scatter)
    torch::Tensor position_ids,                       // [total_tokens] int32
    torch::Tensor compressed_mask,                    // [total_tokens] bool
    // Scalar config
    int64_t num_contexts,
    int64_t num_generations,
    int64_t head_dim,
    int64_t state_dim,
    int64_t compress_ratio,
    int64_t page_size,
    int64_t state_tokens_per_block,
    int64_t compress_tokens_per_block,
    int64_t max_ctx_comp_kv_lens,
    int64_t total_num_comp_tokens,
    int64_t cache_scale_type,
    bool rotate_activation,
    bool is_indexer,
    int64_t nope_dim,
    int64_t rope_dim,
    int64_t next_n)
{
    TORCH_CHECK(x.is_cuda() || TLLM_VULKAN_BACKEND_ACTIVE(), "x must be on GPU");
    TORCH_CHECK(cos_sin_table.scalar_type() == at::kFloat, "cos_sin_table must be float32");
    TORCH_CHECK(cos_sin_table.is_contiguous(), "cos_sin_table must be contiguous");
    TORCH_CHECK(position_ids.scalar_type() == at::kInt, "position_ids must be int32");
    TORCH_CHECK(position_ids.is_contiguous(), "position_ids must be contiguous");
    TORCH_CHECK(compressed_mask.scalar_type() == at::kBool, "compressed_mask must be bool");

    int kv_score_eb = static_cast<int>(x.element_size());
    int state_eb = static_cast<int>(paged_kv_state.element_size());
    int out_eb = kv_score_eb;

    // 1. F.linear: kv_score [num_tokens, 2*state_dim] = x @ w^T
    auto x_contig = x.is_contiguous() ? x : x.contiguous();
    auto w_contig = wkv_gate_weight.is_contiguous() ? wkv_gate_weight : wkv_gate_weight.contiguous();
    auto kv_score = torch::empty({x.size(0), 2 * state_dim}, x.options());
    at::mm_out(kv_score, x_contig, w_contig.t());

    // 2. Allocate output buffer
    auto kv_comp = torch::empty({total_num_comp_tokens, static_cast<int64_t>(head_dim)}, x.options());

    void* stream = nullptr;
#ifdef USE_VULKAN_BACKEND
    bool vulkanActive = TLLM_VULKAN_BACKEND_ACTIVE();
#else
    bool vulkanActive = false;
#endif

    auto maxBlocks = static_cast<int64_t>(block_table_kv_state.size(1));
    auto btSlice = [](torch::Tensor t, int64_t start, int64_t len) {
        return t.slice(0, start, start + len);
    };

    // 3. Run compression kernels — prefill (context) path
    if (num_contexts > 0)
    {
        auto bt_kv = btSlice(block_table_kv_state, 0, num_contexts);
        auto bt_score = btSlice(block_table_score_state, 0, num_contexts);
        auto kv_lens_ctx = btSlice(kv_lens, 0, num_contexts);
        auto cached_lens = btSlice(cached_token_lens, 0, num_contexts);
        auto cu_kv_comp_ctx = btSlice(cu_new_comp_kv, 0, num_contexts + 1);

        if (vulkanActive)
        {
            tensorrt_llm::common::VulkanBackend::launchCompressPrefill(
                kv_score.data_ptr(), ape.data_ptr<float>(), paged_kv_state.data_ptr(), paged_score_state.data_ptr(),
                bt_kv.data_ptr<int32_t>(), bt_score.data_ptr<int32_t>(),
                kv_comp.data_ptr(), kv_lens_ctx.data_ptr<int32_t>(), cached_lens.data_ptr<int32_t>(),
                cu_seq_lens.data_ptr<int32_t>(), cu_kv_comp_ctx.data_ptr<int32_t>(),
                static_cast<uint32_t>(num_contexts), static_cast<uint32_t>(page_size),
                static_cast<int>(maxBlocks), static_cast<uint32_t>(head_dim),
                static_cast<uint32_t>(compress_ratio), static_cast<uint32_t>(max_ctx_comp_kv_lens),
                kv_score_eb, state_eb, out_eb, stream);
        }
        else
        {
            auto cudaStream = at::cuda::getCurrentCUDAStream();
            tk::prefillReductionLaunch(
                kv_score.data_ptr(), ape.data_ptr<float>(), paged_kv_state.data_ptr(), paged_score_state.data_ptr(),
                bt_kv.data_ptr<int32_t>(), bt_score.data_ptr<int32_t>(),
                kv_comp.data_ptr(), kv_lens_ctx.data_ptr<int32_t>(), cached_lens.data_ptr<int32_t>(),
                cu_seq_lens.data_ptr<int32_t>(), cu_kv_comp_ctx.data_ptr<int32_t>(),
                static_cast<int>(num_contexts), static_cast<int>(page_size),
                static_cast<int>(maxBlocks), static_cast<int>(head_dim),
                static_cast<int>(compress_ratio), static_cast<int>(max_ctx_comp_kv_lens),
                kv_score_eb, state_eb, out_eb, cudaStream);
        }
    }

    // Decode (generation) path
    if (num_generations > 0)
    {
        auto bt_kv = btSlice(block_table_kv_state, num_contexts, num_generations);
        auto bt_score = btSlice(block_table_score_state, num_contexts, num_generations);
        auto gen_kv_lens = btSlice(kv_lens, num_contexts, num_generations);
        auto cu_seq_lens_gen = btSlice(cu_seq_lens, num_contexts, cu_seq_lens.size(0) - num_contexts);
        auto cu_kv_comp_gen = btSlice(cu_new_comp_kv, num_contexts, cu_new_comp_kv.size(0) - num_contexts);

        if (vulkanActive)
        {
            tensorrt_llm::common::VulkanBackend::launchCompressDecode(
                kv_score.data_ptr(), ape.data_ptr<float>(), paged_kv_state.data_ptr(), paged_score_state.data_ptr(),
                bt_kv.data_ptr<int32_t>(), bt_score.data_ptr<int32_t>(),
                kv_comp.data_ptr(), gen_kv_lens.data_ptr<int32_t>(),
                cu_seq_lens_gen.data_ptr<int32_t>(), cu_kv_comp_gen.data_ptr<int32_t>(),
                static_cast<uint32_t>(num_generations), static_cast<uint32_t>(page_size),
                static_cast<int>(maxBlocks), static_cast<uint32_t>(head_dim),
                static_cast<uint32_t>(compress_ratio), static_cast<uint32_t>(next_n),
                kv_score_eb, state_eb, out_eb, stream);
        }
        else
        {
            auto cudaStream = at::cuda::getCurrentCUDAStream();
            tk::pagedKvCompressLaunch(
                kv_score.data_ptr(), ape.data_ptr<float>(), paged_kv_state.data_ptr(), paged_score_state.data_ptr(),
                bt_kv.data_ptr<int32_t>(), bt_score.data_ptr<int32_t>(),
                kv_comp.data_ptr(), gen_kv_lens.data_ptr<int32_t>(),
                cu_seq_lens_gen.data_ptr<int32_t>(), cu_kv_comp_gen.data_ptr<int32_t>(),
                static_cast<int>(num_generations), static_cast<int>(page_size),
                static_cast<int>(maxBlocks), static_cast<int>(head_dim),
                static_cast<int>(compress_ratio), static_cast<int>(next_n),
                kv_score_eb, state_eb, out_eb, cudaStream);
        }
    }

    // 4. Scatter to cache with quantization
    int64_t total_tokens = kv_comp.size(0);

    torch::Tensor kv_out;
    torch::Tensor quant_output;
    torch::Tensor scale_output;

    if (is_indexer)
    {
        if (cache_scale_type == 0)  // NONE
        {
            kv_out = torch::empty_like(kv_comp);
        }
        else if (cache_scale_type == 2)  // FP8_BLOCKWISE
        {
            int64_t num_scale_blocks = head_dim / 128;
            quant_output = torch::empty({total_tokens, head_dim}, x.options().dtype(torch::kUInt8));
            scale_output = torch::empty({total_tokens, num_scale_blocks}, x.options().dtype(torch::kFloat32));
        }
        else if (cache_scale_type == 3)  // MXFP4_BLOCKWISE
        {
            int64_t num_scale_blocks = head_dim / 32;
            quant_output = torch::empty({total_tokens, head_dim / 2}, x.options().dtype(torch::kUInt8));
            scale_output = torch::empty({total_tokens, num_scale_blocks}, x.options().dtype(torch::kUInt8));
        }
    }

    if (vulkanActive)
    {
        tensorrt_llm::common::VulkanBackend::launchCompressPostproc(
            kv_comp.data_ptr(),
            kv_out.defined() ? kv_out.data_ptr() : nullptr,
            rms_weight.data_ptr(), rms_eps,
            cos_sin_table.data_ptr<float>(), position_ids.data_ptr<int32_t>(),
            static_cast<int32_t>(nope_dim), static_cast<int32_t>(rope_dim), kv_cache.data_ptr(),
            num_outputs.data_ptr<int32_t>(), cu_new_comp_kv.data_ptr<int32_t>(), start_pos.data_ptr<int32_t>(),
            block_table.data_ptr<int32_t>(),
            reinterpret_cast<uint8_t const*>(compressed_mask.data_ptr()),
            static_cast<uint32_t>(num_outputs.size(0)),
            static_cast<uint32_t>(compress_tokens_per_block),
            static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(block_table.size(1)),
            static_cast<uint32_t>(out_eb),
            static_cast<uint32_t>(total_tokens),
            static_cast<int32_t>(cache_scale_type), rotate_activation,
            quant_output.defined() ? quant_output.data_ptr() : nullptr,
            scale_output.defined() ? scale_output.data_ptr() : nullptr,
            stream);
    }
    else
    {
        auto cudaStream = at::cuda::getCurrentCUDAStream();
        tk::postProcessScatterLaunch(
            kv_comp.data_ptr(),
            kv_out.defined() ? kv_out.data_ptr() : nullptr,
            rms_weight.data_ptr(), rms_eps,
            cos_sin_table.data_ptr<float>(), position_ids.data_ptr<int32_t>(),
            static_cast<int>(nope_dim), static_cast<int>(rope_dim), kv_cache.data_ptr(),
            num_outputs.data_ptr<int32_t>(), cu_new_comp_kv.data_ptr<int32_t>(), start_pos.data_ptr<int32_t>(),
            block_table.data_ptr<int32_t>(), reinterpret_cast<bool const*>(compressed_mask.data_ptr()),
            static_cast<int>(num_outputs.size(0)),
            static_cast<int>(compress_tokens_per_block),
            static_cast<int>(head_dim),
            static_cast<int>(block_table.size(1)),
            static_cast<int>(out_eb),
            static_cast<int>(total_tokens),
            static_cast<int>(cache_scale_type), rotate_activation,
            quant_output.defined() ? quant_output.data_ptr() : nullptr,
            scale_output.defined() ? scale_output.data_ptr() : nullptr, cudaStream);
    }

    // Sync if Vulkan
#ifdef USE_VULKAN_BACKEND
    if (vulkanActive)
    {
        tensorrt_llm::common::VulkanBackend::streamSynchronize(stream);
    }
#endif

    // Return: (primary_output, scale_or_None)
    // - indexer NONE: (kv_out, None)
    // - indexer FP8_BLOCKWISE: (quant_output, scale_output)
    // - indexer MXFP4: (quant_output, scale_output)
    // - main compressor (non-indexer): (kv_comp, None)
    c10::optional<torch::Tensor> primaryOut;
    c10::optional<torch::Tensor> scaleOut;

    if (kv_out.defined())
        primaryOut = kv_out;
    else if (quant_output.defined())
        primaryOut = quant_output;
    else
        primaryOut = kv_comp;

    if (scale_output.defined())
        scaleOut = scale_output;

    return std::make_tuple(primaryOut, scaleOut);
}

} // anonymous namespace

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "compressor_paged_kv_compress("
        "Tensor kv_score, Tensor ape, "
        "Tensor(a!) paged_kv, Tensor(b!) paged_score, "
        "Tensor block_table_kv, Tensor block_table_score, "
        "Tensor(c!) output, "
        "Tensor kv_lens, "
        "Tensor cu_seq_lens, Tensor cu_kv_comp, "
        "int batch_size, int page_size, "
        "int head_dim, int compress_ratio, "
        "int next_n) -> ()");

    m.def(
        "compressor_prefill_reduction("
        "Tensor kv_score, Tensor ape, "
        "Tensor(a!) paged_kv, Tensor(b!) paged_score, "
        "Tensor block_table_kv, Tensor block_table_score, "
        "Tensor(c!) output, "
        "Tensor kv_lens, Tensor start_pos, "
        "Tensor cu_seq_lens, Tensor cu_kv_comp, "
        "int batch_size, int page_size, "
        "int head_dim, int compress_ratio, "
        "int max_outputs) -> ()");

    m.def(
        "compressor_postprocess_scatter("
        "Tensor kv_comp, Tensor(a!)? kv_out, "
        "Tensor rms_weight, float rms_eps, "
        "Tensor cos_sin_table, Tensor position_ids, "
        "int nope_dim, int rope_dim, "
        "Tensor(b!) kv_cache, "
        "Tensor num_outputs, Tensor cu_kv_comp, "
        "Tensor start_pos, Tensor block_offsets, "
        "Tensor compressed_mask, "
        "int tokens_per_block, int cache_scale_type, "
        "bool rotate_activation, "
        "Tensor(c!)? quant_output, Tensor(d!)? scale_output) -> ()");

    // Unified compressor forward — replaces Python Compressor nn.Module
    // Returns: (kv_comp_or_kv_out, quant_output_or_None, scale_output_or_None)
    m.def(
        "compressor_forward("
        "Tensor x, Tensor wkv_gate_weight, Tensor rms_weight, float rms_eps, "
        "Tensor ape, Tensor cos_sin_table, "
        "Tensor(a!) paged_kv_state, Tensor(b!) paged_score_state, "
        "Tensor block_table_kv_state, Tensor block_table_score_state, "
        "Tensor block_table, Tensor(c!) kv_cache, "
        "Tensor kv_lens, Tensor cu_seq_lens, Tensor cu_new_comp_kv, "
        "Tensor cached_token_lens, Tensor start_pos, Tensor num_outputs, "
        "Tensor position_ids, Tensor compressed_mask, "
        "int num_contexts, int num_generations, "
        "int head_dim, int state_dim, int compress_ratio, "
        "int page_size, int state_tokens_per_block, int compress_tokens_per_block, "
        "int max_ctx_comp_kv_lens, int total_num_comp_tokens, "
        "int cache_scale_type, bool rotate_activation, bool is_indexer, "
        "int nope_dim, int rope_dim, int next_n"
        ") -> (Tensor?, Tensor?)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("compressor_paged_kv_compress", &compressorPagedKvCompressOp);
    m.impl("compressor_prefill_reduction", &compressorPrefillReductionOp);
    m.impl("compressor_postprocess_scatter", &compressorPostProcessScatterOp);
    m.impl("compressor_forward", &compressorForwardOp);
}
