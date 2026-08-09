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

#ifndef VULKAN_KERNEL_REGISTRY_H
#define VULKAN_KERNEL_REGISTRY_H

#include "tensorrt_llm/common/vulkanContext.h"
#include "tensorrt_llm/common/vulkanRuntime.h"
#include "tensorrt_llm/common/vulkanShaderCompiler.h"
#include "tensorrt_llm/common/vulkanMemoryAllocator.h"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

TRTLLM_NAMESPACE_BEGIN
namespace common
{
}

namespace kernels
{

// KernelDescriptor: Metadata for a Vulkan kernel
struct KernelDescriptor
{
    std::string name;                         // Kernel name (e.g., "rms_norm")
    std::string shaderPath;                   // Path to .comp file
    std::string entryPoint;                   // Entry point name
    uint32_t blockM = 256;                   // Block size M dimension
    uint32_t blockN = 256;                   // Block size N dimension
    uint32_t blockK = 256;                   // Block size K dimension
    std::vector<uint32_t> requiredFeatures; // Feature flags required
    bool requiresCooperativeMatrix = false;
    bool requiresFP16 = false;
    bool requiresBF16 = false;
    bool isQuantized = false;
    uint32_t quantType = 0; // 0=fp16, 1=q8_0, 2=q4_k, etc.
    uint32_t bindingCount = 3; // Number of descriptor buffer bindings the shader expects
};

// KernelVariant: A compiled shader variant for a specific GPU target
struct KernelVariant
{
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    common::GPUTarget target;
    common::ShaderFeatureFlags features;
    uint32_t subgroupSize = 32;
    uint32_t localSizeX = 0;
    uint32_t localSizeY = 0;
    uint32_t localSizeZ = 0;
    std::string entryPoint = "";
    uint32_t specializationConstants[13] = {0}; // Matches GLSL SPEC_* constants
    bool isValid() const { return module != VK_NULL_HANDLE && pipeline != VK_NULL_HANDLE; }
    void destroy(VkDevice device);
};

// VulkanKernelRegistry: Central registry of Vulkan kernels
// Mirrors CUDA kernel dispatch table
class VulkanKernelRegistry
{
public:
    static std::shared_ptr<VulkanKernelRegistry> getInstance();

    ~VulkanKernelRegistry();

    // Initialize with a Vulkan runtime context
    bool initialize();

    // Register a kernel descriptor
    void registerKernel(KernelDescriptor const& desc);

    // Get compiled kernel variant for a specific target and features
    std::shared_ptr<KernelVariant> getOrCreate(
        std::string const& kernelName,
        common::GPUTarget const& target,
        common::ShaderFeatureFlags const& features);

    // Get all available kernels
    std::vector<std::string> getKernelNames() const;

    // Check if a kernel is available for the current GPU
    bool isKernelAvailable(std::string const& kernelName) const;

    // Get optimal kernel variant for current GPU
    std::shared_ptr<KernelVariant> getBestVariant(
        std::string const& kernelName,
        common::VulkanMemoryManager* memMgr = nullptr);

    // Compile all registered kernels for the current target
    bool compileAll();

    // Get the list of kernel descriptors
    std::vector<KernelDescriptor> getDescriptors() const;

    // Internal: Constructor accessible to getInstance()
    VulkanKernelRegistry();

private:
    bool compileKernel(KernelDescriptor const& desc, common::GPUTarget const& target,
                       common::ShaderFeatureFlags const& features,
                       std::shared_ptr<KernelVariant>& outVariant);
    std::string generateSpecializationConstants(KernelDescriptor const& desc, common::GPUTarget const& target);

    std::shared_ptr<common::VulkanRuntime> mRuntime;
    std::shared_ptr<common::VulkanContext> mContext;
    std::shared_ptr<common::VulkanShaderCompiler> mCompiler;
    std::vector<KernelDescriptor> mDescriptors;
    std::unordered_map<std::string, std::vector<std::shared_ptr<KernelVariant>>> mCompiledKernels;
    bool mInitialized = false;
};

// Kernel dispatcher that mirrors CUDA kernel launches
class VulkanKernelDispatcher
{
public:
    explicit VulkanKernelDispatcher(std::shared_ptr<common::VulkanContext> const& ctx);
    ~VulkanKernelDispatcher();

    // ==================== Elementwise Operations ====================
    // Mirrors CUDA elementwise_add, elementwise_mul, etc.
    common::VulkanResult dispatchElementwiseAdd(
        void* a, void* b, void* output,
        size_t elementCount,
        uint32_t blockSize = 256);

    // ==================== Normalization Operations ====================
    // Mirrors CUDA rmsnorm kernels
    common::VulkanResult dispatchRmsNorm(
        void* input, void* gamma, void* beta, void* output,
        float eps, size_t hiddenDim, size_t tokenCount,
        uint32_t blockSize = 256);

    // ==================== GEMM Operations ====================
    // Mirrors CUDA fpA_intB_gemm, groupGemm, etc.
    common::VulkanResult dispatchFp16Gemm(
        void* a, void* b, void* output,
        uint32_t M, uint32_t N, uint32_t K,
        bool aTransposed = false, bool bTransposed = false,
        uint32_t blockSize = 128);

    common::VulkanResult dispatchQ8_0Gemm(
        void* weight, void* activation, void* output,
        uint32_t M, uint32_t N, uint32_t K,
        uint32_t blocksPerRow = 0,
        uint32_t blockSize = 128);

    // ==================== Attention Operations ====================
    // Mirrors CUDA attention kernels
    common::VulkanResult dispatchSoftmax(
        void* input, void* output,
        uint32_t batchSize, uint32_t numHeads, uint32_t seqLen,
        uint32_t blockSize = 256);

    common::VulkanResult dispatchAttention(
        void* q, void* k, void* v, void* output,
        uint32_t batchSize, uint32_t numHeads,
        uint32_t seqLenQ, uint32_t seqLenK, uint32_t headDim,
        bool causal,
        uint32_t blockSize = 256);

    // ==================== Sparse Operations ====================
    // Mirrors CUDA sparse top-k (histogram) kernels, e.g. topk_kernel in
    // tensorrt-llm/_torch/attention_backend/sparse/kernel.py (triton_topk).
    // scores:       [numHeads * totalTokens]            (float)
    // inputOffsets: [batchSize + 1]                     (uint)
    // outputOffsets:[batchSize + 1]                     (uint)
    // topkIndices:  [numHeads * totalOutputTokens]      (int32, row-local offsets)
    common::VulkanResult dispatchTopk(
        void* scores, void* inputOffsets, void* outputOffsets, void* topkIndices,
        uint32_t topk, uint32_t numHeads, uint32_t batchSize,
        uint32_t totalTokens, uint32_t totalOutputTokens,
        uint32_t blockSize = 64);

    // ==================== Speculative Decoding ====================
    // Vulkan port of externalDraftTokensKernels.cu spec-decode acceptance loop.
    // targetLogits:[B,(draftLen+1)*V], draftLogits:[B,(draftLen+1)*V],
    // uniform:[B*draftLen], draftTokens:[B*(draftLen+1)],
    // acceptCount:[B], acceptedTokens:[B*(draftLen+1)], resampleProbs:[B*V].
    common::VulkanResult dispatchSpecDecodeAccept(
        void* targetLogits, void* draftLogits, void* uniformRng,
        void* draftTokens, void* acceptCount, void* acceptedTokens,
        void* resampleProbs,
         uint32_t batchSize, uint32_t draftLen, uint32_t vocabSize,
         float temperature, float acceptProbFloor, uint32_t blockSize = 1);

    // ==================== Tree Spec-Decoding ====================
    // Vulkan ports of buildDynamicTreeKernelPacked (line 157) and
    // verifyDynamicTreeGreedyPackedKernel (line 284) in dynamicTreeKernels.cu.
    // int64 host buffers (parentList, selectedIndex) are passed as uvec2-pairs:
    // host packs each int64 as {lo,hi} uint32; shader reads .x (indices < 2^31).
    common::VulkanResult dispatchTreeSpecBuild(
        void* parentListUvec2, void* selectedIndexUvec2,
        void* treeMask, void* positions, void* retrieveIndex,
        void* retrieveNextToken, void* retrieveNextSibling,
        uint32_t batchSize, uint32_t draftTokenNum, uint32_t topK,
        uint32_t depth, uint32_t numInt32PerRow, uint32_t blockSize = 1);

    common::VulkanResult dispatchTreeSpecGreedyVerify(
        void* acceptIndex, void* acceptTokenNum, void* acceptToken,
        void* candidates, void* retrievePacked, void* targetPredict,
        void* treeValid, uint32_t batchSize, uint32_t numSpeculativeTokens,
         uint32_t numDraftTokens, uint32_t blockSize = 1);

    // Vulkan port of verifyDynamicTreeRejectionKernel (line 518).
    // int64 acceptIndex/acceptToken are host-packed as uvec2 pairs.
    // rngSamples [B] is host-injected uniform float (Philox-equivalent).
    common::VulkanResult dispatchTreeSpecRejection(
        void* acceptIndex, void* acceptTokenNum, void* acceptToken,
        void* draftTokens, void* targetProbs, void* retrieveNextToken,
        void* retrieveNextSibling, void* treeValid, void* rngSamples,
        uint32_t batchSize, uint32_t numSpeculativeTokens, uint32_t numDraftTokens,
        uint32_t vocabSize,      uint32_t kMaxTriedPerLevel, uint32_t blockSize = 256);

    // ==================== KV Cache Update ====================
    // Vulkan port of updateKVCacheDraftTokenLocationBatchedKernel2D
    // (kvCacheUpdateKernels.cu line 140). Compacts accepted draft tokens
    // in the KV cache: copies K/V from scattered draft positions to
    // contiguous positions starting at (pastKVLen - rewindDraftTokenCommonCount).
    // kvCacheK/V layout: [layer, seq, head, pos, channel] (float).
    // acceptedDraftTokens [seq, maxDraftLen] (int32, -1 padding).
    // rewindSepAdj / seqSlotRemap pass -1 as sentinel when unused.
    common::VulkanResult dispatchKVCacheUpdate2D(
        void* kvCacheK, void* kvCacheV,
        void* acceptedDraftTokensIndices2D, void* numAcceptedTokens,
        void* pastKeyValueLengths, void* rewindDraftTokenSeparateAdjustments,
        void* seqSlotRemapping,
        uint32_t batchSize, uint32_t numKVHeads, uint32_t maxKVCacheLen,
        uint32_t headDim, uint32_t maxDraftLen, int32_t rewindDraftTokenCommonCount,
        uint32_t layerCount, uint32_t blockSize = 128);

    // ==================== MLA Operations ====================
    // Vulkan port of the CuTe-DSL Blackwell MLA decode FMHA
    // (cute_dsl_mla.py::_run_mla_decode). q:[B,S,H,D], kv:[numPages,pageSize,D],
    // pageTable:[B,maxPages], cacheSeqs:[B], out:[B,S,H,dLatent] (D=dLatent+dRope).
    common::VulkanResult dispatchMlaFmha(
        void* q, void* kv, void* pageTable, void* cacheSeqs, void* output,
        uint32_t numHeads, uint32_t seqQLen, uint32_t batchSize,
        uint32_t dLatent, uint32_t dRope, uint32_t pageSize, uint32_t maxPages,
        float softmaxScale, uint32_t slidingWindow, uint32_t storageType,
        float kvScale, uint32_t blockSize = 64);

    // Context/prefill phase of the same FMHA: one workgroup per query token (b,h,s),
    // causal mask applied (query s attends cacheSeqs[b] + s+1 tokens). Same IO + a `causal` flag.
    common::VulkanResult dispatchMlaFmhaPrefill(
        void* q, void* kv, void* pageTable, void* cacheSeqs, void* output,
        uint32_t numHeads, uint32_t seqQLen, uint32_t batchSize,
        uint32_t dLatent, uint32_t dRope, uint32_t pageSize, uint32_t maxPages,
        bool causal, float softmaxScale, uint32_t slidingWindow, uint32_t storageType,
        float kvScale, uint32_t blockSize = 64);

    // ==================== Utility Operations ====================
    common::VulkanResult dispatchFill(
        void* output, float value, size_t elementCount,
        uint32_t blockSize = 256);

    // ==================== Activation Operations ====================
    // SiLU (swish): out = x * sigmoid(x)
    common::VulkanResult dispatchSilu(
        void* input, void* output,
        size_t elementCount,
        uint32_t blockSize = 256);

    // Sigmoid: out = 1 / (1 + exp(-x))
    common::VulkanResult dispatchSigmoid(
        void* input, void* output,
        size_t elementCount,
        uint32_t blockSize = 256);

    // GELU (tanh approximation)
    common::VulkanResult dispatchGelu(
        void* input, void* output,
        size_t elementCount,
        uint32_t blockSize = 256);

    // Relu: out = max(x, 0)
    common::VulkanResult dispatchRelu(
        void* input, void* output,
        size_t elementCount,
        uint32_t blockSize = 256);

    // SwiGLU: input [M, 2*H], output [M, H] = up * sigmoid(gate)
    common::VulkanResult dispatchSwiglu(
        void* input, void* output,
        uint32_t hiddenDim, uint32_t tokenCount,
        uint32_t blockSize = 256);

    // Fused sigmoid+mul: out = a * sigmoid(b), elementwise
    common::VulkanResult dispatchSigmoidMul(
        void* a, void* b, void* output,
        size_t elementCount,
        uint32_t blockSize = 256);

    // Elementwise multiply: out = a * b, elementwise
    common::VulkanResult dispatchElementwiseMul(
        void* a, void* b, void* output,
        size_t elementCount,
        uint32_t blockSize = 256);

    // Row-wise scale: out[i,j] = in[i,j] * scale[i]
    common::VulkanResult dispatchScaleRows(
        void* input, void* scale, void* output,
        uint32_t rows, uint32_t cols,
        uint32_t blockSize = 256);

    // Cast: out = (targetDtype)in, elementwise type conversion
    // targetDtype: 0=fp32 (no-op), 1=fp16, 2=bf16
    common::VulkanResult dispatchCast(
        void* input, void* output,
        size_t elementCount, int32_t targetDtype,
        uint32_t blockSize = 256);

    // IndexAdd: output[indices[i]] += values[i], with atomic adds
    common::VulkanResult dispatchIndexAdd(
        void* output, void* indices, void* values,
        uint32_t outputRows, uint32_t valueRows, uint32_t cols,
        uint32_t blockSize = 256);

    // Top-K per row (general purpose): input [rows, cols], output indices [rows, topk]
    common::VulkanResult dispatchTopKGeneral(
        void* input, void* outputIndices, void* outputValues,
        uint32_t rows, uint32_t cols, uint32_t topk,
        uint32_t blockSize = 64);

private:
    std::shared_ptr<common::VulkanContext> mContext;
    std::shared_ptr<common::VulkanShaderCompiler> mCompiler;
    std::shared_ptr<VulkanKernelRegistry> mKernelRegistry;

    // Command buffer management
    VkCommandBuffer acquireCommandBuffer();
    void submitAndFree(VkCommandBuffer cmdBuf);

    // Descriptor set helpers
    bool allocateDescriptorSet(VkDescriptorSetLayout setLayout, VkDescriptorSet* pSet);
    void freeDescriptorSet(VkDescriptorSet set);

    VkCommandPool mCmdPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> mCmdBuffers;
    VkDescriptorPool mDescriptorPool = VK_NULL_HANDLE;
};

} // namespace kernels
TRTLLM_NAMESPACE_END

#endif // VULKAN_KERNEL_REGISTRY_H
