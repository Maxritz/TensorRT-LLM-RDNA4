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

#ifndef VULKAN_BACKEND_H
#define VULKAN_BACKEND_H

#include "tensorrt_llm/common/vulkanContext.h"
#include "tensorrt_llm/common/vulkanRuntime.h"
#include "tensorrt_llm/common/vulkanMemoryAllocator.h"
#include "tensorrt_llm/kernels/vulkanKernelRegistry.h"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

// VulkanBackend: Transparent backend that intercepts CUDA calls and redirects to Vulkan
// This enables running TensorRT-LLM on AMD GPUs without modifying application code
class VulkanBackend
{
public:
    static std::shared_ptr<VulkanBackend> getInstance();

    ~VulkanBackend();

    VulkanBackend(VulkanBackend const&) = delete;
    VulkanBackend& operator=(VulkanBackend const&) = delete;

    // Initialize the backend
    // Returns true if Vulkan backend was successfully initialized
    bool initialize(uint32_t gpuID = 0);

    // Check if Vulkan backend is active
    bool isActive() const { return mActive; }

    // Get the underlying runtime for direct access
    std::shared_ptr<VulkanRuntime> getRuntime() const { return mRuntime; }
    std::shared_ptr<VulkanContext> getContext() const { return mContext; }
    std::shared_ptr<VulkanMemoryManager> getMemoryManager() const { return mMemoryManager; }

    // ==================== Memory Operations ====================
    // These mirror CUDA API signatures for transparent replacement
    static void* malloc(size_t byteCount);
    static void free(void* ptr);
    static void* memcpyHostToDevice(void* dstDevice, void const* srcHost, size_t byteCount);
    static void* memcpyDeviceToHost(void* dstHost, void const* srcDevice, size_t byteCount);
    static void* memcpyDeviceToDevice(void* dstDevice, void const* srcDevice, size_t byteCount);
    static void memset(void* ptr, int value, size_t byteCount);

    // ==================== Kernel Launch ====================
    // These mirror CUDA kernel launch patterns
    static bool launchRmsNorm(void* input, void* gamma, void* beta, void* output,
                             float eps, size_t hiddenDim, size_t tokenCount,
                             void* stream = nullptr);

    static bool launchElementwiseAdd(void* a, void* b, void* output, size_t elementCount,
                                     void* stream = nullptr);

    static bool launchFp16Gemm(void* a, void* b, void* output,
                               uint32_t M, uint32_t N, uint32_t K,
                               void* stream = nullptr);

    static bool launchQ8_0Gemm(void* weight, void* activation, void* output,
                               uint32_t M, uint32_t N, uint32_t K,
                               uint32_t blocksPerRow = 0,
                               void* stream = nullptr);

    static bool launchSilu(void* input, void* output,
                           size_t elementCount, void* stream = nullptr);

    static bool launchSigmoid(void* input, void* output,
                              size_t elementCount, void* stream = nullptr);

    static bool launchGelu(void* input, void* output,
                           size_t elementCount, void* stream = nullptr);

    static bool launchRelu(void* input, void* output,
                           size_t elementCount, void* stream = nullptr);

    static bool launchSwiglu(void* input, void* output,
                             uint32_t hiddenDim, uint32_t tokenCount,
                             void* stream = nullptr);

    static bool launchSigmoidMul(void* a, void* b, void* output,
                                size_t elementCount, void* stream = nullptr);

    static bool launchElementwiseMul(void* a, void* b, void* output,
                                    size_t elementCount, void* stream = nullptr);

    static bool launchScaleRows(void* input, void* scale, void* output,
                                uint32_t rows, uint32_t cols,
                                void* stream = nullptr);

    static bool launchCast(void* input, void* output,
                           size_t elementCount, int32_t targetDtype,
                           void* stream = nullptr);

     static bool launchIndexAdd(void* output, void* indices, void* values,
                                uint32_t outputRows, uint32_t valueRows,
                                uint32_t cols, void* stream = nullptr);

    static bool launchGather(void* src, void* indices, void* output,
                             size_t numIndices, void* stream = nullptr);

    static bool launchFill(void* output, float value,
                           size_t elementCount, void* stream = nullptr);

    static bool launchCompareEq(void* input, void* output,
                                float threshold, size_t elementCount,
                                void* stream = nullptr);

     static bool launchLayerNorm(void* input, void* gamma, void* beta, void* output,
                                 float eps, size_t hiddenDim, size_t tokenCount,
                                 void* stream = nullptr);

     static bool launchAppendPagedKVCache(
         void* append_key, void* append_value,
         void* batch_indices, void* positions,
         void* kv_cache_k, void* kv_cache_v,
         void* kv_indices, void* kv_indptr, void* kv_last_page_len,
         uint32_t page_size, uint32_t num_kv_heads, uint32_t head_dim,
         uint32_t n_tokens, uint32_t batch_size,
         void* stream = nullptr);

    static bool launchTopKGeneral(void* input, void* outputIndices,
                                  void* outputValues,
                                  uint32_t rows, uint32_t cols, uint32_t topk,
                                  void* stream = nullptr);

    static bool launchSoftmax(void* input, void* output,
                              uint32_t batchSize, uint32_t numHeads, uint32_t seqLen,
                              void* stream = nullptr);

    static bool launchAttention(void* q, void* k, void* v, void* output,
                                 uint32_t batchSize, uint32_t numHeads,
                                 uint32_t seqLenQ, uint32_t seqLenK, uint32_t headDim,
                                 bool causal,
                                 void* stream = nullptr);

    static bool launchTopk(void* scores, void* inputOffsets, void* outputOffsets,
                           void* topkIndices,
                           uint32_t topk, uint32_t numHeads, uint32_t batchSize,
                           uint32_t totalTokens, uint32_t totalOutputTokens,
                           void* stream = nullptr);

    static bool launchSpecDecodeAccept(
        void* targetLogits, void* draftLogits, void* uniformRng, void* draftTokens,
        void* acceptCount, void* acceptedTokens, void* resampleProbs,
        uint32_t batchSize, uint32_t draftLen, uint32_t vocabSize,
        float temperature, float acceptProbFloor, void* stream = nullptr);

    static bool launchTreeSpecBuild(
        void* parentListUvec2, void* selectedIndexUvec2,
        void* treeMask, void* positions, void* retrieveIndex,
        void* retrieveNextToken, void* retrieveNextSibling,
        uint32_t batchSize, uint32_t draftTokenNum, uint32_t topK,
        uint32_t depth, uint32_t numInt32PerRow, void* stream = nullptr);

    static bool launchTreeSpecGreedyVerify(
        void* acceptIndex, void* acceptTokenNum, void* acceptToken,
        void* candidates, void* retrievePacked, void* targetPredict,
        void* treeValid,
         uint32_t batchSize, uint32_t numSpeculativeTokens, uint32_t numDraftTokens,
         void* stream = nullptr);

    static bool launchTreeSpecRejection(
        void* acceptIndex, void* acceptTokenNum, void* acceptToken,
        void* draftTokens, void* targetProbs, void* retrieveNextToken,
        void* retrieveNextSibling, void* treeValid, void* rngSamples,
        uint32_t batchSize, uint32_t numSpeculativeTokens, uint32_t numDraftTokens,
         uint32_t vocabSize, uint32_t kMaxTriedPerLevel, void* stream = nullptr);

    static bool launchKVCacheUpdate2D(void* kvCacheK, void* kvCacheV,
        void* acceptedDraftTokensIndices2D, void* numAcceptedTokens,
        void* pastKeyValueLengths, void* rewindDraftTokenSeparateAdjustments,
        void* seqSlotRemapping,
        uint32_t batchSize, uint32_t numKVHeads, uint32_t maxKVCacheLen,
        uint32_t headDim, uint32_t maxDraftLen, int32_t rewindDraftTokenCommonCount,
        uint32_t layerCount, void* stream = nullptr);

    static bool launchMlaFmha(void* q, void* kv, void* pageTable, void* cacheSeqs,
                              void* output,
                              uint32_t numHeads, uint32_t seqQLen, uint32_t batchSize,
                              uint32_t dLatent, uint32_t dRope, uint32_t pageSize,
                              uint32_t maxPages, float softmaxScale,
                              uint32_t slidingWindow = 0, uint32_t storageType = 0,
                              float kvScale = 1.0f,
                              void* stream = nullptr);

    static bool launchMlaFmhaPrefill(void* q, void* kv, void* pageTable, void* cacheSeqs,
                                     void* output,
                                     uint32_t numHeads, uint32_t seqQLen, uint32_t batchSize,
                                     uint32_t dLatent, uint32_t dRope, uint32_t pageSize,
                                     uint32_t maxPages, bool causal, float softmaxScale,
                                     uint32_t slidingWindow = 0, uint32_t storageType = 0,
                                     float kvScale = 1.0f,
                                     void* stream = nullptr);

    // ==================== MHC Operations ====================
    static bool launchMhcGemm(void* x, void* wT, void* y, void* r,
                              uint32_t M, uint32_t N, uint32_t K, uint32_t tileN,
                              void* stream = nullptr);

    static bool launchMhcFuse(void* yAcc, void* rAcc, void* residual,
                              void* hcScale, void* hcBase,
                              void* postMix, void* combMix, void* layerInput, void* normW,
                              uint32_t M, uint32_t K, uint32_t hiddenSize,
                              float rmsEps, float hcPreEps, float hcSinkhornEps,
                              float hcPostMultValue, int32_t sinkhornRepeat,
                              uint32_t numSplits, bool applyNorm, float normEps,
                              void* stream = nullptr);

    // ==================== Mamba2 MTP SSM Cache ====================
    static bool launchMtpSSMCache(
        void* state, void* x, void* dt, void* A,
        void* B, void* C, void* out, void* intermediateStates,
        void* D, void* z, void* dtBias,
        void* ssmBatchIndices, void* interSsmBatchIndices, void* retrieveParentToken,
        uint32_t bs, uint32_t nheads, uint32_t headDim, uint32_t ssmDim, uint32_t ngroups,
        int cacheSteps, int padSlotId, bool disableStateUpdate,
        bool hasD, bool hasZ, bool hasDtBias,
        bool hasSsmBatchIndices, bool hasInterSsmBatchIndices, bool hasParentToken,
        bool dtSoftplus,
        uint32_t strideNheadsHdimSsmDim, uint32_t strideHdimSsmDim,
        uint32_t strideCacheNheadsHdim, uint32_t strideNheadsHdim,
        uint32_t strideCacheNgroupsSsmDim, uint32_t strideNgroupsSsmDim,
        void* stream = nullptr);

    // ==================== KV Cache Compression ====================
    static bool launchCompressDecode(
        void* kvScore, void* ape, void* pagedKv, void* pagedScore,
        void* blockTableKv, void* blockTableScore, void* output,
        void* kvLens, void* cuSeqLens, void* cuKvComp,
        uint32_t batchSize, uint32_t pageSize, uint32_t maxBlocks,
        uint32_t headDim, uint32_t compressRatio, uint32_t nextN,
        uint32_t kvScoreElemBytes, uint32_t stateElemBytes, uint32_t outElemBytes,
        void* stream = nullptr);

    static bool launchCompressPrefill(
        void* kvScore, void* ape, void* pagedKv, void* pagedScore,
        void* blockTableKv, void* blockTableScore, void* output,
        void* kvLens, void* startPos, void* cuSeqLens, void* cuKvComp,
        uint32_t batchSize, uint32_t pageSize, uint32_t maxBlocks,
        uint32_t headDim, uint32_t compressRatio, uint32_t maxOutputs,
        uint32_t kvScoreElemBytes, uint32_t stateElemBytes, uint32_t outElemBytes,
        void* stream = nullptr);

    static bool launchCompressPostproc(
        void* kvComp, void* rmsWeight, float rmsEps,
        void* cosSinTable, void* positionIds,
        int32_t nopeDim, int32_t ropeDim,
        void* kvCache,
        void* numOutputs, void* cuKvComp, void* startPos, void* blockOffsets,
        void* compressedMask,
        uint32_t batchSize, uint32_t tokensPerBlock, uint32_t headDim,
        uint32_t maxBlocksPerSeq, uint32_t elemBytes, uint32_t totalTokens,
        int32_t cacheScaleType, bool rotateActivation,
        void* quantOutput, void* scaleOutput,
        void* stream = nullptr);

    // ==================== FLA Gated Delta Rule ====================
    static bool launchGatedDeltaRule(
        void* aLog, void* dtBias, void* a, void* b,
        void* q, void* k, void* v,
        void* initialState, void* initIndices, void* output,
        uint32_t N, uint32_t T, uint32_t H, uint32_t HV, uint32_t V, uint32_t K,
        bool hasInitialState, bool disableStateUpdate,
        float scale, float softplusBeta, float softplusThreshold, bool useL2Norm,
        void* stream = nullptr);

    // ==================== Synchronization ====================
    static void streamSynchronize(void* stream = nullptr);
    static void deviceSynchronize();

    // ==================== Utility ====================
    // Get current GPU utilization
    float getGpuUtilization();

    // Get error string for last operation
    std::string getLastError() const;

    // ==================== Internal Access ====================
    VulkanBackend() = default;
    bool mActive = false;
    std::shared_ptr<VulkanRuntime> mRuntime;
    std::shared_ptr<VulkanContext> mContext;
    std::shared_ptr<VulkanMemoryManager> mMemoryManager;
    std::shared_ptr<kernels::VulkanKernelDispatcher> mDispatcher;

    std::string mLastError;
};

// Helper macros to check for Vulkan backend at runtime
#define TLLM_VULKAN_BACKEND_ACTIVE() (tensorrt_llm::common::VulkanBackend::getInstance()->isActive())

#define TLLM_VULKAN_IF_AVAILABLE(cuda_call, vulkan_call) \
    do { \
        if (TLLM_VULKAN_BACKEND_ACTIVE()) { \
            vulkan_call; \
        } else { \
            cuda_call; \
        } \
    } while(0)

} // namespace common
TRTLLM_NAMESPACE_END

#endif // VULKAN_BACKEND_H
