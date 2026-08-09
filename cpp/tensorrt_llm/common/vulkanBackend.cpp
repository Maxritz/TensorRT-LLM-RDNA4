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

#include "tensorrt_llm/common/vulkanBackend.h"

#include <vulkan/vulkan.h>

#include <cstring>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

// ==================== Singleton Management ====================

std::shared_ptr<VulkanBackend> VulkanBackend::getInstance()
{
    static std::shared_ptr<VulkanBackend> instance = std::make_shared<VulkanBackend>();
    return instance;
}

VulkanBackend::~VulkanBackend()
{
    if (mActive)
    {
        mDispatcher.reset();
        mMemoryManager.reset();
        mContext.reset();
        mRuntime->shutdown();
        mRuntime.reset();
        mActive = false;
    }
}

bool VulkanBackend::initialize(uint32_t gpuID)
{
    if (mActive)
    {
        return true; // Already initialized
    }

    // Get or create Vulkan runtime
    mRuntime = VulkanRuntime::getInstance();
    if (!mRuntime)
    {
        mLastError = "Failed to get Vulkan runtime instance";
        return false;
    }

    if (!mRuntime->isInitialized())
    {
        VulkanResult result = mRuntime->initialize(gpuID);
        if (result != VulkanResult::SUCCESS)
        {
            mLastError = std::string("Failed to initialize Vulkan runtime: ") + VulkanContext::getErrorString(result);
            return false;
        }
    }

    mContext = mRuntime->getContext();
    if (!mContext || mContext->getDevice() == VK_NULL_HANDLE)
    {
        mLastError = "Failed to get Vulkan context";
        return false;
    }

    mMemoryManager = VulkanMemoryManager::create(mContext);
    if (!mMemoryManager)
    {
        mLastError = "Failed to create Vulkan memory manager";
        return false;
    }

    mDispatcher = std::make_shared<kernels::VulkanKernelDispatcher>(mContext);
    if (!mDispatcher)
    {
        mLastError = "Failed to create kernel dispatcher";
        return false;
    }

    // Compile all kernels
    kernels::VulkanKernelRegistry::getInstance()->initialize();
    kernels::VulkanKernelRegistry::getInstance()->compileAll();

    mActive = true;
    TLLM_LOG_INFO("Vulkan backend initialized for device: %s",
        mContext->getDeviceInfo().deviceName.c_str());

    return true;
}

// ==================== Memory Operations ====================

void* VulkanBackend::malloc(size_t byteCount)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mMemoryManager)
    {
        backend->mLastError = "Vulkan backend not initialized";
        return nullptr;
    }

    void* ptr = nullptr;
    VulkanResult result = backend->mMemoryManager->malloc(byteCount, &ptr, true);
    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("malloc failed: ") + VulkanContext::getErrorString(result);
        return nullptr;
    }

    return ptr;
}

void VulkanBackend::free(void* ptr)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mMemoryManager)
    {
        return;
    }

    backend->mMemoryManager->free(ptr);
}

void* VulkanBackend::memcpyHostToDevice(void* dstDevice, void const* srcHost, size_t byteCount)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mMemoryManager)
    {
        backend->mLastError = "Vulkan backend not initialized";
        return nullptr;
    }

    VulkanResult result = backend->mMemoryManager->memcpyHtoD(dstDevice, srcHost, byteCount);
    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("memcpy HtoD failed: ") + VulkanContext::getErrorString(result);
        return nullptr;
    }

    return dstDevice;
}

void* VulkanBackend::memcpyDeviceToHost(void* dstHost, void const* srcDevice, size_t byteCount)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mMemoryManager)
    {
        backend->mLastError = "Vulkan backend not initialized";
        return nullptr;
    }

    VulkanResult result = backend->mMemoryManager->memcpyDtoH(dstHost, srcDevice, byteCount);
    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("memcpy DtoH failed: ") + VulkanContext::getErrorString(result);
        return nullptr;
    }

    return dstHost;
}

void* VulkanBackend::memcpyDeviceToDevice(void* dstDevice, void const* srcDevice, size_t byteCount)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mMemoryManager)
    {
        backend->mLastError = "Vulkan backend not initialized";
        return nullptr;
    }

    VulkanResult result = backend->mMemoryManager->memcpyDtoD(dstDevice, srcDevice, byteCount);
    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("memcpy DtoD failed: ") + VulkanContext::getErrorString(result);
        return nullptr;
    }

    return dstDevice;
}

void VulkanBackend::memset(void* ptr, int value, size_t byteCount)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mMemoryManager)
    {
        return;
    }

    backend->mMemoryManager->memset(ptr, value, byteCount);
}

// ==================== Kernel Launch ====================

bool VulkanBackend::launchRmsNorm(void* input, void* gamma, void* beta, void* output,
                                  float eps, size_t hiddenDim, size_t tokenCount,
                                  void* stream)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchRmsNorm(
        input, gamma, beta, output, eps, hiddenDim, tokenCount);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("RMS norm launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchElementwiseAdd(void* a, void* b, void* output, size_t elementCount,
                                         void* stream)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchElementwiseAdd(
        a, b, output, elementCount);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("Elementwise add launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchFp16Gemm(void* a, void* b, void* output,
                                   uint32_t M, uint32_t N, uint32_t K,
                                   void* stream)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchFp16Gemm(
        a, b, output, M, N, K, false, false);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("FP16 GEMM launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchQ8_0Gemm(void* weight, void* activation, void* output,
                                   uint32_t M, uint32_t N, uint32_t K,
                                   uint32_t blocksPerRow,
                                   void* stream)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchQ8_0Gemm(
        weight, activation, output, M, N, K, blocksPerRow);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("Q8_0 GEMM launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchSilu(void* input, void* output,
                               size_t elementCount, void* stream)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchSilu(
        input, output, elementCount);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("SiLU launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchGelu(void* input, void* output,
                               size_t elementCount, void* stream)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchGelu(
        input, output, elementCount);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("GELU launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchSigmoid(void* input, void* output,
                                  size_t elementCount, void* stream)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchSigmoid(
        input, output, elementCount);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("Sigmoid launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchRelu(void* input, void* output,
                               size_t elementCount, void* stream)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchRelu(
        input, output, elementCount);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("ReLU launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchSwiglu(void* input, void* output,
                                 uint32_t hiddenDim, uint32_t tokenCount,
                                 void* stream)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchSwiglu(
        input, output, hiddenDim, tokenCount);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("SwiGLU launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchTopKGeneral(void* input, void* outputIndices,
                                      void* outputValues,
                                      uint32_t rows, uint32_t cols, uint32_t topk,
                                      void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchTopKGeneral(
        input, outputIndices, outputValues, rows, cols, topk);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("TopK launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchSoftmax(void* input, void* output,
                                  uint32_t batchSize, uint32_t numHeads, uint32_t seqLen,
                                  void* stream)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchSoftmax(
        input, output, batchSize, numHeads, seqLen);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("Softmax launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchAttention(void* q, void* k, void* v, void* output,
                                    uint32_t batchSize, uint32_t numHeads,
                                    uint32_t seqLenQ, uint32_t seqLenK, uint32_t headDim,
                                    bool causal,
                                    void* stream)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchAttention(
        q, k, v, output, batchSize, numHeads, seqLenQ, seqLenK, headDim, causal);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("Attention launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchTopk(void* scores, void* inputOffsets, void* outputOffsets,
                               void* topkIndices,
                               uint32_t topk, uint32_t numHeads, uint32_t batchSize,
                               uint32_t totalTokens, uint32_t totalOutputTokens,
                               void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchTopk(
        scores, inputOffsets, outputOffsets, topkIndices,
        topk, numHeads, batchSize, totalTokens, totalOutputTokens);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("Topk launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchSpecDecodeAccept(
    void* targetLogits, void* draftLogits, void* uniformRng, void* draftTokens,
    void* acceptCount, void* acceptedTokens, void* resampleProbs,
    uint32_t batchSize, uint32_t draftLen, uint32_t vocabSize,
    float temperature, float acceptProbFloor, void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchSpecDecodeAccept(
        targetLogits, draftLogits, uniformRng, draftTokens,
        acceptCount, acceptedTokens, resampleProbs,
        batchSize, draftLen, vocabSize, temperature, acceptProbFloor);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("SpecDecodeAccept launch failed: ") +
                              VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchTreeSpecBuild(
    void* parentListUvec2, void* selectedIndexUvec2, void* treeMask,
    void* positions, void* retrieveIndex, void* retrieveNextToken,
    void* retrieveNextSibling,
    uint32_t batchSize, uint32_t draftTokenNum, uint32_t topK,
    uint32_t depth, uint32_t numInt32PerRow, void*)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher) return false;

    VulkanResult r = backend->mDispatcher->dispatchTreeSpecBuild(
        parentListUvec2, selectedIndexUvec2, treeMask, positions, retrieveIndex,
        retrieveNextToken, retrieveNextSibling, batchSize, draftTokenNum, topK, depth,
        numInt32PerRow);
    if (r != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("TreeSpecBuild launch failed: ") +
                              VulkanContext::getErrorString(r);
        return false;
    }
    return true;
}

bool VulkanBackend::launchTreeSpecGreedyVerify(
    void* acceptIndex, void* acceptTokenNum, void* acceptToken,
    void* candidates, void* retrievePacked, void* targetPredict,
    void* treeValid,
    uint32_t batchSize, uint32_t numSpeculativeTokens, uint32_t numDraftTokens, void*)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher) return false;

    VulkanResult r = backend->mDispatcher->dispatchTreeSpecGreedyVerify(
        acceptIndex, acceptTokenNum, acceptToken, candidates, retrievePacked,
        targetPredict, treeValid, batchSize, numSpeculativeTokens, numDraftTokens);
    if (r != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("TreeSpecGreedyVerify launch failed: ") +
                              VulkanContext::getErrorString(r);
        return false;
    }
    return true;
}

bool VulkanBackend::launchTreeSpecRejection(
    void* acceptIndex, void* acceptTokenNum, void* acceptToken, void* draftTokens,
    void* targetProbs, void* retrieveNextToken, void* retrieveNextSibling,
    void* treeValid, void* rngSamples,
    uint32_t batchSize, uint32_t numSpeculativeTokens, uint32_t numDraftTokens,
    uint32_t vocabSize, uint32_t kMaxTriedPerLevel, void*)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
        return false;

    VulkanResult r = backend->mDispatcher->dispatchTreeSpecRejection(
        acceptIndex, acceptTokenNum, acceptToken, draftTokens, targetProbs,
        retrieveNextToken, retrieveNextSibling, treeValid, rngSamples,
        batchSize, numSpeculativeTokens, numDraftTokens, vocabSize, kMaxTriedPerLevel);
    if (r != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("TreeSpecRejection launch failed: ") +
                              VulkanContext::getErrorString(r);
        return false;
    }
    return true;
}

bool VulkanBackend::launchKVCacheUpdate2D(void* kvCacheK, void* kvCacheV,
    void* acceptedDraftTokensIndices2D, void* numAcceptedTokens,
    void* pastKeyValueLengths, void* rewindDraftTokenSeparateAdjustments,
    void* seqSlotRemapping,
    uint32_t batchSize, uint32_t numKVHeads, uint32_t maxKVCacheLen,
    uint32_t headDim, uint32_t maxDraftLen, int32_t rewindDraftTokenCommonCount,
    uint32_t layerCount, void*)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
        return false;

    VulkanResult r = backend->mDispatcher->dispatchKVCacheUpdate2D(
        kvCacheK, kvCacheV, acceptedDraftTokensIndices2D, numAcceptedTokens,
        pastKeyValueLengths, rewindDraftTokenSeparateAdjustments, seqSlotRemapping,
        batchSize, numKVHeads, maxKVCacheLen, headDim, maxDraftLen,
        rewindDraftTokenCommonCount, layerCount);
    if (r != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("KVCacheUpdate2D launch failed: ") +
                              VulkanContext::getErrorString(r);
        return false;
    }
    return true;
}

bool VulkanBackend::launchMlaFmha(void* q, void* kv, void* pageTable, void* cacheSeqs,
                                  void* output,
                                  uint32_t numHeads, uint32_t seqQLen, uint32_t batchSize,
                                  uint32_t dLatent, uint32_t dRope, uint32_t pageSize,
                                  uint32_t maxPages, float softmaxScale,
                                  uint32_t slidingWindow, uint32_t storageType,
                                  float kvScale, void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchMlaFmha(
        q, kv, pageTable, cacheSeqs, output,
        numHeads, seqQLen, batchSize, dLatent, dRope, pageSize, maxPages, softmaxScale,
        slidingWindow, storageType, kvScale);
    (void)slidingWindow; (void)storageType; (void)kvScale;

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("MLA FMHA launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchMlaFmhaPrefill(void* q, void* kv, void* pageTable, void* cacheSeqs,
                                         void* output,
                                         uint32_t numHeads, uint32_t seqQLen, uint32_t batchSize,
                                         uint32_t dLatent, uint32_t dRope, uint32_t pageSize,
                                         uint32_t maxPages, bool causal, float softmaxScale,
                                         uint32_t slidingWindow, uint32_t storageType,
                                         float kvScale, void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchMlaFmhaPrefill(
        q, kv, pageTable, cacheSeqs, output,
        numHeads, seqQLen, batchSize, dLatent, dRope, pageSize, maxPages, causal,
        softmaxScale, slidingWindow, storageType, kvScale);
    (void)slidingWindow; (void)storageType; (void)kvScale;

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("MLA FMHA prefill launch failed: ") +
                              VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

// ==================== Synchronization ====================

void VulkanBackend::streamSynchronize(void* stream)
{
    auto backend = getInstance();
    if (!backend->mActive)
    {
        return;
    }
}

void VulkanBackend::deviceSynchronize()
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mRuntime)
    {
        return;
    }

    backend->mRuntime->deviceSynchronize();
}

// ==================== Utility ====================

float VulkanBackend::getGpuUtilization()
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mRuntime)
    {
        return 0.0f;
    }

    auto tracker = backend->mRuntime->getUtilizationTracker();
    if (tracker)
    {
        return tracker->getUtilPercentage();
    }

    return 0.0f;
}

std::string VulkanBackend::getLastError() const
{
    return mLastError;
}

} // namespace common
TRTLLM_NAMESPACE_END
