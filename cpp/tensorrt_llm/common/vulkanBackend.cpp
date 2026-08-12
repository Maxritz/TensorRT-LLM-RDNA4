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

bool VulkanBackend::launchSigmoidMul(void* a, void* b, void* output,
                                     size_t elementCount, void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchSigmoidMul(
        a, b, output, elementCount);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("SigmoidMul launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchElementwiseMul(void* a, void* b, void* output,
                                         size_t elementCount, void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchElementwiseMul(
        a, b, output, elementCount);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("ElementwiseMul launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchScaleRows(void* input, void* scale, void* output,
                                    uint32_t rows, uint32_t cols,
                                    void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchScaleRows(
        input, scale, output, rows, cols);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("ScaleRows launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchCast(void* input, void* output,
                               size_t elementCount, int32_t targetDtype,
                               void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchCast(
        input, output, elementCount, targetDtype);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("Cast launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchIndexAdd(void* output, void* indices, void* values,
                                   uint32_t outputRows, uint32_t valueRows,
                                   uint32_t cols, void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchIndexAdd(
        output, indices, values, outputRows, valueRows, cols);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("IndexAdd launch failed: ") + VulkanContext::getErrorString(result);
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

bool VulkanBackend::launchLayerNorm(void* input, void* gamma, void* beta, void* output,
                                    float eps, size_t hiddenDim, size_t tokenCount,
                                    void* stream)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchLayerNorm(
        input, gamma, beta, output, eps, hiddenDim, tokenCount);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("LayerNorm launch failed: ") + VulkanContext::getErrorString(result);
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

bool VulkanBackend::launchAppendPagedKVCache(
    void* append_key, void* append_value,
    void* batch_indices, void* positions,
    void* kv_cache_k, void* kv_cache_v,
    void* kv_indices, void* kv_indptr, void* kv_last_page_len,
    uint32_t page_size, uint32_t num_kv_heads, uint32_t head_dim,
    uint32_t n_tokens, uint32_t batch_size,
    void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    // Compute max_pages from kv_indptr: the last entry gives total_pages
    // We need to pass max_pages to the shader
    // kv_indptr[batch_size] = total number of pages used
    uint32_t total_pages = 0;
    if (kv_indptr) {
        auto* indptr = static_cast<uint32_t*>(kv_indptr);
        total_pages = indptr[batch_size];
    }
    uint32_t max_pages = total_pages > 0 ? total_pages : 1;

    VulkanResult result = backend->mDispatcher->dispatchAppendPagedKVCache(
        append_key, append_value,
        batch_indices, positions,
        kv_cache_k, kv_cache_v,
        kv_indices, kv_indptr, kv_last_page_len,
        page_size, num_kv_heads, head_dim, n_tokens, batch_size, max_pages);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("AppendPagedKVCache launch failed: ") + VulkanContext::getErrorString(result);
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
    if (!backend->mActive || !backend->mRuntime)
    {
        return;
    }

    backend->mRuntime->streamSynchronize();
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

bool VulkanBackend::launchGather(void* src, void* indices, void* output,
                                 size_t numIndices, void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchGather(
        src, indices, output, static_cast<uint32_t>(numIndices));

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("Gather launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchFill(void* output, float value,
                               size_t elementCount, void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchFill(
        output, value, elementCount);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("Fill launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchCompareEq(void* input, void* output,
                                    float threshold, size_t elementCount,
                                    void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchCompareEq(
        input, output, threshold, elementCount);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("CompareEq launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

bool VulkanBackend::launchMhcGemm(void* x, void* wT, void* y, void* r,
                                  uint32_t M, uint32_t N, uint32_t K, uint32_t tileN,
                                  void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
        return false;

    VulkanResult result = backend->mDispatcher->dispatchMhcGemm(
        x, wT, y, r, M, N, K, tileN);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("MHC GEMM launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }
    return true;
}

bool VulkanBackend::launchMhcFuse(void* yAcc, void* rAcc, void* residual,
                                  void* hcScale, void* hcBase,
                                  void* postMix, void* combMix, void* layerInput, void* normW,
                                  uint32_t M, uint32_t K, uint32_t hiddenSize,
                                  float rmsEps, float hcPreEps, float hcSinkhornEps,
                                  float hcPostMultValue, int32_t sinkhornRepeat,
                                  uint32_t numSplits, bool applyNorm, float normEps,
                                  void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
        return false;

    VulkanResult result = backend->mDispatcher->dispatchMhcFuse(
        yAcc, rAcc, residual, hcScale, hcBase,
        postMix, combMix, layerInput, normW,
        M, K, hiddenSize,
        rmsEps, hcPreEps, hcSinkhornEps, hcPostMultValue,
        sinkhornRepeat, numSplits, applyNorm, normEps);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("MHC fuse launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }
    return true;
}

bool VulkanBackend::launchMtpSSMCache(
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
    void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
        return false;

    VulkanResult result = backend->mDispatcher->dispatchMtpSSMCache(
        state, x, dt, A, B, C, out, intermediateStates,
        D, z, dtBias, ssmBatchIndices, interSsmBatchIndices, retrieveParentToken,
        bs, nheads, headDim, ssmDim, ngroups,
        cacheSteps, padSlotId, disableStateUpdate,
        hasD, hasZ, hasDtBias,
        hasSsmBatchIndices, hasInterSsmBatchIndices, hasParentToken,
        dtSoftplus,
        strideNheadsHdimSsmDim, strideHdimSsmDim,
        strideCacheNheadsHdim, strideNheadsHdim,
        strideCacheNgroupsSsmDim, strideNgroupsSsmDim);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("MTP SSM cache launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }
    return true;
}

bool VulkanBackend::launchCompressDecode(
    void* kvScore, void* ape, void* pagedKv, void* pagedScore,
    void* blockTableKv, void* blockTableScore, void* output,
    void* kvLens, void* cuSeqLens, void* cuKvComp,
    uint32_t batchSize, uint32_t pageSize, uint32_t maxBlocks,
    uint32_t headDim, uint32_t compressRatio, uint32_t nextN,
    uint32_t kvScoreElemBytes, uint32_t stateElemBytes, uint32_t outElemBytes,
    void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
        return false;

    VulkanResult result = backend->mDispatcher->dispatchCompressDecode(
        kvScore, ape, pagedKv, pagedScore,
        blockTableKv, blockTableScore, output,
        kvLens, cuSeqLens, cuKvComp,
        batchSize, pageSize, maxBlocks,
        headDim, compressRatio, nextN,
        kvScoreElemBytes, stateElemBytes, outElemBytes);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("CompressDecode launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }
    return true;
}

bool VulkanBackend::launchCompressPrefill(
    void* kvScore, void* ape, void* pagedKv, void* pagedScore,
    void* blockTableKv, void* blockTableScore, void* output,
    void* kvLens, void* startPos, void* cuSeqLens, void* cuKvComp,
    uint32_t batchSize, uint32_t pageSize, uint32_t maxBlocks,
    uint32_t headDim, uint32_t compressRatio, uint32_t maxOutputs,
    uint32_t kvScoreElemBytes, uint32_t stateElemBytes, uint32_t outElemBytes,
    void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
        return false;

    VulkanResult result = backend->mDispatcher->dispatchCompressPrefill(
        kvScore, ape, pagedKv, pagedScore,
        blockTableKv, blockTableScore, output,
        kvLens, startPos, cuSeqLens, cuKvComp,
        batchSize, pageSize, maxBlocks,
        headDim, compressRatio, maxOutputs,
        kvScoreElemBytes, stateElemBytes, outElemBytes);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("CompressPrefill launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }
    return true;
}

bool VulkanBackend::launchCompressPostproc(
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
    void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
        return false;

    VulkanResult result = backend->mDispatcher->dispatchCompressPostproc(
        kvComp, rmsWeight, rmsEps,
        cosSinTable, positionIds,
        nopeDim, ropeDim,
        kvCache,
        numOutputs, cuKvComp, startPos, blockOffsets,
        compressedMask,
        batchSize, tokensPerBlock, headDim,
        maxBlocksPerSeq, elemBytes, totalTokens,
        cacheScaleType, rotateActivation,
        quantOutput, scaleOutput);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("CompressPostproc launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }
    return true;
}

bool VulkanBackend::launchGatedDeltaRule(
    void* aLog, void* dtBias, void* a, void* b,
    void* q, void* k, void* v,
    void* initialState, void* initIndices, void* output,
    uint32_t N, uint32_t T, uint32_t H, uint32_t HV, uint32_t V, uint32_t K,
    bool hasInitialState, bool disableStateUpdate,
    float scale, float softplusBeta, float softplusThreshold, bool useL2Norm,
    void* /*stream*/)
{
    auto backend = getInstance();
    if (!backend->mActive || !backend->mDispatcher)
    {
        return false;
    }

    VulkanResult result = backend->mDispatcher->dispatchGatedDeltaRule(
        aLog, dtBias, a, b, q, k, v,
        initialState, initIndices, output,
        N, T, H, HV, V, K,
        hasInitialState, disableStateUpdate,
        scale, softplusBeta, softplusThreshold, useL2Norm);

    if (result != VulkanResult::SUCCESS)
    {
        backend->mLastError = std::string("GatedDeltaRule launch failed: ") + VulkanContext::getErrorString(result);
        return false;
    }

    return true;
}

} // namespace common
TRTLLM_NAMESPACE_END
