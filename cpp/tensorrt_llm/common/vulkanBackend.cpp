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
