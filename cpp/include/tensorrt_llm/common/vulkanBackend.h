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
