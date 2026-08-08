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

#ifndef VULKAN_FENCE_H
#define VULKAN_FENCE_H

#include "tensorrt_llm/common/vulkanCommon.h"
#include "tensorrt_llm/common/vulkanContext.h"

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

// GPUFence: Maps to CUDA event/fence functionality
// Provides synchronization primitives for GPU command ordering
class GPUFence
{
public:
    explicit GPUFence(std::shared_ptr<VulkanContext> const& ctx);
    ~GPUFence();

    GPUFence(GPUFence const&) = delete;
    GPUFence& operator=(GPUFence const&) = delete;
    GPUFence(GPUFence&& other) noexcept;
    GPUFence& operator=(GPUFence&& other) noexcept;

    // CUDA-equivalent methods
    // cuEventRecord -> record the fence on a specific queue
    VulkanResult record(VkQueue queue, uint32_t queueIndex = 0);

    // cuEventSynchronize -> wait for fence to complete
    VulkanResult synchronize(uint64_t timeoutNs = UINT64_MAX);

    // cuEventQuery -> check if fence has completed (non-blocking)
    bool isReady();

    // cuStreamWaitExternalSemaphoresAsync -> insert wait on this fence
    VulkanResult wait(VkQueue queue);

    // cuFenceCreate (CUDA graph) -> create a timeline semaphore-based fence
    VulkanResult createTimelineSemaphore();

    // Get the underlying Vulkan fence handle
    VkFence getFence() const { return mFence; }
    VkSemaphore getSemaphore() const { return mSemaphore; }

    // Timeline semaphore support for CUDA graph-like operations
    VkSemaphore getTimelineSemaphore() const { return mTimelineSemaphore; }
    uint64_t getTimelineValue() const { return mTimelineValue; }

private:
    void moveFrom(GPUFence& other) noexcept;

    std::shared_ptr<VulkanContext> mContext;
    VkFence mFence = VK_NULL_HANDLE;
    VkSemaphore mSemaphore = VK_NULL_HANDLE;
    VkSemaphore mTimelineSemaphore = VK_NULL_HANDLE;
    uint64_t mTimelineValue = 0;
    bool mInitialized = false;
};

// GPUFencePool: Manages a pool of fences for stream synchronization
// Similar to CUDA stream-ordered operations
class GPUFencePool
{
public:
    explicit GPUFencePool(std::shared_ptr<VulkanContext> const& ctx, uint32_t initialSize = 16);
    ~GPUFencePool();

    // Acquire a fence from the pool (reusable)
    std::shared_ptr<GPUFence> acquire();

    // Return a fence to the pool
    void release(std::shared_ptr<GPUFence>& fence);

    // Cleanup unused fences
    void gc();

    // Synchronize all fences in the pool
    VulkanResult synchronizeAll(uint64_t timeoutNs = UINT64_MAX);

    // Wait for all submitted work to complete
    VulkanResult waitIdle(uint64_t timeoutNs = UINT64_MAX);

private:
    std::shared_ptr<VulkanContext> mContext;
    std::vector<std::shared_ptr<GPUFence>> mAvailableFences;
    std::vector<std::shared_ptr<GPUFence>> mUsedFences;
    uint32_t mMaxFences;
};

} // namespace common
TRTLLM_NAMESPACE_END

#endif // VULKAN_FENCE_H
