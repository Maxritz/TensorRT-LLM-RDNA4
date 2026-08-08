/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    ://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/vulkanFence.h"

#include <vulkan/vulkan.h>

#include <cstring>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

// ==================== GPUFence Implementation ====================

GPUFence::GPUFence(std::shared_ptr<VulkanContext> const& ctx)
    : mContext(ctx)
{
    if (!mContext)
    {
        return;
    }

    VkDevice device = mContext->getDevice();

    // Create a fence for CPU-side synchronization
    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    // Create fence in signaled state (CUDA events start unsignaled, but we want to support both)
    // We'll use VK_FENCE_CREATE_SIGNALED_BIT for reuse pattern
    fenceCreateInfo.flags = 0; // Start unsignaled

    VkResult result = vkCreateFence(device, &fenceCreateInfo, nullptr, &mFence);
    if (result != VK_SUCCESS)
    {
        mContext->setLastResult(VulkanResult::INITIALIZATION_FAILED);
        return;
    }

    // Create a binary semaphore for GPU-GPU synchronization
    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    result = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &mSemaphore);
    if (result != VK_SUCCESS)
    {
        mContext->setLastResult(VulkanResult::INITIALIZATION_FAILED);
        vkDestroyFence(device, mFence, nullptr);
        mFence = VK_NULL_HANDLE;
        return;
    }

    mInitialized = true;
    mContext->setLastResult(VulkanResult::SUCCESS);
}

GPUFence::~GPUFence()
{
    if (mContext && mContext->getDevice() != VK_NULL_HANDLE)
    {
        VkDevice device = mContext->getDevice();
        if (mTimelineSemaphore != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(device, mTimelineSemaphore, nullptr);
        }
        if (mSemaphore != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(device, mSemaphore, nullptr);
        }
        if (mFence != VK_NULL_HANDLE)
        {
            vkDestroyFence(device, mFence, nullptr);
        }
    }
}

GPUFence::GPUFence(GPUFence&& other) noexcept
{
    moveFrom(other);
}

GPUFence& GPUFence::operator=(GPUFence&& other) noexcept
{
    if (this != &other)
    {
        // Clean up existing resources first
        if (mContext && mContext->getDevice() != VK_NULL_HANDLE)
        {
            VkDevice device = mContext->getDevice();
            if (mFence != VK_NULL_HANDLE)
                vkDestroyFence(device, mFence, nullptr);
            if (mSemaphore != VK_NULL_HANDLE)
                vkDestroySemaphore(device, mSemaphore, nullptr);
            if (mTimelineSemaphore != VK_NULL_HANDLE)
                vkDestroySemaphore(device, mTimelineSemaphore, nullptr);
        }
        moveFrom(other);
    }
    return *this;
}

void GPUFence::moveFrom(GPUFence& other) noexcept
{
    mContext = other.mContext;
    mFence = other.mFence;
    mSemaphore = other.mSemaphore;
    mTimelineSemaphore = other.mTimelineSemaphore;
    mTimelineValue = other.mTimelineValue;
    mInitialized = other.mInitialized;

    other.mFence = VK_NULL_HANDLE;
    other.mSemaphore = VK_NULL_HANDLE;
    other.mTimelineSemaphore = VK_NULL_HANDLE;
    other.mTimelineValue = 0;
    other.mInitialized = false;
}

VulkanResult GPUFence::record(VkQueue queue, uint32_t queueIndex)
{
    if (!mInitialized || mContext->getDevice() == VK_NULL_HANDLE)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    VkDevice device = mContext->getDevice();

    // Reset fence before reuse
    vkResetFences(device, 1, &mFence);

    // Signal the fence when the queue reaches this point
    VkResult result = vkQueueSubmit(queue, 0, nullptr, mFence);
    if (result != VK_SUCCESS)
    {
        mContext->setLastResult(VulkanResult::UNKNOWN_ERROR);
        return mContext->getResult();
    }

    // Signal the semaphore as well for GPU-GPU synchronization
    VkSemaphore submitSemaphore = mSemaphore;
    if (mTimelineSemaphore != VK_NULL_HANDLE)
    {
        submitSemaphore = mTimelineSemaphore;
        mTimelineValue++;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 0; // No command buffers, we're just signaling
    if (submitSemaphore != VK_NULL_HANDLE)
    {
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &submitSemaphore;
    }

    result = vkQueueSubmit(queue, 1, &submitInfo, mFence);
    if (result != VK_SUCCESS)
    {
        mContext->setLastResult(VulkanResult::UNKNOWN_ERROR);
        return mContext->getResult();
    }

    mContext->setLastResult(VulkanResult::SUCCESS);
    return VulkanResult::SUCCESS;
}

VulkanResult GPUFence::synchronize(uint64_t timeoutNs)
{
    if (!mInitialized || mContext->getDevice() == VK_NULL_HANDLE)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    VkDevice device = mContext->getDevice();

    VkResult result = vkWaitForFences(device, 1, &mFence, VK_TRUE, timeoutNs);

    if (result == VK_TIMEOUT)
    {
        mContext->setLastResult(VulkanResult::UNKNOWN_ERROR);
        return VulkanResult::UNKNOWN_ERROR;
    }
    else if (result != VK_SUCCESS)
    {
        mContext->setLastResult(VulkanResult::DEVICE_LOST);
        return VulkanResult::DEVICE_LOST;
    }

    mContext->setLastResult(VulkanResult::SUCCESS);
    return VulkanResult::SUCCESS;
}

bool GPUFence::isReady()
{
    if (!mInitialized || mContext->getDevice() == VK_NULL_HANDLE)
    {
        return false;
    }

    VkDevice device = mContext->getDevice();

    VkResult result = vkGetFenceStatus(device, mFence);
    return (result == VK_SUCCESS);
}

VulkanResult GPUFence::wait(VkQueue queue)
{
    if (!mInitialized || mContext->getDevice() == VK_NULL_HANDLE)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    if (mSemaphore == VK_NULL_HANDLE)
    {
        mContext->setLastResult(VulkanResult::INITIALIZATION_FAILED);
        return VulkanResult::INITIALIZATION_FAILED;
    }

    VkSemaphore waitSemaphore = mSemaphore;
    if (mTimelineSemaphore != VK_NULL_HANDLE)
    {
        waitSemaphore = mTimelineSemaphore;
    }

    VkPipelineStageFlags stageFlags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &waitSemaphore;
    submitInfo.pWaitDstStageMask = &stageFlags;
    submitInfo.commandBufferCount = 0;

    VkResult result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS)
    {
        mContext->setLastResult(VulkanResult::UNKNOWN_ERROR);
        return mContext->getResult();
    }

    mContext->setLastResult(VulkanResult::SUCCESS);
    return VulkanResult::SUCCESS;
}

VulkanResult GPUFence::createTimelineSemaphore()
{
    if (!mContext || mContext->getDevice() == VK_NULL_HANDLE)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    if (mTimelineSemaphore != VK_NULL_HANDLE)
    {
        return VulkanResult::SUCCESS; // Already created
    }

    VkSemaphoreTypeCreateInfo timelineCreateTimeInfo{};
    timelineCreateTimeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineCreateTimeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineCreateTimeInfo.initialValue = 0;

    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = &timelineCreateTimeInfo;

    VkResult result = vkCreateSemaphore(mContext->getDevice(), &semaphoreCreateInfo, nullptr, &mTimelineSemaphore);
    if (result != VK_SUCCESS)
    {
        mContext->setLastResult(VulkanResult::FEATURE_NOT_PRESENT);
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    mContext->setLastResult(VulkanResult::SUCCESS);
    return VulkanResult::SUCCESS;
}

// ==================== GPUFencePool Implementation ====================

GPUFencePool::GPUFencePool(std::shared_ptr<VulkanContext> const& ctx, uint32_t initialSize)
    : mContext(ctx)
    , mMaxFences(initialSize)
{
    for (uint32_t i = 0; i < initialSize; ++i)
    {
        auto fence = std::make_shared<GPUFence>(ctx);
        if (fence && fence->getFence() != VK_NULL_HANDLE)
        {
            mAvailableFences.push_back(fence);
        }
    }
}

GPUFencePool::~GPUFencePool()
{
    // All shared_ptr cleanup handles destruction
}

std::shared_ptr<GPUFence> GPUFencePool::acquire()
{
    if (!mAvailableFences.empty())
    {
        auto fence = mAvailableFences.back();
        mAvailableFences.pop_back();
        mUsedFences.push_back(fence);
        return fence;
    }

    // Create a new fence if pool is empty
    auto fence = std::make_shared<GPUFence>(mContext);
    if (fence && fence->getFence() != VK_NULL_HANDLE)
    {
        mUsedFences.push_back(fence);
        return fence;
    }

    return nullptr;
}

void GPUFencePool::release(std::shared_ptr<GPUFence>& fence)
{
    if (fence && fence->getFence() != VK_NULL_HANDLE)
    {
        // Check if fence is ready before returning to pool
        // In production, you might want to defer this
        auto it = std::find(mUsedFences.begin(), mUsedFences.end(), fence);
        if (it != mUsedFences.end())
        {
            mUsedFences.erase(it);
            mAvailableFences.push_back(fence);
        }
    }
}

void GPUFencePool::gc()
{
    // Move completed fences from used to available
    auto it = mUsedFences.begin();
    while (it != mUsedFences.end())
    {
        if ((*it)->isReady())
        {
            mAvailableFences.push_back(*it);
            it = mUsedFences.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

VulkanResult GPUFencePool::synchronizeAll(uint64_t timeoutNs)
{
    for (auto& fence : mUsedFences)
    {
        if (!fence->isReady())
        {
            VulkanResult result = fence->synchronize(timeoutNs);
            if (result != VulkanResult::SUCCESS)
            {
                return result;
            }
        }
    }

    for (auto& fence : mAvailableFences)
    {
        // Reset available fences
        VkFence fenceHandle = fence->getFence();
        if (fenceHandle != VK_NULL_HANDLE)
        {
            vkResetFences(mContext->getDevice(), 1, &fenceHandle);
        }
    }

    return VulkanResult::SUCCESS;
}

VulkanResult GPUFencePool::waitIdle(uint64_t timeoutNs)
{
    if (mContext->getDevice() == VK_NULL_HANDLE)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    // First wait for all used fences
    VulkanResult result = synchronizeAll(timeoutNs);
    if (result != VulkanResult::SUCCESS)
    {
        return result;
    }

    // Then wait for all queues to be idle
    vkDeviceWaitIdle(mContext->getDevice());

    mContext->setLastResult(VulkanResult::SUCCESS);
    return VulkanResult::SUCCESS;
}

} // namespace common
TRTLLM_NAMESPACE_END
