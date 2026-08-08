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

#include "tensorrt_llm/common/vulkanUtilizationTracker.h"

#include <vulkan/vulkan.h>

#include <cstring>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

// ==================== GpuUtilizationTracker Implementation ====================

GpuUtilizationTracker::GpuUtilizationTracker(std::shared_ptr<VulkanContext> const& ctx)
    : mContext(ctx)
    , mLastSampleTime(std::chrono::high_resolution_clock::now())
{
    if (!mContext || mContext->getDevice() == VK_NULL_HANDLE)
    {
        return;
    }

    // Get timestamp period from physical device properties
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(mContext->getPhysicalDevice(), &props);
    mTimestampPeriodNs = props.limits.timestampPeriod;

    if (createQueryPools() != VK_SUCCESS)
    {
        TLLM_LOG_WARNING("Failed to create query pools for utilization tracking");
    }
}

GpuUtilizationTracker::~GpuUtilizationTracker()
{
    if (mContext && mContext->getDevice() != VK_NULL_HANDLE)
    {
        if (mTimestampPool != VK_NULL_HANDLE)
        {
            vkDestroyQueryPool(mContext->getDevice(), mTimestampPool, nullptr);
        }
        if (mDurationPool != VK_NULL_HANDLE)
        {
            vkDestroyQueryPool(mContext->getDevice(), mDurationPool, nullptr);
        }
    }
}

VkResult GpuUtilizationTracker::createQueryPools()
{
    VkDevice device = mContext->getDevice();

    // Create timestamp query pool for measuring time
    VkQueryPoolCreateInfo timestampPoolInfo{};
    timestampPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    timestampPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    timestampPoolInfo.queryCount = MAX_QUERIES;

    VkResult result = vkCreateQueryPool(device, &timestampPoolInfo, nullptr, &mTimestampPool);
    if (result != VK_SUCCESS)
    {
        return result;
    }

    // Create duration query pool for measuring active time
    VkQueryPoolCreateInfo durationPoolInfo{};
    durationPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    durationPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP; // Use timestamps for duration
    durationPoolInfo.queryCount = MAX_QUERIES;

    // Some implementations don't support duration query type, that's OK
    // We can compute durations from timestamps
    vkCreateQueryPool(device, &durationPoolInfo, nullptr, &mDurationPool);

    // Reset the query pools individually. The tracker does not own a
    // VkCommandPool, so there is nothing to call vkResetCommandPool on here.
    if (mTimestampPool != VK_NULL_HANDLE)
    {
        vkResetQueryPool(device, mTimestampPool, 0, MAX_QUERIES);
    }
    if (mDurationPool != VK_NULL_HANDLE)
    {
        vkResetQueryPool(device, mDurationPool, 0, MAX_QUERIES);
    }

    return VK_SUCCESS;
}

void GpuUtilizationTracker::resetQueryPools()
{
    if (mContext->getDevice() == VK_NULL_HANDLE)
    {
        return;
    }

    VkDevice device = mContext->getDevice();

    if (mTimestampPool != VK_NULL_HANDLE)
    {
        vkResetQueryPool(device, mTimestampPool, 0, MAX_QUERIES);
    }
    if (mDurationPool != VK_NULL_HANDLE)
    {
        vkResetQueryPool(device, mDurationPool, 0, MAX_QUERIES);
    }

    mActiveTimeNs = 0;
    mTotalTimeNs = 0;
    mQueryIndex = 0;
}

void GpuUtilizationTracker::recordEntry(VkCommandBuffer cmdBuf)
{
    if (mTimestampPool == VK_NULL_HANDLE || mQueryIndex >= MAX_QUERIES - 1)
    {
        // Query pool exhausted, get results first
        updateFromDevice();
        if (mQueryIndex >= MAX_QUERIES - 1)
        {
            return;
        }
    }

    vkCmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, mTimestampPool, mQueryIndex++);
}

void GpuUtilizationTracker::recordExit(VkCommandBuffer cmdBuf)
{
    if (mTimestampPool == VK_NULL_HANDLE || mQueryIndex >= MAX_QUERIES - 1)
    {
        updateFromDevice();
        if (mQueryIndex >= MAX_QUERIES - 1)
        {
            return;
        }
    }

    // Use BOTTOM_OF_PIPE stage to measure full work time
    vkCmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, mTimestampPool, mQueryIndex++);
}

void GpuUtilizationTracker::updateFromDevice()
{
    if (mTimestampPool == VK_NULL_HANDLE || mQueryIndex == 0)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(mMutex);

    VkDevice device = mContext->getDevice();

    // Get timestamp results
    std::vector<uint64_t> timestamps(mQueryIndex, 0);
    VkResult result = vkGetQueryPoolResults(device, mTimestampPool, 0, mQueryIndex,
        sizeof(uint64_t) * mQueryIndex, timestamps.data(),
        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

    if (result == VK_NOT_READY || result == VK_ERROR_DEVICE_LOST)
    {
        return;
    }

    // Calculate active time from paired timestamps
    for (uint32_t i = 0; i + 1 < mQueryIndex; i += 2)
    {
        uint64_t entryTs = timestamps[i];
        uint64_t exitTs = timestamps[i + 1];

        if (entryTs > 0 || exitTs > 0)
        {
            uint64_t durationNs = (exitTs - entryTs) * mTimestampPeriodNs;
            mActiveTimeNs += durationNs;
        }
    }

    // Calculate total elapsed time
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now - mLastSampleTime).count();
    mTotalTimeNs += static_cast<uint64_t>(elapsedNs);
    mLastSampleTime = now;

    // Reset query pool for next batch
    vkResetQueryPool(device, mTimestampPool, 0, mQueryIndex);
    mQueryIndex = 0;
    mDirty.store(false);
}

float GpuUtilizationTracker::getUtilPercentage()
{
    updateFromDevice();
    return static_cast<float>(mActiveTimeNs) / static_cast<float>(mTotalTimeNs);
}

GpuUtilizationTracker::UtilizationStats GpuUtilizationTracker::getUtilizationStats()
{
    updateFromDevice();

    UtilizationStats stats{};
    stats.timestampPeriodNs = mTimestampPeriodNs;
    stats.activeTimeNs = mActiveTimeNs;
    stats.totalTimeNs = mTotalTimeNs;

    if (mTotalTimeNs > 0)
    {
        stats.utilizationPct = static_cast<float>(mActiveTimeNs) / static_cast<float>(mTotalTimeNs);
    }
    else
    {
        stats.utilizationPct = 0.0f;
    }

    return stats;
}

void GpuUtilizationTracker::reset()
{
    resetQueryPools();
    mLastSampleTime = std::chrono::high_resolution_clock::now();
}

// ==================== StreamUtilizationTracker Implementation ====================

StreamUtilizationTracker::StreamUtilizationTracker(std::shared_ptr<VulkanContext> const& ctx)
    : mContext(ctx)
{
}

void StreamUtilizationTracker::beginTracking(VkCommandBuffer cmdBuf, uint32_t streamId)
{
    if (streamId >= mTrackers.size())
    {
        mTrackers.resize(streamId + 1);
    }

    if (!mTrackers[streamId])
    {
        mTrackers[streamId] = std::make_unique<GpuUtilizationTracker>(mContext);
        mActiveStreams.push_back(streamId);
    }

    mTrackers[streamId]->recordEntry(cmdBuf);
}

void StreamUtilizationTracker::endTracking(VkCommandBuffer cmdBuf, uint32_t streamId)
{
    if (streamId < mTrackers.size() && mTrackers[streamId])
    {
        mTrackers[streamId]->recordExit(cmdBuf);
    }
}

float StreamUtilizationTracker::getStreamUtilization(uint32_t streamId)
{
    if (streamId < mTrackers.size() && mTrackers[streamId])
    {
        return mTrackers[streamId]->getUtilPercentage();
    }
    return 0.0f;
}

float StreamUtilizationTracker::getOverallUtilization()
{
    float totalUtil = 0.0f;
    uint32_t activeCount = 0;

    for (uint32_t streamId : mActiveStreams)
    {
        if (streamId < mTrackers.size() && mTrackers[streamId])
        {
            totalUtil += mTrackers[streamId]->getUtilPercentage();
            activeCount++;
        }
    }

    return activeCount > 0 ? totalUtil / static_cast<float>(activeCount) : 0.0f;
}

} // namespace common
TRTLLM_NAMESPACE_END
