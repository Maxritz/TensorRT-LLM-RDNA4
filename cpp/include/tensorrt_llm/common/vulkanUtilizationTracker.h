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

#ifndef VULKAN_UTILIZATION_TRACKER_H
#define VULKAN_UTILIZATION_TRACKER_H

#include "tensorrt_llm/common/vulkanCommon.h"
#include "tensorrt_llm/common/vulkanContext.h"
#include "tensorrt_llm/common/vulkanFence.h"

#include <vulkan/vulkan.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

// GpuUtilizationTracker: Maps to CUDA utilization tracking via query pools
// Uses timestamp queries to measure GPU busy time and calculate utilization
class GpuUtilizationTracker
{
public:
    explicit GpuUtilizationTracker(std::shared_ptr<VulkanContext> const& ctx);
    ~GpuUtilizationTracker();

    GpuUtilizationTracker(GpuUtilizationTracker const&) = delete;
    GpuUtilizationTracker& operator=(GpuUtilizationTracker const&) = delete;

    // Record entry/exit timestamps around work (analogous to CUDA events)
    // Call these on command buffers to measure active time
    void recordEntry(VkCommandBuffer cmdBuf);
    void recordExit(VkCommandBuffer cmdBuf);

    // Get utilization percentage between last sample and now
    // Returns a value between 0.0f and 1.0f (0% to 100%)
    float getUtilPercentage();

    // Get raw timing data
    struct UtilizationStats
    {
        float utilizationPct;     // 0.0 to 1.0
        uint64_t activeTimeNs;    // GPU busy time
        uint64_t totalTimeNs;     // Total elapsed time
        uint64_t timestampPeriodNs; // Timer resolution
    };
    UtilizationStats getUtilizationStats();

    // Reset all counters
    void reset();

    // Get timestamp period in nanoseconds
    uint64_t getTimestampPeriod() const { return mTimestampPeriodNs; }

private:
    void updateFromDevice();

    std::shared_ptr<VulkanContext> mContext;
    VkQueryPool mTimestampPool = VK_NULL_HANDLE;
    VkQueryPool mDurationPool = VK_NULL_HANDLE;

    static constexpr uint32_t MAX_QUERIES = 256;
    uint32_t mQueryIndex = 0;

    uint64_t mTimestampPeriodNs = 1;
    uint64_t mActiveTimeNs = 0;
    uint64_t mTotalTimeNs = 0;
    std::atomic<bool> mDirty{false};

    std::chrono::high_resolution_clock::time_point mLastSampleTime;
    std::mutex mMutex;

    VkResult createQueryPools();
    void resetQueryPools();
};

// StreamUtilizationTracker: Tracks utilization per-stream (like CUDA streams)
class StreamUtilizationTracker
{
public:
    explicit StreamUtilizationTracker(std::shared_ptr<VulkanContext> const& ctx);
    ~StreamUtilizationTracker() = default;

    // Associate a command buffer with a stream for tracking
    void beginTracking(VkCommandBuffer cmdBuf, uint32_t streamId);
    void endTracking(VkCommandBuffer cmdBuf, uint32_t streamId);

    // Get utilization for a specific stream
    float getStreamUtilization(uint32_t streamId);

    // Get average utilization across all streams
    float getOverallUtilization();

private:
    std::shared_ptr<VulkanContext> mContext;
    std::vector<std::unique_ptr<GpuUtilizationTracker>> mTrackers;
    std::vector<uint32_t> mActiveStreams;
};

} // namespace common
TRTLLM_NAMESPACE_END

#endif // VULKAN_UTILIZATION_TRACKER_H
