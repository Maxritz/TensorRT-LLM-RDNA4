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

#ifndef VULKAN_CONTEXT_H
#define VULKAN_CONTEXT_H

#include "tensorrt_llm/common/vulkanCommon.h"
#include <vulkan/vulkan.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

// Vulkan error codes that mirror CUDA error patterns
enum class VulkanResult
{
    SUCCESS = 0,
    INVALID_VALUE = 1,
    OUT_OF_MEMORY = 2,
    INVALID_HANDLE = 3,
    INITIALIZATION_FAILED = 4,
    FEATURE_NOT_PRESENT = 5,
    DEVICE_LOST = 6,
    UNKNOWN_ERROR = 100
};

struct VulkanLimits
{
    uint32_t maxComputeWorkGroupInvocations;
    uint32_t maxComputeWorkGroupCount[3];
    uint32_t maxComputeWorkGroupSize[3];
    uint32_t maxPushConstantsSize;
    uint32_t minStorageBufferOffsetAlignment;
    uint32_t minUniformBufferOffsetAlignment;
    size_t maxMemoryAllocationCount;
    size_t maxMemoryAllocationSize;
};

struct VulkanDeviceInfo
{
    std::string deviceName;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t apiVersion;
    uint32_t driverVersion;
    VkPhysicalDeviceType deviceType;
    bool hasCooperativeMatrix;
    bool hasFP16;
    bool hasBF16;
    uint32_t subgroupSize;
    uint32_t gpuQueueIndex;
    uint32_t computeQueueCount;
    VulkanLimits limits;
};

class VulkanContext : public std::enable_shared_from_this<VulkanContext>
{
public:
    static std::shared_ptr<VulkanContext> create(uint32_t gpuID = 0);

    ~VulkanContext();
    VulkanContext(VulkanContext const&) = delete;
    VulkanContext operator=(VulkanContext const&) = delete;
    VulkanContext(VulkanContext&&) = delete;
    VulkanContext operator=(VulkanContext&&) = delete;

    // Device info
    VulkanDeviceInfo const& getDeviceInfo() const { return mDeviceInfo; }

    // Core handles
    VkInstance getInstance() const { return mInstance; }
    VkPhysicalDevice getPhysicalDevice() const { return mPhysicalDevice; }
    VkDevice getDevice() const { return mDevice; }
    VkQueue getComputeQueue(uint32_t index = 0) const { return mComputeQueues.at(index); }
    uint32_t getComputeQueueCount() const { return static_cast<uint32_t>(mComputeQueues.size()); }

    // Queue family info
    uint32_t getComputeQueueFamilyIndex() const { return mComputeQueueFamilyIndex; }

    // Device properties
    bool supportsCooperativeMatrix() const { return mDeviceInfo.hasCooperativeMatrix; }
    bool supportsFP16() const { return mDeviceInfo.hasFP16; }
    bool supportsBF16() const { return mDeviceInfo.hasBF16; }
    uint32_t getSubgroupSize() const { return mDeviceInfo.subgroupSize; }

    // Validation
    VulkanResult getResult() const { return mLastResult; }
    void setLastResult(VulkanResult result) { mLastResult = result; }

    // Error string mapping
    static char const* getErrorString(VulkanResult result);

private:
    VulkanContext(uint32_t gpuID);

    VulkanResult initialize();
    void detectExtensions();
    void detectDeviceInfo(uint32_t gpuID);

    uint32_t mGpuID = 0;
    VkInstance mInstance = VK_NULL_HANDLE;
    VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
    VkDevice mDevice = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT mDebugMessenger = VK_NULL_HANDLE;

    std::vector<VkQueue> mComputeQueues;
    uint32_t mComputeQueueFamilyIndex = 0;

    VulkanDeviceInfo mDeviceInfo{};
    VulkanResult mLastResult = VulkanResult::SUCCESS;

    std::vector<const char*> mRequiredExtensions;
    std::vector<const char*> mRequiredLayers;
};

} // namespace common
TRTLLM_NAMESPACE_END

#endif // VULKAN_CONTEXT_H
