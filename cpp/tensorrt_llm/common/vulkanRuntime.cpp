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

#include "tensorrt_llm/common/vulkanRuntime.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstring>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

// ==================== Singleton Instance Management ====================

std::shared_ptr<VulkanRuntime> VulkanRuntime::getInstance()
{
    static std::shared_ptr<VulkanRuntime> instance = std::make_shared<VulkanRuntime>();
    return instance;
}

VulkanRuntime::VulkanRuntime()
    : mInitialized(false)
{
}

VulkanRuntime::~VulkanRuntime()
{
    shutdown();
}

// ==================== Device Management ====================

VulkanResult VulkanRuntime::initialize(uint32_t gpuID)
{
    if (mInitialized)
    {
        return VulkanResult::SUCCESS;
    }

    // Create Vulkan context
    mContext = VulkanContext::create(gpuID);
    if (!mContext || mContext->getDevice() == VK_NULL_HANDLE)
    {
        TLLM_LOG_ERROR("Failed to create Vulkan context for GPU %u", gpuID);
        return VulkanResult::INITIALIZATION_FAILED;
    }

    mGpuID = gpuID;
    mInitialized = true;

    // Initialize fence pool
    mFencePool = std::make_shared<GPUFencePool>(mContext, 16);

    // Initialize utilization tracker
    mUtilizationTracker = std::make_unique<GpuUtilizationTracker>(mContext);

    // Cache alignment values
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(mContext->getPhysicalDevice(), &props);
    mMinStorageBufferAlignment = std::max(props.limits.minStorageBufferOffsetAlignment, VkDeviceSize{256});
    mMinUniformBufferAlignment = std::max(props.limits.minUniformBufferOffsetAlignment, VkDeviceSize{256});

    TLLM_LOG_INFO("Vulkan runtime initialized for GPU %u: %s",
        gpuID, mContext->getDeviceInfo().deviceName.c_str());

    return VulkanResult::SUCCESS;
}

void VulkanRuntime::shutdown()
{
    if (mUtilizationTracker)
    {
        mUtilizationTracker.reset();
    }

    if (mContext && mContext->getDevice() != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(mContext->getDevice());
    }

    mContext.reset();
    mInitialized = false;
}

// ==================== Memory Management ====================

VulkanResult VulkanRuntime::allocateDeviceMemory(size_t size, VkDeviceMemory* pMemory, VkDeviceSize* offset)
{
    if (!mInitialized || !pMemory)
    {
        return VulkanResult::INVALID_VALUE;
    }

    VkDevice device = mContext->getDevice();

    // Find a memory type that supports device-local access
    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(mContext->getPhysicalDevice(), &memProperties);

    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
    {
        if ((memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (memProperties.memoryTypes[i].heapIndex < memProperties.memoryHeapCount) &&
            (memProperties.memoryHeaps[memProperties.memoryTypes[i].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
        {
            memoryTypeIndex = i;
            break;
        }
    }

    if (memoryTypeIndex == UINT32_MAX)
    {
        TLLM_LOG_ERROR("No device-local memory type found");
        return VulkanResult::OUT_OF_MEMORY;
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    VkResult result = vkAllocateMemory(device, &allocInfo, nullptr, pMemory);
    if (result != VK_SUCCESS)
    {
        TLLM_LOG_ERROR("Failed to allocate %zu bytes of device memory: %d", size, result);
        return translateVkResult(result);
    }

    if (offset)
    {
        *offset = 0;
    }

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanRuntime::freeDeviceMemory(VkDeviceMemory memory)
{
    if (!mInitialized || memory == VK_NULL_HANDLE)
    {
        return VulkanResult::INVALID_VALUE;
    }

    vkFreeMemory(mContext->getDevice(), memory, nullptr);
    return VulkanResult::SUCCESS;
}

VulkanResult VulkanRuntime::hostToDevice(void* dstDevice, void const* srcHost, size_t byteCount)
{
    if (!mInitialized || !dstDevice || !srcHost || byteCount == 0)
    {
        return VulkanResult::INVALID_VALUE;
    }

    // In Vulkan, we need to use staging buffers for host-to-device transfers
    // This implementation uses a simple approach with mapped memory

    VkDevice device = mContext->getDevice();
    VkPhysicalDevice physDev = mContext->getPhysicalDevice();

    // Find a memory type that is host visible and coherent
    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProperties);

    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
    {
        if ((memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        {
            memoryTypeIndex = i;
            break;
        }
    }

    if (memoryTypeIndex == UINT32_MAX)
    {
        TLLM_LOG_ERROR("No host-visible coherent memory type found");
        return VulkanResult::OUT_OF_MEMORY;
    }

    // Create staging buffer
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = byteCount;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer);
    if (result != VK_SUCCESS)
    {
        return translateVkResult(result);
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = byteCount;
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    result = vkAllocateMemory(device, &allocInfo, nullptr, &stagingMemory);
    if (result != VK_SUCCESS)
    {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        return translateVkResult(result);
    }

    vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);

    // Copy data to staging buffer
    void* mappedData = nullptr;
    result = vkMapMemory(device, stagingMemory, 0, byteCount, 0, &mappedData);
    if (result != VK_SUCCESS)
    {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
        return translateVkResult(result);
    }

    memcpy(mappedData, srcHost, byteCount);
    vkUnmapMemory(device, stagingMemory);

    // Copy from staging to destination
    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandPool = VK_NULL_HANDLE; // We'll use the default pool
    cmdAllocInfo.commandBufferCount = 1;

    // Get a command buffer from the context's pool
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    // For now, we need to create a temporary command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = mContext->getComputeQueueFamilyIndex();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    result = vkCreateCommandPool(device, &poolInfo, nullptr, &cmdPool);
    if (result != VK_SUCCESS)
    {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
        return translateVkResult(result);
    }

    cmdAllocInfo.commandPool = cmdPool;
    VkCommandBuffer cmdBuf;
    vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuf);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = byteCount;

    vkCmdCopyBuffer(cmdBuf, stagingBuffer, (VkBuffer)dstDevice, 1, &copyRegion);

    vkEndCommandBuffer(cmdBuf);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;

    vkQueueSubmit(mContext->getComputeQueue(0), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(mContext->getComputeQueue(0));

    // Cleanup
    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanRuntime::copyMemory(void* dst, void const* src, size_t byteCount)
{
    if (!mInitialized || !dst || !src || byteCount == 0)
    {
        return VulkanResult::INVALID_VALUE;
    }

    VkCommandBuffer cmdBuf;
    VkCommandPool cmdPool;

    // Allocate a one-time command buffer
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = mContext->getComputeQueueFamilyIndex();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkResult result = vkCreateCommandPool(mContext->getDevice(), &poolInfo, nullptr, &cmdPool);
    if (result != VK_SUCCESS)
    {
        return translateVkResult(result);
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmdPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    result = vkAllocateCommandBuffers(mContext->getDevice(), &allocInfo, &cmdBuf);
    if (result != VK_SUCCESS)
    {
        vkDestroyCommandPool(mContext->getDevice(), cmdPool, nullptr);
        return translateVkResult(result);
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = byteCount;

    vkCmdCopyBuffer(cmdBuf, (VkBuffer)src, (VkBuffer)dst, 1, &copyRegion);

    vkEndCommandBuffer(cmdBuf);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;

    vkQueueSubmit(mContext->getComputeQueue(0), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(mContext->getComputeQueue(0));

    vkDestroyCommandPool(mContext->getDevice(), cmdPool, nullptr);

    return VulkanResult::SUCCESS;
}

// ==================== Module/Shader Management ====================

VulkanResult VulkanRuntime::loadShaderModule(void const* shaderData, size_t dataSize, VkShaderModule* pShaderModule)
{
    if (!mInitialized || !shaderData || !pShaderModule || dataSize == 0)
    {
        return VulkanResult::INVALID_VALUE;
    }

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = dataSize;
    createInfo.pCode = static_cast<uint32_t const*>(shaderData);

    VkResult result = vkCreateShaderModule(mContext->getDevice(), &createInfo, nullptr, pShaderModule);
    if (result != VK_SUCCESS)
    {
        return translateVkResult(result);
    }

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanRuntime::getEntryPoint(VkShaderModule module, char const* name, VkPipelineShaderStageCreateInfo* pStageInfo)
{
    if (!mInitialized || module == VK_NULL_HANDLE || !name || !pStageInfo)
    {
        return VulkanResult::INVALID_VALUE;
    }

    pStageInfo->sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pStageInfo->stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pStageInfo->module = module;
    pStageInfo->pName = name;
    pStageInfo->pNext = nullptr;

    return VulkanResult::SUCCESS;
}

void VulkanRuntime::unloadShaderModule(VkShaderModule module)
{
    if (module != VK_NULL_HANDLE)
    {
        vkDestroyShaderModule(mContext->getDevice(), module, nullptr);
    }
}

// ==================== Pipeline Management ====================

VulkanResult VulkanRuntime::createComputePipeline(VkShaderModule shaderModule, char const* entryPoint,
                                                  VkPipelineLayout layout, VkPipeline* pPipeline)
{
    if (!mInitialized || shaderModule == VK_NULL_HANDLE || !entryPoint || !pPipeline)
    {
        return VulkanResult::INVALID_VALUE;
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = entryPoint;
    pipelineInfo.layout = layout;
    pipelineInfo.basePipelineIndex = -1;

    VkResult result = vkCreateComputePipelines(mContext->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, pPipeline);
    if (result != VK_SUCCESS)
    {
        return translateVkResult(result);
    }

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanRuntime::destroyComputePipeline(VkPipeline pipeline)
{
    if (pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(mContext->getDevice(), pipeline, nullptr);
    }
    return VulkanResult::SUCCESS;
}

// ==================== Kernel Launch ====================

VulkanResult VulkanRuntime::launchKernel(VkPipeline pipeline, VkCommandBuffer cmdBuf,
                                         uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                                         uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                                         std::vector<VkDescriptorSet> const& descriptors,
                                         std::vector<uint32_t> const& constants)
{
    if (!mInitialized || pipeline == VK_NULL_HANDLE || cmdBuf == VK_NULL_HANDLE)
    {
        return VulkanResult::INVALID_VALUE;
    }

    VkResult result = vkBeginCommandBuffer(cmdBuf, nullptr);
    if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST)
    {
        // Command buffer might already be in recording state
        // Check if it's already begun
    }

    // Bind pipeline
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    // Bind descriptor sets
    if (!descriptors.empty())
    {
        vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
            mContext->getDeviceInfo().hasCooperativeMatrix ?
                VK_NULL_HANDLE : VK_NULL_HANDLE, // Pipeline layout would be needed here
            0, static_cast<uint32_t>(descriptors.size()), descriptors.data(),
            0, nullptr);
    }

    // Push constants
    if (!constants.empty())
    {
        vkCmdPushConstants(cmdBuf, VK_NULL_HANDLE, // Pipeline layout needed
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(uint32_t) * constants.size(), constants.data());
    }

    // Dispatch - translate CUDA grid/block dimensions to Vulkan workgroups
    // CUDA: gridX blocks, each with blockX threads
    // Vulkan: gridX workgroups, each with blockX invocations
    vkCmdDispatch(cmdBuf, gridX, gridY, gridZ);

    result = vkEndCommandBuffer(cmdBuf);
    if (result != VK_SUCCESS)
    {
        return translateVkResult(result);
    }

    // Submit command buffer
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;

    result = vkQueueSubmit(mContext->getComputeQueue(0), 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS)
    {
        return translateVkResult(result);
    }

    return VulkanResult::SUCCESS;
}

// ==================== Stream Management ====================

VulkanResult VulkanRuntime::createStream(uint32_t flags, VkCommandPool* pPool)
{
    if (!mInitialized || !pPool)
    {
        return VulkanResult::INVALID_VALUE;
    }

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = mContext->getComputeQueueFamilyIndex();

    // Translate CUDA stream flags
    if (flags & 0x1 /* cudaStreamNonBlocking */)
    {
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    }
    else
    {
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                         VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    }

    VkResult result = vkCreateCommandPool(mContext->getDevice(), &poolInfo, nullptr, pPool);
    if (result != VK_SUCCESS)
    {
        return translateVkResult(result);
    }

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanRuntime::destroyStream(VkCommandPool pool)
{
    if (pool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(mContext->getDevice(), pool, nullptr);
    }
    return VulkanResult::SUCCESS;
}

VulkanResult VulkanRuntime::streamSynchronize(VkCommandPool pool)
{
    if (!mInitialized)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    // Use bounded fence-based wait instead of vkQueueWaitIdle to avoid permanent hangs.
    const uint64_t timeoutNs = 10'000'000'000ULL; // 10s

    if (mFencePool)
    {
        return mFencePool->synchronizeAll(timeoutNs);
    }

    // Fallback: if no fence pool, use queue wait idle (no timeout possible)
    VkResult result = vkQueueWaitIdle(mContext->getComputeQueue(0));
    if (result != VK_SUCCESS)
    {
        return translateVkResult(result);
    }

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanRuntime::deviceSynchronize()
{
    if (!mInitialized)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    const uint64_t timeoutNs = 10'000'000'000ULL; // 10s

    if (mFencePool)
    {
        VulkanResult result = mFencePool->waitIdle(timeoutNs);
        if (result != VulkanResult::SUCCESS)
        {
            return result;
        }
    }
    else
    {
        // No fence pool, use device wait idle (no timeout)
        VkResult result = vkDeviceWaitIdle(mContext->getDevice());
        if (result != VK_SUCCESS)
        {
            return translateVkResult(result);
        }
    }

    return VulkanResult::SUCCESS;
}

// ==================== Fence Management ====================

VkResult VulkanRuntime::allocateFence(std::shared_ptr<GPUFence>* ppFence)
{
    if (!ppFence)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    *ppFence = mFencePool->acquire();
    return (*ppFence)->getFence() != VK_NULL_HANDLE ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED;
}

void VulkanRuntime::returnFence(std::shared_ptr<GPUFence>& fence)
{
    if (fence)
    {
        mFencePool->release(fence);
    }
}

// ==================== Query/Utilization ====================

std::shared_ptr<GpuUtilizationTracker> VulkanRuntime::getUtilizationTracker()
{
    if (!mUtilizationTracker)
    {
        mUtilizationTracker = std::make_unique<GpuUtilizationTracker>(mContext);
    }
    return mUtilizationTracker ? std::shared_ptr<GpuUtilizationTracker>(mUtilizationTracker.get(), [](auto*){}) : nullptr;
}

// ==================== Helper Functions ====================

VulkanResult VulkanRuntime::translateVkResult(VkResult vkResult)
{
    switch (vkResult)
    {
        case VK_SUCCESS:
            return VulkanResult::SUCCESS;
        case VK_ERROR_OUT_OF_HOST_MEMORY:
            return VulkanResult::OUT_OF_MEMORY;
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            return VulkanResult::OUT_OF_MEMORY;
        case VK_ERROR_INITIALIZATION_FAILED:
            return VulkanResult::INITIALIZATION_FAILED;
        case VK_ERROR_DEVICE_LOST:
            return VulkanResult::DEVICE_LOST;
        case VK_ERROR_VALIDATION_FAILED_EXT:
            return VulkanResult::INVALID_VALUE;
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            return VulkanResult::FEATURE_NOT_PRESENT;
        case VK_ERROR_FEATURE_NOT_PRESENT:
            return VulkanResult::FEATURE_NOT_PRESENT;
        case VK_ERROR_INCOMPATIBLE_DRIVER:
            return VulkanResult::UNKNOWN_ERROR;
#ifdef VK_ERROR_INCOMPATIBLE_CLIENT_DRIVER
        case VK_ERROR_INCOMPATIBLE_CLIENT_DRIVER:
            return VulkanResult::UNKNOWN_ERROR;
#endif
#ifdef VK_ERROR_INVALID_MESSAGE_OBJECT_KHR
        case VK_ERROR_INVALID_MESSAGE_OBJECT_KHR:
            return VulkanResult::INVALID_VALUE;
#endif
#ifdef VK_ERROR_INVALID_OBJECT_KHR
        case VK_ERROR_INVALID_OBJECT_KHR:
            return VulkanResult::INVALID_HANDLE;
#endif
        case VK_ERROR_FRAGMENTATION:
            return VulkanResult::OUT_OF_MEMORY;
        case VK_ERROR_UNKNOWN:
        default:
            return VulkanResult::UNKNOWN_ERROR;
    }
}

} // namespace common
TRTLLM_NAMESPACE_END
