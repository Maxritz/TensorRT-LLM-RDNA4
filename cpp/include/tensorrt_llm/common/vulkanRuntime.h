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

#ifndef VULKAN_RUNTIME_H
#define VULKAN_RUNTIME_H

#include "tensorrt_llm/common/vulkanContext.h"
#include "tensorrt_llm/common/vulkanFence.h"
#include "tensorrt_llm/common/vulkanUtilizationTracker.h"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

// Forward declaration of the CUDA driver wrapper for compatibility
class CUDADriverWrapper;

// VulkanRuntime: Primary entry point for CUDA-to-Vulkan translation
// Mirrors the CUDADriverWrapper pattern but translates to Vulkan calls
class VulkanRuntime
{
public:
    static std::shared_ptr<VulkanRuntime> getInstance();

    ~VulkanRuntime();
    VulkanRuntime(VulkanRuntime const&) = delete;
    VulkanRuntime& operator=(VulkanRuntime const&) = delete;
    VulkanRuntime(VulkanRuntime&&) = delete;
    VulkanRuntime& operator=(VulkanRuntime&&) = delete;

    // ==================== Device Management ====================
    // Mirrors: cuInit, cuDeviceGet, cuCtxCreate
    VulkanResult initialize(uint32_t gpuID = 0);
    void shutdown();

    bool isInitialized() const { return mInitialized; }
    uint32_t getGpuID() const { return mGpuID; }
    std::shared_ptr<VulkanContext> getContext() const { return mContext; }

    // ==================== Memory Management ====================
    // Mirrors: cuMemAlloc, cuMemFree, cuMemCpyHtoD
    VulkanResult allocateDeviceMemory(size_t size, VkDeviceMemory* pMemory, VkDeviceSize* offset = nullptr);
    VulkanResult freeDeviceMemory(VkDeviceMemory memory);
    VulkanResult hostToDevice(void* dstDevice, void const* srcHost, size_t byteCount);
    VulkanResult copyMemory(void* dst, void const* src, size_t byteCount);

    // ==================== Module/Shader Management ====================
    // Mirrors: cuModuleLoadData, cuModuleLoad, cuModuleUnload, cuModuleGetFunction
    VulkanResult loadShaderModule(void const* shaderData, size_t dataSize, VkShaderModule* pShaderModule);
    VulkanResult getEntryPoint(VkShaderModule module, char const* name, VkPipelineShaderStageCreateInfo* pStageInfo);
    void unloadShaderModule(VkShaderModule module);

    // ==================== Kernel/Pipeline Management ====================
    // Mirrors: cuLaunchKernel, cuFuncSetAttribute
    VulkanResult createComputePipeline(VkShaderModule shaderModule, char const* entryPoint,
                                       VkPipelineLayout layout, VkPipeline* pPipeline);
    VulkanResult destroyComputePipeline(VkPipeline pipeline);

    // ==================== Kernel Launch ====================
    // Mirrors: cuLaunchKernel
    VulkanResult launchKernel(VkPipeline pipeline, VkCommandBuffer cmdBuf,
                              uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                              uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                              std::vector<VkDescriptorSet> const& descriptors = {},
                              std::vector<uint32_t> const& constants = {});

    // ==================== Stream/Synchronization ====================
    // Mirrors: cuStreamCreate, cuStreamDestroy, cuStreamSynchronize
    VulkanResult createStream(uint32_t flags, VkCommandPool* pPool);
    VulkanResult destroyStream(VkCommandPool pool);
    VulkanResult streamSynchronize(VkCommandPool pool = VK_NULL_HANDLE);
    VulkanResult deviceSynchronize();

    // ==================== Event/Fence Operations ====================
    VkResult allocateFence(std::shared_ptr<GPUFence>* ppFence);
    void returnFence(std::shared_ptr<GPUFence>& fence);

    // ==================== Query/Utilization ====================
    std::shared_ptr<GpuUtilizationTracker> getUtilizationTracker();

    // ==================== Helper Functions ====================
    // Translate CUDA error codes to Vulkan error codes
    static VulkanResult translateVkResult(VkResult vkResult);

    // Get the underlying Vulkan driver for direct calls
    VkDevice getDevice() const { return mContext ? mContext->getDevice() : VK_NULL_HANDLE; }

    // ==================== Internal Access ====================
    VulkanRuntime();

    bool mInitialized = false;
    uint32_t mGpuID = 0;
    std::shared_ptr<VulkanContext> mContext;
    std::shared_ptr<GPUFencePool> mFencePool;
    std::unique_ptr<GpuUtilizationTracker> mUtilizationTracker;

    // Cached Vulkan properties for common calculations
    uint32_t mMinStorageBufferAlignment = 256;
    uint32_t mMinUniformBufferAlignment = 256;
};

} // namespace common
TRTLLM_NAMESPACE_END

#endif // VULKAN_RUNTIME_H
