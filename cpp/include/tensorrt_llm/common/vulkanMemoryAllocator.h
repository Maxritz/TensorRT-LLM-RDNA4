/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 - you may not use this file except in compliance with the License.
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

#ifndef VULKAN_MEMORY_ALLOCATOR_H
#define VULKAN_MEMORY_ALLOCATOR_H

#include "tensorrt_llm/common/vulkanContext.h"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

// VulkanMemoryAllocation: Represents a single GPU memory allocation
// Mirrors CUDA's CUdeviceptr and size pair
struct VulkanMemoryAllocation
{
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    VkDeviceSize alignment = 0;
    uint32_t memoryTypeIndex = 0;
    void* mappedPtr = nullptr;   // Host mapped pointer if persistent mapping
    bool isPersistentMapped = false;
    bool isBuffer = true;          // Buffer vs image allocation
    VkBuffer buffer = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;

    bool isValid() const { return size > 0 && memory != VK_NULL_HANDLE; }
    void* getDevicePtr() const { return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(memory) + offset); }
};

// VulkanMemoryPool: Sub-allocator for GPU memory
// Similar to CUDA memory pools, manages allocations from larger memory blocks
class VulkanMemoryPool
{
public:
    explicit VulkanMemoryPool(std::shared_ptr<VulkanContext> const& ctx,
                              VkDeviceSize blockSize = 256 * 1024 * 1024, // Default 256MB blocks
                              uint32_t memoryTypeIndex = 0);
    ~VulkanMemoryPool();

    VulkanMemoryPool(VulkanMemoryPool const&) = delete;
    VulkanMemoryPool& operator=(VulkanMemoryPool const&) = delete;

    // Allocate memory from the pool
    // Returns offset within the pool's memory block
    VulkanResult allocate(VkDeviceSize size, VkDeviceSize alignment, VulkanMemoryAllocation* pAllocation);

    // Free memory back to the pool
    VulkanResult deallocate(VulkanMemoryAllocation const& allocation);

    // Trim the pool, releasing unused memory blocks
    VulkanResult trimTo(VkDeviceSize minSize = 0);

    // Get pool statistics
    VkDeviceSize getUsedSize() const { return mUsedSize; }
    VkDeviceSize getTotalSize() const { return mBlockSize * mBlocks.size(); }
    float getUtilization() const { return getTotalSize() > 0 ? static_cast<float>(mUsedSize) / static_cast<float>(getTotalSize()) : 0.0f; }

private:
    struct PoolBlock
    {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize totalSize = 0;
        VkDeviceSize usedSize = 0;
        std::vector<std::pair<VkDeviceSize, VkDeviceSize>> freeRanges; // {offset, size}
    };

    std::shared_ptr<VulkanContext> mContext;
    uint32_t mMemoryTypeIndex;
    VkDeviceSize mBlockSize;
    VkDeviceSize mUsedSize = 0;
    std::vector<std::unique_ptr<PoolBlock>> mBlocks;
    std::mutex mMutex;

    VulkanResult allocateBlock(PoolBlock* pBlock);
};

// VulkanMemoryManager: High-level memory manager
// Maps CUDA memory management patterns (malloc/free/pool) to Vulkan memory
class VulkanMemoryManager
{
public:
    static std::shared_ptr<VulkanMemoryManager> create(std::shared_ptr<VulkanContext> const& ctx);

    ~VulkanMemoryManager();

    VulkanMemoryManager(VulkanMemoryManager const&) = delete;
    VulkanMemoryManager& operator=(VulkanMemoryManager const&) = delete;

    // ==================== Memory Allocation ====================
    // Mirrors: cudaMalloc, cudaMallocAsync
    VulkanResult malloc(size_t byteCount, void** pPtr, bool deviceLocal = true);

    // Mirrors: cudaMallocManaged, cudaMallocHost
    VulkanResult mallocManaged(size_t byteCount, void** pPtr);
    VulkanResult mallocHost(size_t byteCount, void** pPtr);

    // Mirrors: cudaFree, cudaFreeAsync
    VulkanResult free(void* ptr);

    // Mirrors: cudaMallocPitch, cudaMalloc3D
    VulkanResult mallocPitch(size_t width, size_t height, size_t elementSize, void** pPtr, size_t* pPitch);

    // ==================== Memory Pool Management ====================
    // Mirrors: cudaMemPoolCreate, cudaMemPoolMalloc, cudaMemPoolFree
    VkResult createMemoryPool(uint32_t flags, VkExtent3D extent, void** pPool);
    VkResult destroyMemoryPool(void* pool);
    VulkanResult poolMalloc(void* pool, size_t byteCount, void** pPtr);
    VulkanResult poolFree(void* pool, void* ptr);

    // ==================== Data Transfer ====================
    // Mirrors: cudaMemcpy, cudaMemcpyAsync, cudaMemcpyHtoD, cudaMemcpyDtoH
    VulkanResult memcpy(void* dst, void const* src, size_t byteCount, int kind);
    VulkanResult memcpyHtoD(void* dstDevice, void const* srcHost, size_t byteCount);
    VulkanResult memcpyDtoH(void* dstHost, void const* srcDevice, size_t byteCount);
    VulkanResult memcpyDtoD(void* dstDevice, void const* srcDevice, size_t byteCount);

    // ==================== Memory Operations ====================
    // Mirrors: cudaMemset, cudaMemsetD32
    VulkanResult memset(void* ptr, int value, size_t byteCount);
    VulkanResult memsetD32(void* ptr, uint32_t value, size_t count);

    // ==================== Memory Registration ====================
    // Mirrors: cudaIpcGetMemHandle, cudaIpcOpenMemHandle
    // ==================== Memory Mapping ====================
    // Mirrors: cudaHostAlloc, cudaHostRegister
    VulkanResult hostAlloc(size_t byteCount, size_t alignment, void** pPtr);
    VulkanResult hostRegister(void* ptr, size_t byteCount, uint32_t flags);

    // ==================== Querying ====================
    // Mirrors: cuMemGetInfo
    VulkanResult getMemoryInfo(size_t* pFree, size_t* pTotal);

    // ==================== Buffer Management ====================
    // Get buffer and offset for a given device pointer
    bool getBufferForPtr(void* ptr, VkBuffer* pBuffer, VkDeviceSize* pOffset);

    // Get the original allocation for a sub-allocated pointer
    VulkanMemoryAllocation const* getAllocation(void* ptr);

private:
    explicit VulkanMemoryManager(std::shared_ptr<VulkanContext> const& ctx);
    bool initialize();

    std::shared_ptr<VulkanContext> mContext;
    std::unique_ptr<VulkanMemoryPool> mDevicePool;
    std::unique_ptr<VulkanMemoryPool> mHostVisiblePool;

    // Track all allocations by pointer for easy lookup
    struct PtrInfo
    {
        VulkanMemoryAllocation allocation;
        bool isPoolAllocation = false;
        void* poolPtr = nullptr; // Original pointer if sub-allocated
    };

    std::unordered_map<uintptr_t, PtrInfo> mAllocations;
    std::mutex mMutex;

    // Helper to find memory type index
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    // Find memory type for specific requirements
    uint32_t findMemoryType(VkPhysicalDeviceMemoryProperties const& memProperties,
                           uint32_t typeFilter, VkMemoryPropertyFlags properties);

    // Run a recorder on a one-shot command buffer, submit it to the compute queue,
    // and wait for completion. Used for staging transfers (HtoD/DtoH/fill).
    VulkanResult submitOneShot(std::function<void(VkCommandBuffer)>&& recorder);

    // Create/destroy a short-lived host-visible staging buffer used to shuttle
    // data to/from device-local (non-mappable) buffers.
    VulkanResult createStagingBuffer(VkDeviceSize size, VkBuffer* pBuffer,
        VkDeviceMemory* pMemory, void** pMappedPtr);
    void destroyStagingBuffer(VkBuffer buffer, VkDeviceMemory memory);
};

} // namespace common
TRTLLM_NAMESPACE_END

#endif // VULKAN_MEMORY_ALLOCATOR_H
