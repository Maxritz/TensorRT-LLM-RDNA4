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

#include "tensorrt_llm/common/vulkanMemoryAllocator.h"
#include "tensorrt_llm/common/vulkanRuntime.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstring>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

// ==================== VulkanMemoryPool Implementation ====================

VulkanMemoryPool::VulkanMemoryPool(
    std::shared_ptr<VulkanContext> const& ctx,
    VkDeviceSize blockSize,
    uint32_t memoryTypeIndex)
    : mContext(ctx)
    , mMemoryTypeIndex(memoryTypeIndex)
    , mBlockSize(blockSize)
{
}

VulkanMemoryPool::~VulkanMemoryPool()
{
    for (auto& block : mBlocks)
    {
        if (block->memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(mContext->getDevice(), block->memory, nullptr);
        }
    }
}

VulkanResult VulkanMemoryPool::allocate(VkDeviceSize size, VkDeviceSize alignment, VulkanMemoryAllocation* pAllocation)
{
    if (!pAllocation || size == 0)
    {
        return VulkanResult::INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(mMutex);

    VkDeviceSize alignedSize = (size + alignment - 1) & ~(alignment - 1);

    // Search existing blocks for a free range
    for (auto& block : mBlocks)
    {
        for (auto it = block->freeRanges.begin(); it != block->freeRanges.end(); ++it)
        {
            if (it->second >= alignedSize)
            {
                // Found a fitting range
                pAllocation->memory = block->memory;
                pAllocation->offset = it->first;
                pAllocation->size = size;
                pAllocation->alignment = alignment;
                pAllocation->memoryTypeIndex = mMemoryTypeIndex;

                // Split the free range
                if (it->second > alignedSize)
                {
                    it->first += alignedSize;
                    it->second -= alignedSize;
                }
                else
                {
                    block->freeRanges.erase(it);
                }

                block->usedSize += alignedSize;
                mUsedSize += alignedSize;

                return VulkanResult::SUCCESS;
            }
        }
    }

    // No existing block fits, create a new one
    auto newBlock = std::make_unique<PoolBlock>();
    if (allocateBlock(newBlock.get()) != VulkanResult::SUCCESS)
    {
        return VulkanResult::OUT_OF_MEMORY;
    }

    // Try again with the new block
    newBlock->freeRanges.push_back({0, newBlock->totalSize});

    mBlocks.push_back(std::move(newBlock));
    PoolBlock* block = mBlocks.back().get();

    // This allocation goes at offset 0
    pAllocation->memory = block->memory;
    pAllocation->offset = 0;
    pAllocation->size = size;
    pAllocation->alignment = alignment;
    pAllocation->memoryTypeIndex = mMemoryTypeIndex;

    if (block->freeRanges.front().second > alignedSize)
    {
        block->freeRanges.front().first = alignedSize;
        block->freeRanges.front().second -= alignedSize;
    }
    else
    {
        block->freeRanges.erase(block->freeRanges.begin());
    }

    block->usedSize += alignedSize;
    mUsedSize += alignedSize;

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanMemoryPool::deallocate(VulkanMemoryAllocation const& allocation)
{
    if (!allocation.isValid())
    {
        return VulkanResult::INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(mMutex);

    // Find the block containing this allocation
    for (auto& block : mBlocks)
    {
        if (block->memory == allocation.memory)
        {
            VkDeviceSize alignedSize = (allocation.size + allocation.alignment - 1) & ~(allocation.alignment - 1);
            block->usedSize -= alignedSize;
            mUsedSize -= alignedSize;

            // Add the deallocated range back to free list
            block->freeRanges.push_back({allocation.offset, alignedSize});

            // Merge adjacent free ranges
            std::sort(block->freeRanges.begin(), block->freeRanges.end());
            for (size_t i = 0; i + 1 < block->freeRanges.size(); ++i)
            {
                if (block->freeRanges[i].first + block->freeRanges[i].second == block->freeRanges[i + 1].first)
                {
                    block->freeRanges[i].second += block->freeRanges[i + 1].second;
                    block->freeRanges.erase(block->freeRanges.begin() + i + 1);
                    --i;
                }
            }

            break;
        }
    }

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanMemoryPool::trimTo(VkDeviceSize minSize)
{
    std::lock_guard<std::mutex> lock(mMutex);

    auto newEnd = std::remove_if(mBlocks.begin(), mBlocks.end(),
        [minSize](std::unique_ptr<PoolBlock> const& block) {
            return block->usedSize == 0 && block->totalSize < minSize;
        });

    mBlocks.erase(newEnd, mBlocks.end());

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanMemoryPool::allocateBlock(PoolBlock* pBlock)
{
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = mBlockSize;
    allocInfo.memoryTypeIndex = mMemoryTypeIndex;

    VkResult result = vkAllocateMemory(mContext->getDevice(), &allocInfo, nullptr, &pBlock->memory);
    if (result != VK_SUCCESS)
    {
        return VulkanRuntime::translateVkResult(result);
    }

    pBlock->totalSize = mBlockSize;
    pBlock->usedSize = 0;

    return VulkanResult::SUCCESS;
}

// ==================== VulkanMemoryManager Implementation ====================

std::shared_ptr<VulkanMemoryManager> VulkanMemoryManager::create(std::shared_ptr<VulkanContext> const& ctx)
{
    auto manager = std::shared_ptr<VulkanMemoryManager>(new VulkanMemoryManager(ctx));
    if (!manager->initialize())
    {
        return nullptr;
    }
    return manager;
}

VulkanMemoryManager::VulkanMemoryManager(std::shared_ptr<VulkanContext> const& ctx)
    : mContext(ctx)
{
}

VulkanMemoryManager::~VulkanMemoryManager()
{
    // All shared_ptr cleanup handles pool destruction
}

bool VulkanMemoryManager::initialize()
{
    if (!mContext)
    {
        return false;
    }

    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(mContext->getPhysicalDevice(), &memProperties);

    // Find memory types for different pools
    uint32_t deviceLocalType = findMemoryType(
        memProperties,
        0xFFFFFFFF, // All types
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    uint32_t hostVisibleType = findMemoryType(
        memProperties,
        0xFFFFFFFF,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Find memory type that is BOTH device-local and host-visible (ReBAR / unified)
    mHostVisibleDeviceLocalType = findMemoryType(
        memProperties,
        0xFFFFFFFF,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Create memory pools
    if (deviceLocalType != UINT32_MAX)
    {
        mDevicePool = std::make_unique<VulkanMemoryPool>(
            mContext,
            256 * 1024 * 1024, // 256MB blocks
            deviceLocalType);
    }

    if (hostVisibleType != UINT32_MAX)
    {
        mHostVisiblePool = std::make_unique<VulkanMemoryPool>(
            mContext,
            64 * 1024 * 1024, // 64MB blocks for host-visible
            hostVisibleType);
    }

    return mDevicePool != nullptr;
}

uint32_t VulkanMemoryManager::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(mContext->getPhysicalDevice(), &memProperties);
    return findMemoryType(memProperties, typeFilter, properties);
}

uint32_t VulkanMemoryManager::findMemoryType(
    VkPhysicalDeviceMemoryProperties const& memProperties,
    uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
    {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    return UINT32_MAX;
}

VulkanResult VulkanMemoryManager::malloc(size_t byteCount, void** pPtr, bool deviceLocal)
{
    if (!mContext || !pPtr || byteCount == 0)
    {
        return VulkanResult::INVALID_VALUE;
    }

    VulkanMemoryAllocation allocation;
    VkDeviceSize alignment = 256; // Standard alignment

    VulkanResult allocResult = VulkanResult::OUT_OF_MEMORY;
    bool usedPool = false;

    if (deviceLocal && mDevicePool)
    {
        allocResult = mDevicePool->allocate(byteCount, alignment, &allocation);
        usedPool = (allocResult == VulkanResult::SUCCESS);
    }
    else if (mHostVisiblePool)
    {
        allocResult = mHostVisiblePool->allocate(byteCount, alignment, &allocation);
        usedPool = (allocResult == VulkanResult::SUCCESS);
    }

    // If pool allocation failed, fall back to direct allocation
    if (allocResult != VulkanResult::SUCCESS)
    {
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = byteCount;

        uint32_t memType;
        bool useHostVisibleDeviceLocal = false;

        if (deviceLocal && mHostVisibleDeviceLocalType != UINT32_MAX)
        {
            // Prefer DEVICE_LOCAL + HOST_VISIBLE (ReBAR) — enables direct
            // memcpy without staging buffer + GPU copy, avoiding timeouts on
            // large transfers (e.g. 544 MB lm_head weights on AMD RX 9070 XT).
            memType = mHostVisibleDeviceLocalType;
            useHostVisibleDeviceLocal = true;
        }
        else
        {
            memType = deviceLocal ? findMemoryType(0xFFFFFFFF, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                                  : findMemoryType(0xFFFFFFFF, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }

        if (memType == UINT32_MAX)
        {
            return VulkanResult::OUT_OF_MEMORY;
        }

        allocInfo.memoryTypeIndex = memType;

        VkResult vkRes = vkAllocateMemory(mContext->getDevice(), &allocInfo, nullptr, &allocation.memory);
        if (vkRes != VK_SUCCESS)
        {
            return VulkanRuntime::translateVkResult(vkRes);
        }

        allocation.offset = 0;
        allocation.size = byteCount;
        allocation.memoryTypeIndex = memType;
        allocation.alignment = alignment;

        // Persistently map host-visible device-local memory so memcpyHtoD
        // can bypass the staging buffer entirely.
        if (useHostVisibleDeviceLocal)
        {
            vkRes = vkMapMemory(mContext->getDevice(), allocation.memory, 0, byteCount, 0, &allocation.mappedPtr);
            if (vkRes != VK_SUCCESS)
            {
                vkFreeMemory(mContext->getDevice(), allocation.memory, nullptr);
                return VulkanRuntime::translateVkResult(vkRes);
            }
            allocation.isPersistentMapped = true;
        }
    }

    // Create a buffer object for this memory
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = byteCount;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult vkRes = vkCreateBuffer(mContext->getDevice(), &bufferInfo, nullptr, &allocation.buffer);
    if (vkRes != VK_SUCCESS)
    {
        if (!usedPool)
        {
            vkFreeMemory(mContext->getDevice(), allocation.memory, nullptr);
        }
        else
        {
            mDevicePool->deallocate(allocation);
        }
        return VulkanRuntime::translateVkResult(vkRes);
    }

    vkBindBufferMemory(mContext->getDevice(), allocation.buffer, allocation.memory, allocation.offset);

    // Track allocation
    uintptr_t ptr = reinterpret_cast<uintptr_t>(allocation.buffer);
    PtrInfo info{};
    info.allocation = allocation;
    info.isPoolAllocation = usedPool;
    info.poolPtr = nullptr;

    mAllocations[ptr] = info;

    *pPtr = allocation.buffer;

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanMemoryManager::free(void* ptr)
{
    if (!ptr)
    {
        return VulkanResult::INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(mMutex);

    uintptr_t handle = reinterpret_cast<uintptr_t>(ptr);

    // Check if it's a buffer or memory handle
    auto it = mAllocations.find(handle);
    if (it == mAllocations.end())
    {
        // Try treating it as a raw memory handle
        return VulkanResult::INVALID_VALUE;
    }

    PtrInfo const& info = it->second;

    if (info.allocation.buffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(mContext->getDevice(), info.allocation.buffer, nullptr);
    }

    if (info.isPoolAllocation)
    {
        if (mDevicePool)
        {
            mDevicePool->deallocate(info.allocation);
        }
    }
    else
    {
        if (info.allocation.isPersistentMapped && info.allocation.mappedPtr != nullptr)
        {
            vkUnmapMemory(mContext->getDevice(), info.allocation.memory);
        }
        vkFreeMemory(mContext->getDevice(), info.allocation.memory, nullptr);
    }

    mAllocations.erase(it);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanMemoryManager::mallocHost(size_t byteCount, void** pPtr)
{
    if (!mContext || !pPtr || byteCount == 0)
    {
        return VulkanResult::INVALID_VALUE;
    }

    uint32_t memType = findMemoryType(0xFFFFFFFF,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (memType == UINT32_MAX)
    {
        return VulkanResult::OUT_OF_MEMORY;
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = byteCount;
    allocInfo.memoryTypeIndex = memType;

    VkDeviceMemory memory;
    VkResult result = vkAllocateMemory(mContext->getDevice(), &allocInfo, nullptr, &memory);
    if (result != VK_SUCCESS)
    {
        return VulkanRuntime::translateVkResult(result);
    }

    void* mappedPtr = nullptr;
    result = vkMapMemory(mContext->getDevice(), memory, 0, byteCount, 0, &mappedPtr);
    if (result != VK_SUCCESS)
    {
        vkFreeMemory(mContext->getDevice(), memory, nullptr);
        return VulkanRuntime::translateVkResult(result);
    }

    // Track this allocation
    PtrInfo info{};
    info.allocation.memory = memory;
    info.allocation.size = byteCount;
    info.allocation.memoryTypeIndex = memType;
    info.allocation.mappedPtr = mappedPtr;
    info.allocation.isPersistentMapped = true;
    info.isPoolAllocation = false;

    uintptr_t handle = reinterpret_cast<uintptr_t>(mappedPtr);
    mAllocations[handle] = info;

    *pPtr = mappedPtr;

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanMemoryManager::hostRegister(void* ptr, size_t byteCount, uint32_t flags)
{
    // In Vulkan, host registration is implicit when using host-visible memory
    // This is essentially a no-op since we already map memory directly
    return VulkanResult::SUCCESS;
}

VulkanResult VulkanMemoryManager::getMemoryInfo(size_t* pFree, size_t* pTotal)
{
    if (!mContext)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(mContext->getPhysicalDevice(), &memProperties);

    if (pTotal)
    {
        VkDeviceSize total = 0;
        for (uint32_t i = 0; i < memProperties.memoryHeapCount; ++i)
        {
            if (memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            {
                total += memProperties.memoryHeaps[i].size;
            }
        }
        *pTotal = static_cast<size_t>(total);
    }

    if (pFree)
    {
        VkDeviceSize used = 0;
        if (mDevicePool)
        {
            used = mDevicePool->getUsedSize();
        }
        VkDeviceSize total = 0;
        for (uint32_t i = 0; i < memProperties.memoryHeapCount; ++i)
        {
            if (memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            {
                total += memProperties.memoryHeaps[i].size;
            }
        }
        *pFree = static_cast<size_t>(total - used);
    }

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanMemoryManager::memcpyHtoD(void* dstDevice, void const* srcHost, size_t byteCount)
{
    if (!dstDevice || !srcHost || byteCount == 0)
    {
        return VulkanResult::INVALID_VALUE;
    }

    // Find the allocation for this device pointer (a VkBuffer handle).
    uintptr_t handle = reinterpret_cast<uintptr_t>(dstDevice);
    auto it = mAllocations.find(handle);
    if (it == mAllocations.end())
    {
        return VulkanResult::INVALID_VALUE;
    }

    VkDevice device = mContext->getDevice();
    VulkanMemoryAllocation const& alloc = it->second.allocation;

    // Host-visible mapped memory: direct copy.
    if (alloc.isPersistentMapped && alloc.mappedPtr != nullptr)
    {
        std::memcpy(static_cast<char*>(alloc.mappedPtr) + alloc.offset, srcHost, byteCount);
        return VulkanResult::SUCCESS;
    }

    // Device-local memory: stage through a host-visible buffer and copy on GPU.
    VkBuffer stagingBuf = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    void* mappedPtr = nullptr;
    VulkanResult st = createStagingBuffer(byteCount, &stagingBuf, &stagingMem, &mappedPtr);
    if (st != VulkanResult::SUCCESS)
    {
        return st;
    }

    std::memcpy(mappedPtr, srcHost, byteCount);
    vkUnmapMemory(device, stagingMem);

    VkBuffer dstBuf = alloc.buffer;
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = static_cast<VkDeviceSize>(byteCount);

    VulkanResult result = submitOneShot([&](VkCommandBuffer cmdBuf) {
        vkCmdCopyBuffer(cmdBuf, stagingBuf, dstBuf, 1, &copyRegion);
    });

    destroyStagingBuffer(stagingBuf, stagingMem);
    return result;
}

VulkanResult VulkanMemoryManager::memcpyDtoH(void* dstHost, void const* srcDevice, size_t byteCount)
{
    if (!dstHost || !srcDevice || byteCount == 0)
    {
        return VulkanResult::INVALID_VALUE;
    }

    uintptr_t handle = reinterpret_cast<uintptr_t>(srcDevice);
    auto it = mAllocations.find(handle);
    if (it == mAllocations.end())
    {
        return VulkanResult::INVALID_VALUE;
    }

    VkDevice device = mContext->getDevice();
    VulkanMemoryAllocation const& alloc = it->second.allocation;

    // Host-visible mapped memory: direct copy.
    if (alloc.isPersistentMapped && alloc.mappedPtr != nullptr)
    {
        std::memcpy(dstHost, static_cast<char*>(alloc.mappedPtr) + alloc.offset, byteCount);
        return VulkanResult::SUCCESS;
    }

    // Device-local memory: stage device -> host-visible buffer, then copy to host.
    VkBuffer stagingBuf = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    void* mappedPtr = nullptr;
    VulkanResult st = createStagingBuffer(byteCount, &stagingBuf, &stagingMem, &mappedPtr);
    if (st != VulkanResult::SUCCESS)
    {
        return st;
    }

    VkBuffer srcBuf = alloc.buffer;
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = static_cast<VkDeviceSize>(byteCount);

    VulkanResult result = submitOneShot([&](VkCommandBuffer cmdBuf) {
        vkCmdCopyBuffer(cmdBuf, srcBuf, stagingBuf, 1, &copyRegion);
    });

    if (result == VulkanResult::SUCCESS)
    {
        std::memcpy(dstHost, mappedPtr, byteCount);
    }

    vkUnmapMemory(device, stagingMem);
    destroyStagingBuffer(stagingBuf, stagingMem);
    return result;
}

VulkanResult VulkanMemoryManager::memcpyDtoD(void* dstDevice, void const* srcDevice, size_t byteCount)
{
    if (!dstDevice || !srcDevice || byteCount == 0)
    {
        return VulkanResult::INVALID_VALUE;
    }

    uintptr_t dstHandle = reinterpret_cast<uintptr_t>(dstDevice);
    uintptr_t srcHandle = reinterpret_cast<uintptr_t>(srcDevice);

    auto dit = mAllocations.find(dstHandle);
    auto sit = mAllocations.find(srcHandle);
    if (dit == mAllocations.end() || sit == mAllocations.end())
    {
        return VulkanResult::INVALID_VALUE;
    }

    // Fast path: both host-visible mapped allocations.
    VulkanMemoryAllocation const& dstAlloc = dit->second.allocation;
    VulkanMemoryAllocation const& srcAlloc = sit->second.allocation;
    if (dstAlloc.isPersistentMapped && dstAlloc.mappedPtr != nullptr &&
        srcAlloc.isPersistentMapped && srcAlloc.mappedPtr != nullptr)
    {
        std::memcpy(static_cast<char*>(dstAlloc.mappedPtr) + dstAlloc.offset,
                    static_cast<char*>(srcAlloc.mappedPtr) + srcAlloc.offset, byteCount);
        return VulkanResult::SUCCESS;
    }

    VkBuffer dstBuf = dstAlloc.buffer;
    VkBuffer srcBuf = srcAlloc.buffer;
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = static_cast<VkDeviceSize>(byteCount);

    return submitOneShot([&](VkCommandBuffer cmdBuf) {
        vkCmdCopyBuffer(cmdBuf, srcBuf, dstBuf, 1, &copyRegion);
    });
}

VulkanResult VulkanMemoryManager::memcpy(void* dst, void const* src, size_t byteCount, int kind)
{
    // Dispatch based on kind (HtoD,DtoH,DtoD etc.)
    if (kind == 1 /* cudaMemcpyHostToDevice */)
    {
        return memcpyHtoD(dst, src, byteCount);
    }
    else if (kind == 2 /* cudaMemcpyDeviceToHost */)
    {
        return memcpyDtoH(dst, src, byteCount);
    }
    else if (kind == 3 /* cudaMemcpyDeviceToDevice */)
    {
        return memcpyDtoD(dst, src, byteCount);
    }
    // Default: host-to-host
    std::memcpy(dst, src, byteCount);
    return VulkanResult::SUCCESS;
}

VulkanResult VulkanMemoryManager::memset(void* ptr, int value, size_t byteCount)
{
    if (!ptr || byteCount == 0)
    {
        return VulkanResult::INVALID_VALUE;
    }

    uintptr_t handle = reinterpret_cast<uintptr_t>(ptr);
    auto it = mAllocations.find(handle);
    if (it == mAllocations.end())
    {
        return VulkanResult::INVALID_VALUE;
    }

    VulkanMemoryAllocation const& alloc = it->second.allocation;

    // Host-visible mapped memory: direct fill.
    if (alloc.isPersistentMapped && alloc.mappedPtr != nullptr)
    {
        std::memset(static_cast<char*>(alloc.mappedPtr) + alloc.offset, value, byteCount);
        return VulkanResult::SUCCESS;
    }

    // Device-local memory: fill via a one-shot command buffer. vkCmdFillBuffer
    // fills 4-byte words whose least-significant byte is (value & 0xFF).
    uint32_t fillValue = static_cast<uint32_t>(static_cast<uint8_t>(value)) * 0x01010101u;
    VkBuffer buffer = alloc.buffer;
    return submitOneShot([&](VkCommandBuffer cmdBuf) {
        vkCmdFillBuffer(cmdBuf, buffer, 0, static_cast<VkDeviceSize>(byteCount), fillValue);
    });
}

bool VulkanMemoryManager::getBufferForPtr(void* ptr, VkBuffer* pBuffer, VkDeviceSize* pOffset)
{
    uintptr_t handle = reinterpret_cast<uintptr_t>(ptr);
    auto it = mAllocations.find(handle);
    if (it == mAllocations.end())
    {
        return false;
    }

    if (pBuffer)
    {
        *pBuffer = it->second.allocation.buffer;
    }
    if (pOffset)
    {
        *pOffset = it->second.allocation.offset;
    }

    return true;
}

VulkanMemoryAllocation const* VulkanMemoryManager::getAllocation(void* ptr)
{
    uintptr_t handle = reinterpret_cast<uintptr_t>(ptr);
    auto it = mAllocations.find(handle);
    return it != mAllocations.end() ? &it->second.allocation : nullptr;
}

VulkanResult VulkanMemoryManager::submitOneShot(std::function<void(VkCommandBuffer)>&& recorder)
{
    VkDevice device = mContext->getDevice();

    // One-shot command pool + buffer for a single submit + wait.
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = mContext->getComputeQueueFamilyIndex();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkResult result = vkCreateCommandPool(device, &poolInfo, nullptr, &pool);
    if (result != VK_SUCCESS)
    {
        return VulkanRuntime::translateVkResult(result);
    }

    VkCommandBuffer cmdBuf;
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    result = vkAllocateCommandBuffers(device, &allocInfo, &cmdBuf);
    if (result != VK_SUCCESS)
    {
        vkDestroyCommandPool(device, pool, nullptr);
        return VulkanRuntime::translateVkResult(result);
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    result = vkBeginCommandBuffer(cmdBuf, &beginInfo);
    if (result != VK_SUCCESS)
    {
        vkFreeCommandBuffers(device, pool, 1, &cmdBuf);
        vkDestroyCommandPool(device, pool, nullptr);
        return VulkanRuntime::translateVkResult(result);
    }

    recorder(cmdBuf);

    result = vkEndCommandBuffer(cmdBuf);
    if (result != VK_SUCCESS)
    {
        vkFreeCommandBuffers(device, pool, 1, &cmdBuf);
        vkDestroyCommandPool(device, pool, nullptr);
        return VulkanRuntime::translateVkResult(result);
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;

    result = vkQueueSubmit(mContext->getComputeQueue(0), 1, &submitInfo, VK_NULL_HANDLE);
    if (result == VK_SUCCESS)
    {
        vkQueueWaitIdle(mContext->getComputeQueue(0));
    }

    vkFreeCommandBuffers(device, pool, 1, &cmdBuf);
    vkDestroyCommandPool(device, pool, nullptr);

    return result == VK_SUCCESS ? VulkanResult::SUCCESS : VulkanRuntime::translateVkResult(result);
}

VulkanResult VulkanMemoryManager::createStagingBuffer(VkDeviceSize size, VkBuffer* pBuffer,
    VkDeviceMemory* pMemory, void** pMappedPtr)
{
    VkDevice device = mContext->getDevice();

    uint32_t memType = findMemoryType(0xFFFFFFFF,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memType == UINT32_MAX)
    {
        return VulkanResult::OUT_OF_MEMORY;
    }

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, pBuffer);
    if (result != VK_SUCCESS)
    {
        return VulkanRuntime::translateVkResult(result);
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = size;
    allocInfo.memoryTypeIndex = memType;

    result = vkAllocateMemory(device, &allocInfo, nullptr, pMemory);
    if (result != VK_SUCCESS)
    {
        vkDestroyBuffer(device, *pBuffer, nullptr);
        *pBuffer = VK_NULL_HANDLE;
        return VulkanRuntime::translateVkResult(result);
    }

    vkBindBufferMemory(device, *pBuffer, *pMemory, 0);

    result = vkMapMemory(device, *pMemory, 0, size, 0, pMappedPtr);
    if (result != VK_SUCCESS)
    {
        vkDestroyBuffer(device, *pBuffer, nullptr);
        vkFreeMemory(device, *pMemory, nullptr);
        *pBuffer = VK_NULL_HANDLE;
        *pMemory = VK_NULL_HANDLE;
        return VulkanRuntime::translateVkResult(result);
    }

    return VulkanResult::SUCCESS;
}

void VulkanMemoryManager::destroyStagingBuffer(VkBuffer buffer, VkDeviceMemory memory)
{
    VkDevice device = mContext->getDevice();
    if (buffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(device, buffer, nullptr);
    }
    if (memory != VK_NULL_HANDLE)
    {
        vkFreeMemory(device, memory, nullptr);
    }
}

// ==================== Missing Function Implementations ====================

VulkanResult VulkanMemoryManager::mallocPitch(size_t width, size_t height, size_t elementSize, void** pPtr, size_t* pPitch)
{
    if (!pPtr || !pPitch)
    {
        return VulkanResult::INVALID_VALUE;
    }

    // In Vulkan, buffer row pitch is simply the width padded to alignment
    VkDeviceSize alignment = mContext->getDeviceInfo().limits.minStorageBufferOffsetAlignment;
    size_t rowSize = width * elementSize;
    size_t pitch = (rowSize + alignment - 1) & ~(alignment - 1);

    size_t totalSize = pitch * height;
    void* ptr;
    VulkanResult result = malloc(totalSize, &ptr, true);
    if (result != VulkanResult::SUCCESS)
    {
        return result;
    }

    *pPtr = ptr;
    *pPitch = pitch;
    return VulkanResult::SUCCESS;
}

VulkanResult VulkanMemoryManager::mallocManaged(size_t byteCount, void** pPtr)
{
    // Vulkan doesn't have true unified memory
    // We use host-visible device-local memory as a compromise
    // This memory is accessible from both CPU and GPU (though possibly slower for GPU)
    uint32_t memType = findMemoryType(0xFFFFFFFF,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (memType == UINT32_MAX)
    {
        return VulkanResult::OUT_OF_MEMORY;
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = byteCount;
    allocInfo.memoryTypeIndex = memType;

    VkDeviceMemory memory;
    VkResult vkRes = vkAllocateMemory(mContext->getDevice(), &allocInfo, nullptr, &memory);
    if (vkRes != VK_SUCCESS)
    {
        return VulkanRuntime::translateVkResult(vkRes);
    }

    // Create buffer and bind
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = byteCount;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    vkRes = vkCreateBuffer(mContext->getDevice(), &bufferInfo, nullptr, &buffer);
    if (vkRes != VK_SUCCESS)
    {
        vkFreeMemory(mContext->getDevice(), memory, nullptr);
        return VulkanRuntime::translateVkResult(vkRes);
    }

    vkBindBufferMemory(mContext->getDevice(), buffer, memory, 0);

    // Track allocation
    PtrInfo info{};
    info.allocation.memory = memory;
    info.allocation.buffer = buffer;
    info.allocation.size = byteCount;
    info.allocation.memoryTypeIndex = memType;
    info.allocation.offset = 0;
    info.isPoolAllocation = false;

    uintptr_t handle = reinterpret_cast<uintptr_t>(buffer);
    mAllocations[handle] = info;

    *pPtr = buffer;
    return VulkanResult::SUCCESS;
}

VkResult VulkanMemoryManager::createMemoryPool(uint32_t flags, VkExtent3D extent, void** pPool)
{
    if (!pPool)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    uint32_t memType = findMemoryType(0xFFFFFFFF,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (memType == UINT32_MAX)
    {
        return VK_ERROR_OUT_OF_POOL_MEMORY;
    }

    auto pool = std::make_unique<VulkanMemoryPool>(mContext, 256 * 1024 * 1024, memType);
    *pPool = pool.release();
    return VK_SUCCESS;
}

VkResult VulkanMemoryManager::destroyMemoryPool(void* pool)
{
    if (pool)
    {
        delete static_cast<VulkanMemoryPool*>(pool);
    }
    return VK_SUCCESS;
}

VulkanResult VulkanMemoryManager::poolMalloc(void* pool, size_t byteCount, void** pPtr)
{
    if (!pool || !pPtr)
    {
        return VulkanResult::INVALID_VALUE;
    }

    auto* vulkanPool = static_cast<VulkanMemoryPool*>(pool);
    VulkanMemoryAllocation allocation;
    VkDeviceSize alignment = mContext->getDeviceInfo().limits.minStorageBufferOffsetAlignment;

    VulkanResult result = vulkanPool->allocate(byteCount, alignment, &allocation);
    if (result != VulkanResult::SUCCESS)
    {
        return result;
    }

    // Create buffer for the allocation
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = byteCount;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult vkRes = vkCreateBuffer(mContext->getDevice(), &bufferInfo, nullptr, &allocation.buffer);
    if (vkRes != VK_SUCCESS)
    {
        vulkanPool->deallocate(allocation);
        return VulkanRuntime::translateVkResult(vkRes);
    }

    vkBindBufferMemory(mContext->getDevice(), allocation.buffer, allocation.memory, allocation.offset);

    // Track allocation
    PtrInfo info{};
    info.allocation = allocation;
    info.isPoolAllocation = true;
    info.poolPtr = pool;

    uintptr_t handle = reinterpret_cast<uintptr_t>(allocation.buffer);
    mAllocations[handle] = info;

    *pPtr = allocation.buffer;
    return VulkanResult::SUCCESS;
}

VulkanResult VulkanMemoryManager::poolFree(void* pool, void* ptr)
{
    if (!pool || !ptr)
    {
        return VulkanResult::INVALID_VALUE;
    }

    uintptr_t handle = reinterpret_cast<uintptr_t>(ptr);
    auto it = mAllocations.find(handle);
    if (it == mAllocations.end())
    {
        return VulkanResult::INVALID_VALUE;
    }

    auto* vulkanPool = static_cast<VulkanMemoryPool*>(pool);

    if (it->second.allocation.buffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(mContext->getDevice(), it->second.allocation.buffer, nullptr);
    }

    vulkanPool->deallocate(it->second.allocation);
    mAllocations.erase(it);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanMemoryManager::memsetD32(void* ptr, uint32_t value, size_t count)
{
    // Fill 32-bit values - similar to memset but for 4-byte values
    return memset(ptr, static_cast<int>(value & 0xFF), count * 4);
}

} // namespace common
TRTLLM_NAMESPACE_END
