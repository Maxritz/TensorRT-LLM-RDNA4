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

#ifndef VULKAN_KERNEL_REGISTRY_H
#define VULKAN_KERNEL_REGISTRY_H

#include "tensorrt_llm/common/vulkanContext.h"
#include "tensorrt_llm/common/vulkanRuntime.h"
#include "tensorrt_llm/common/vulkanShaderCompiler.h"
#include "tensorrt_llm/common/vulkanMemoryAllocator.h"

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
}

namespace kernels
{

// KernelDescriptor: Metadata for a Vulkan kernel
struct KernelDescriptor
{
    std::string name;                         // Kernel name (e.g., "rms_norm")
    std::string shaderPath;                   // Path to .comp file
    std::string entryPoint;                   // Entry point name
    uint32_t blockM = 256;                   // Block size M dimension
    uint32_t blockN = 256;                   // Block size N dimension
    uint32_t blockK = 256;                   // Block size K dimension
    std::vector<uint32_t> requiredFeatures; // Feature flags required
    bool requiresCooperativeMatrix = false;
    bool requiresFP16 = false;
    bool requiresBF16 = false;
    bool isQuantized = false;
    uint32_t quantType = 0; // 0=fp16, 1=q8_0, 2=q4_k, etc.
    uint32_t bindingCount = 3; // Number of descriptor buffer bindings the shader expects
};

// KernelVariant: A compiled shader variant for a specific GPU target
struct KernelVariant
{
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    common::GPUTarget target;
    common::ShaderFeatureFlags features;
    uint32_t subgroupSize = 32;
    uint32_t localSizeX = 0;
    uint32_t localSizeY = 0;
    uint32_t localSizeZ = 0;
    std::string entryPoint = "";
    uint32_t specializationConstants[13] = {0}; // Matches GLSL SPEC_* constants
    bool isValid() const { return module != VK_NULL_HANDLE && pipeline != VK_NULL_HANDLE; }
    void destroy(VkDevice device);
};

// VulkanKernelRegistry: Central registry of Vulkan kernels
// Mirrors CUDA kernel dispatch table
class VulkanKernelRegistry
{
public:
    static std::shared_ptr<VulkanKernelRegistry> getInstance();

    ~VulkanKernelRegistry();

    // Initialize with a Vulkan runtime context
    bool initialize();

    // Register a kernel descriptor
    void registerKernel(KernelDescriptor const& desc);

    // Get compiled kernel variant for a specific target and features
    std::shared_ptr<KernelVariant> getOrCreate(
        std::string const& kernelName,
        common::GPUTarget const& target,
        common::ShaderFeatureFlags const& features);

    // Get all available kernels
    std::vector<std::string> getKernelNames() const;

    // Check if a kernel is available for the current GPU
    bool isKernelAvailable(std::string const& kernelName) const;

    // Get optimal kernel variant for current GPU
    std::shared_ptr<KernelVariant> getBestVariant(
        std::string const& kernelName,
        common::VulkanMemoryManager* memMgr = nullptr);

    // Compile all registered kernels for the current target
    bool compileAll();

    // Get the list of kernel descriptors
    std::vector<KernelDescriptor> getDescriptors() const;

    // Internal: Constructor accessible to getInstance()
    VulkanKernelRegistry();

private:
    bool compileKernel(KernelDescriptor const& desc, common::GPUTarget const& target,
                       common::ShaderFeatureFlags const& features,
                       std::shared_ptr<KernelVariant>& outVariant);
    std::string generateSpecializationConstants(KernelDescriptor const& desc, common::GPUTarget const& target);

    std::shared_ptr<common::VulkanRuntime> mRuntime;
    std::shared_ptr<common::VulkanContext> mContext;
    std::shared_ptr<common::VulkanShaderCompiler> mCompiler;
    std::vector<KernelDescriptor> mDescriptors;
    std::unordered_map<std::string, std::vector<std::shared_ptr<KernelVariant>>> mCompiledKernels;
    bool mInitialized = false;
};

// Kernel dispatcher that mirrors CUDA kernel launches
class VulkanKernelDispatcher
{
public:
    explicit VulkanKernelDispatcher(std::shared_ptr<common::VulkanContext> const& ctx);
    ~VulkanKernelDispatcher();

    // ==================== Elementwise Operations ====================
    // Mirrors CUDA elementwise_add, elementwise_mul, etc.
    common::VulkanResult dispatchElementwiseAdd(
        void* a, void* b, void* output,
        size_t elementCount,
        uint32_t blockSize = 256);

    // ==================== Normalization Operations ====================
    // Mirrors CUDA rmsnorm kernels
    common::VulkanResult dispatchRmsNorm(
        void* input, void* gamma, void* beta, void* output,
        float eps, size_t hiddenDim, size_t tokenCount,
        uint32_t blockSize = 256);

    // ==================== GEMM Operations ====================
    // Mirrors CUDA fpA_intB_gemm, groupGemm, etc.
    common::VulkanResult dispatchFp16Gemm(
        void* a, void* b, void* output,
        uint32_t M, uint32_t N, uint32_t K,
        bool aTransposed = false, bool bTransposed = false,
        uint32_t blockSize = 128);

    common::VulkanResult dispatchQ8_0Gemm(
        void* weight, void* activation, void* output,
        uint32_t M, uint32_t N, uint32_t K,
        uint32_t blocksPerRow = 0,
        uint32_t blockSize = 128);

    // ==================== Attention Operations ====================
    // Mirrors CUDA attention kernels
    common::VulkanResult dispatchSoftmax(
        void* input, void* output,
        uint32_t batchSize, uint32_t numHeads, uint32_t seqLen,
        uint32_t blockSize = 256);

    common::VulkanResult dispatchAttention(
        void* q, void* k, void* v, void* output,
        uint32_t batchSize, uint32_t numHeads,
        uint32_t seqLenQ, uint32_t seqLenK, uint32_t headDim,
        bool causal,
        uint32_t blockSize = 256);

    // ==================== Utility Operations ====================
    common::VulkanResult dispatchFill(
        void* output, float value, size_t elementCount,
        uint32_t blockSize = 256);

private:
    std::shared_ptr<common::VulkanContext> mContext;
    std::shared_ptr<common::VulkanShaderCompiler> mCompiler;
    std::shared_ptr<VulkanKernelRegistry> mKernelRegistry;

    // Command buffer management
    VkCommandBuffer acquireCommandBuffer();
    void submitAndFree(VkCommandBuffer cmdBuf);

    // Descriptor set helpers
    bool allocateDescriptorSet(VkDescriptorSetLayout setLayout, VkDescriptorSet* pSet);
    void freeDescriptorSet(VkDescriptorSet set);

    VkCommandPool mCmdPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> mCmdBuffers;
    VkDescriptorPool mDescriptorPool = VK_NULL_HANDLE;
};

} // namespace kernels
TRTLLM_NAMESPACE_END

#endif // VULKAN_KERNEL_REGISTRY_H
