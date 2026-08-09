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

#include "tensorrt_llm/kernels/vulkanKernelRegistry.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "tensorrt_llm/common/vulkanRuntime.h"
#include "tensorrt_llm/common/vulkanContext.h"
#include "tensorrt_llm/common/vulkanShaderCompiler.h"
#include "tensorrt_llm/common/vulkanMemoryAllocator.h"
#include "tensorrt_llm/common/vulkanFence.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <filesystem>

TRTLLM_NAMESPACE_BEGIN
namespace common
{
} // namespace common

namespace kernels
{

using namespace common;

// ==================== VulkanKernelRegistry Implementation ====================

std::shared_ptr<VulkanKernelRegistry> VulkanKernelRegistry::getInstance()
{
    static std::shared_ptr<VulkanKernelRegistry> instance = std::make_shared<VulkanKernelRegistry>();
    return instance;
}

VulkanKernelRegistry::VulkanKernelRegistry()
{
}

VulkanKernelRegistry::~VulkanKernelRegistry()
{
}

bool VulkanKernelRegistry::initialize()
{
    mRuntime = VulkanRuntime::getInstance();
    if (!mRuntime)
    {
        TLLM_LOG_ERROR("Failed to get Vulkan runtime instance");
        return false;
    }

    if (!mRuntime->isInitialized())
    {
        if (mRuntime->initialize() != VulkanResult::SUCCESS)
        {
            TLLM_LOG_ERROR("Failed to initialize Vulkan runtime");
            return false;
        }
    }

    mContext = mRuntime->getContext();
    if (!mContext)
    {
        TLLM_LOG_ERROR("Failed to get Vulkan context");
        return false;
    }

    mCompiler = VulkanShaderCompiler::create(mContext);
    if (!mCompiler)
    {
        TLLM_LOG_ERROR("Failed to create Vulkan shader compiler");
        return false;
    }

    mInitialized = true;

    // Register kernel descriptors
    {
        KernelDescriptor desc;
        desc.name = "elementwise_add";
        desc.shaderPath = "elementwise_add.comp";
        desc.entryPoint = "main";
        desc.blockM = 256;
        desc.blockN = 256;
        desc.blockK = 256;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = true;
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "rms_norm";
        desc.shaderPath = "rms_norm.comp";
        desc.entryPoint = "main";
        desc.blockM = 256;
        desc.blockN = 256;
        desc.blockK = 256;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = true;
        desc.bindingCount = 4; // input, gamma, beta, output
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "q8_0_gemm";
        desc.shaderPath = "q8_0_gemm.comp";
        desc.entryPoint = "main";
        desc.blockM = 256;
        desc.blockN = 256;
        desc.blockK = 256;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = true;
        desc.isQuantized = true;
        desc.quantType = 1; // Q8_0
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "fp16_gemm";
        desc.shaderPath = "fp16_gemm.comp";
        desc.entryPoint = "main";
        desc.blockM = 256;
        desc.blockN = 256;
        desc.blockK = 256;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = true;
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "softmax";
        desc.shaderPath = "softmax.comp";
        desc.entryPoint = "main";
        desc.blockM = 1;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.bindingCount = 2; // input, output
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "attention";
        desc.shaderPath = "attention.comp";
        desc.entryPoint = "main";
        desc.blockM = 1;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.bindingCount = 4; // q, k, v, output
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "topk";
        desc.shaderPath = "topk.comp";
        desc.entryPoint = "main";
        desc.blockM = 1;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false; // operates on f32 scores, f32/f16-agnostic
        desc.bindingCount = 4;     // scores, input_offsets, output_offsets, topk_indices
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "mla_fmha";
        desc.shaderPath = "mla_fmha.comp";
        desc.entryPoint = "main";
        desc.blockM = 1;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false; // fp32 compute; fp16 I/O is a storage detail
        desc.bindingCount = 5;     // q, kv, pageTable, cacheSeqs, out
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "mla_fmha_prefill";
        desc.shaderPath = "mla_fmha_prefill.comp";
        desc.entryPoint = "main";
        desc.blockM = 1;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.bindingCount = 5;     // q, kv, pageTable, cacheSeqs, out
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "spec_accept";
        desc.shaderPath = "spec_accept.comp";
        desc.entryPoint = "main";
        desc.blockM = 1;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.requiresBF16 = false;
        desc.bindingCount = 7;
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "tree_spec_build";
        desc.shaderPath = "tree_spec_build.comp";
        desc.entryPoint = "main";
        desc.blockM = 1;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.requiresBF16 = false;
        desc.bindingCount = 7;
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "tree_spec_greedy_verify";
        desc.shaderPath = "tree_spec_greedy_verify.comp";
        desc.entryPoint = "main";
        desc.blockM = 1;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.requiresBF16 = false;
        desc.bindingCount = 7;
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "tree_spec_rejection";
        desc.shaderPath = "tree_spec_rejection.comp";
        desc.entryPoint = "main";
        desc.blockM = 1;
        desc.blockN = 256;   // local_size_x = 256
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.requiresBF16 = false;
        desc.bindingCount = 9;
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "kv_cache_update_2d";
        desc.shaderPath = "kv_cache_update_2d.comp";
        desc.entryPoint = "main";
        desc.blockM = 1;
        desc.blockN = 1;
        desc.blockK = 128;   // local_size_x = 128
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.requiresBF16 = false;
        desc.bindingCount = 7;
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "silu";
        desc.shaderPath = "silu.comp";
        desc.entryPoint = "main";
        desc.blockM = 256;   // local_size_x = 256
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.bindingCount = 2; // input, output
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "gelu";
        desc.shaderPath = "gelu.comp";
        desc.entryPoint = "main";
        desc.blockM = 256;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.bindingCount = 2; // input, output
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "swiglu";
        desc.shaderPath = "swiglu.comp";
        desc.entryPoint = "main";
        desc.blockM = 256;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.bindingCount = 2; // input, output
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "sigmoid";
        desc.shaderPath = "sigmoid.comp";
        desc.entryPoint = "main";
        desc.blockM = 256;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.bindingCount = 2; // input, output
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "relu";
        desc.shaderPath = "relu.comp";
        desc.entryPoint = "main";
        desc.blockM = 256;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.bindingCount = 2; // input, output
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "topk_general";
        desc.shaderPath = "topk_general.comp";
        desc.entryPoint = "main";
        desc.blockM = 64;    // local_size_x = 64
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.bindingCount = 3; // input, output_indices, output_values
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "sigmoid_mul";
        desc.shaderPath = "sigmoid_mul.comp";
        desc.entryPoint = "main";
        desc.blockM = 256;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.bindingCount = 3; // input_a, input_b, output
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "elementwise_mul";
        desc.shaderPath = "elementwise_mul.comp";
        desc.entryPoint = "main";
        desc.blockM = 256;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.bindingCount = 3; // input_a, input_b, output
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "scale_rows";
        desc.shaderPath = "scale_rows.comp";
        desc.entryPoint = "main";
        desc.blockM = 256;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.bindingCount = 3; // input, scale, output
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "cast";
        desc.shaderPath = "cast.comp";
        desc.entryPoint = "main";
        desc.blockM = 256;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.bindingCount = 2; // input, output
        registerKernel(desc);
    }

    {
        KernelDescriptor desc;
        desc.name = "index_add";
        desc.shaderPath = "index_add.comp";
        desc.entryPoint = "main";
        desc.blockM = 256;
        desc.blockN = 1;
        desc.blockK = 1;
        desc.requiresCooperativeMatrix = false;
        desc.requiresFP16 = false;
        desc.bindingCount = 3; // output, indices, values
        registerKernel(desc);
    }

    return true;
}

void VulkanKernelRegistry::registerKernel(KernelDescriptor const& desc)
{
    mDescriptors.push_back(desc);
}

std::shared_ptr<KernelVariant> VulkanKernelRegistry::getOrCreate(
    std::string const& kernelName,
    GPUTarget const& target,
    ShaderFeatureFlags const& features)
{
    if (!mInitialized)
    {
        return nullptr;
    }

    // Check if already compiled
    auto it = mCompiledKernels.find(kernelName);
    if (it != mCompiledKernels.end())
    {
        for (auto const& variant : it->second)
        {
            if (variant->target.arch == target.arch &&
                variant->features.enableFP16 == features.enableFP16 &&
                variant->features.enableCooperativeMatrix == features.enableCooperativeMatrix)
            {
                return variant;
            }
        }
    }

    // Find the kernel descriptor
    KernelDescriptor const* desc = nullptr;
    for (auto const& d : mDescriptors)
    {
        if (d.name == kernelName)
        {
            desc = &d;
            break;
        }
    }

    if (!desc)
    {
        TLLM_LOG_ERROR("Kernel not found: %s", kernelName.c_str());
        return nullptr;
    }

    // Compile this variant
    auto variant = std::make_shared<KernelVariant>();
    if (compileKernel(*desc, target, features, variant) != true)
    {
        return nullptr;
    }

    // Cache it
    mCompiledKernels[kernelName].push_back(variant);
    return variant;
}

bool VulkanKernelRegistry::compileKernel(
    KernelDescriptor const& desc,
    GPUTarget const& target,
    ShaderFeatureFlags const& features,
    std::shared_ptr<KernelVariant>& outVariant)
{
    if (!mCompiler || !mContext)
    {
        return false;
    }

    // Read shader source - resolve relative to kernel source file location
    // The __FILE__ macro gives us the source file path, from which we can
    // locate the shaders/ directory
    std::filesystem::path sourceDir = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path shaderDir;

    // Check common shader locations
    std::vector<std::filesystem::path> searchPaths = {
        sourceDir / "vulkan" / "shaders",
        sourceDir.parent_path() / "shaders",
        "shaders"
    };

    for (auto const& path : searchPaths)
    {
        if (std::filesystem::exists(path))
        {
            shaderDir = path;
            break;
        }
    }

    if (shaderDir.empty())
    {
        TLLM_LOG_ERROR("Failed to find shaders directory");
        return false;
    }

    std::string shaderPath = (shaderDir / desc.shaderPath).string();

    std::ifstream file(shaderPath);
    if (!file.is_open())
    {
        TLLM_LOG_ERROR("Failed to open shader: %s", shaderPath.c_str());
        return false;
    }

    std::stringstream ss;
    ss << file.rdbuf();
    std::string source = ss.str();

    // Compile to SPIR-V
    std::vector<uint32_t> spirv;
    GPUTarget compileTarget = target;
    ShaderFeatureFlags compileFeatures = features;

    // Override features based on kernel requirements
    if (desc.requiresFP16)
    {
        compileFeatures.enableFP16 = true;
    }
    if (desc.requiresCooperativeMatrix)
    {
        compileFeatures.enableCooperativeMatrix = compileTarget.hasCooperativeMatrix;
    }

    VulkanResult result = mCompiler->compile(source, desc.entryPoint, &spirv, false, &compileFeatures, &compileTarget);
    if (result != VulkanResult::SUCCESS)
    {
        TLLM_LOG_ERROR("Failed to compile shader %s: %s", desc.name.c_str(),
            VulkanContext::getErrorString(result));
        return false;
    }

    // Create shader module
    result = mCompiler->createShaderModule(spirv, &outVariant->module);
    if (result != VulkanResult::SUCCESS)
    {
        TLLM_LOG_ERROR("Failed to create shader module for %s", desc.name.c_str());
        return false;
    }

    outVariant->target = compileTarget;
    outVariant->features = compileFeatures;
    outVariant->subgroupSize = compileTarget.subgroupSize;
    outVariant->localSizeX = desc.blockM;
    outVariant->localSizeY = 1;
    outVariant->localSizeZ = 1;
    outVariant->entryPoint = desc.entryPoint;

    // Set specialization constants
    std::memset(outVariant->specializationConstants, 0, sizeof(outVariant->specializationConstants));
    outVariant->specializationConstants[0] = compileTarget.subgroupSize;  // SPEC_SUBGROUP_SIZE
    outVariant->specializationConstants[1] = 1;  // SPEC_WG_SIZE_X (computed at dispatch time)
    outVariant->specializationConstants[2] = desc.blockM;  // SPEC_D
    outVariant->specializationConstants[3] = desc.blockN;  // SPEC_FFN_DIM
    outVariant->specializationConstants[10] = desc.quantType;  // SPEC_QUANT_TYPE

    // Create a descriptor set layout with one storage-buffer binding per shader
    // buffer binding (bindings 0..bindingCount-1), all visible to the compute stage.
    uint32_t bindingCount = desc.bindingCount;
    std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount);
    for (uint32_t i = 0; i < bindingCount; ++i)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo setLayoutInfo{};
    setLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setLayoutInfo.bindingCount = bindingCount;
    setLayoutInfo.pBindings = bindings.data();

    VkResult vkRes = vkCreateDescriptorSetLayout(mContext->getDevice(), &setLayoutInfo, nullptr, &outVariant->setLayout);
    if (vkRes != VK_SUCCESS)
    {
        TLLM_LOG_ERROR("Failed to create descriptor set layout for %s: %d", desc.name.c_str(), vkRes);
        return false;
    }

    // Create pipeline layout with the descriptor set layout + push constants
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 256; // Standard push constant buffer size

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &outVariant->setLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;

    vkRes = vkCreatePipelineLayout(mContext->getDevice(), &layoutInfo, nullptr, &outVariant->pipelineLayout);
    if (vkRes != VK_SUCCESS)
    {
        TLLM_LOG_ERROR("Failed to create pipeline layout for %s: %d", desc.name.c_str(), vkRes);
        vkDestroyDescriptorSetLayout(mContext->getDevice(), outVariant->setLayout, nullptr);
        outVariant->setLayout = VK_NULL_HANDLE;
        return false;
    }

    // Create compute pipeline
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = outVariant->module;
    pipelineInfo.stage.pName = desc.entryPoint.c_str();
    pipelineInfo.layout = outVariant->pipelineLayout;

    vkRes = vkCreateComputePipelines(mContext->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &outVariant->pipeline);
    if (vkRes != VK_SUCCESS)
    {
        TLLM_LOG_ERROR("Failed to create compute pipeline for %s: %d", desc.name.c_str(), vkRes);
        vkDestroyPipelineLayout(mContext->getDevice(), outVariant->pipelineLayout, nullptr);
        outVariant->pipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(mContext->getDevice(), outVariant->setLayout, nullptr);
        outVariant->setLayout = VK_NULL_HANDLE;
        return false;
    }

    return true;
}

std::shared_ptr<KernelVariant> VulkanKernelRegistry::getBestVariant(
    std::string const& kernelName,
    VulkanMemoryManager* memMgr)
{
    if (!mInitialized)
    {
        return nullptr;
    }

    GPUTarget target = mCompiler->detectTarget();
    ShaderFeatureFlags features = mCompiler->getFeatureFlags();

    return getOrCreate(kernelName, target, features);
}

bool VulkanKernelRegistry::compileAll()
{
    if (!mInitialized)
    {
        return false;
    }

    GPUTarget target = mCompiler->detectTarget();

    for (auto const& desc : mDescriptors)
    {
        if (!isKernelAvailable(desc.name))
        {
            // Check if kernel requires unsupported features
            if (desc.requiresCooperativeMatrix && !target.hasCooperativeMatrix)
            {
                TLLM_LOG_WARNING("Skipping kernel %s: requires cooperative matrix support", desc.name.c_str());
                continue;
            }
            if (desc.requiresFP16 && !target.hasFP16)
            {
                TLLM_LOG_WARNING("Skipping kernel %s: requires FP16 support", desc.name.c_str());
                continue;
            }

            TLLM_LOG_ERROR("Failed to compile kernel: %s", desc.name.c_str());
            return false;
        }
    }

    return true;
}

std::vector<std::string> VulkanKernelRegistry::getKernelNames() const
{
    std::vector<std::string> names;
    for (auto const& desc : mDescriptors)
    {
        names.push_back(desc.name);
    }
    return names;
}

bool VulkanKernelRegistry::isKernelAvailable(std::string const& kernelName) const
{
    // Check if we can compile this kernel for the current target
    auto const& variants = mCompiledKernels.find(kernelName);
    if (variants != mCompiledKernels.end())
    {
        return !variants->second.empty();
    }

    // Try to compile on-demand
    for (auto const& desc : mDescriptors)
    {
        if (desc.name == kernelName)
        {
            // Check feature requirements
            VulkanDeviceInfo const& info = mContext->getDeviceInfo();
            if (desc.requiresCooperativeMatrix && !info.hasCooperativeMatrix)
            {
                return false;
            }
            if (desc.requiresFP16 && !info.hasFP16)
            {
                return false;
            }
            return true;
        }
    }

    return false;
}

std::vector<KernelDescriptor> VulkanKernelRegistry::getDescriptors() const
{
    return mDescriptors;
}

// ==================== VulkanKernelDispatcher Implementation ====================

VulkanKernelDispatcher::VulkanKernelDispatcher(std::shared_ptr<VulkanContext> const& ctx)
    : mContext(ctx)
    , mKernelRegistry(VulkanKernelRegistry::getInstance())
{
    if (mContext && mKernelRegistry)
    {
        // Create command pool for this dispatcher
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = mContext->getComputeQueueFamilyIndex();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

        vkCreateCommandPool(mContext->getDevice(), &poolInfo, nullptr, &mCmdPool);

        // Allocate command buffers
        mCmdBuffers.resize(4);
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = mCmdPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 4;

        vkAllocateCommandBuffers(mContext->getDevice(), &allocInfo, mCmdBuffers.data());
    }

    // Create a descriptor pool large enough for several sets of storage buffers.
    if (mContext)
    {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 64 * 4;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        // FREE_DESCRIPTOR_SET_BIT is required because dispatchElementwiseAdd frees
        // the per-dispatch descriptor set after submission.
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 64;

        vkCreateDescriptorPool(mContext->getDevice(), &poolInfo, nullptr, &mDescriptorPool);
    }
}

bool VulkanKernelDispatcher::allocateDescriptorSet(VkDescriptorSetLayout setLayout, VkDescriptorSet* pSet)
{
    if (!mContext || mDescriptorPool == VK_NULL_HANDLE || !pSet)
    {
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = mDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &setLayout;

    VkResult result = vkAllocateDescriptorSets(mContext->getDevice(), &allocInfo, pSet);
    return result == VK_SUCCESS;
}

void VulkanKernelDispatcher::freeDescriptorSet(VkDescriptorSet set)
{
    if (mContext && mDescriptorPool != VK_NULL_HANDLE && set != VK_NULL_HANDLE)
    {
        vkFreeDescriptorSets(mContext->getDevice(), mDescriptorPool, 1, &set);
    }
}

VulkanKernelDispatcher::~VulkanKernelDispatcher()
{
    if (mContext)
    {
        VkDevice device = mContext->getDevice();
        if (mDescriptorPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device, mDescriptorPool, nullptr);
        }
        if (mCmdPool != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(device, mCmdPool, nullptr);
        }
    }
}

void KernelVariant::destroy(VkDevice device)
{
    if (pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }
    if (pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }
    if (setLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
        setLayout = VK_NULL_HANDLE;
    }
    if (module != VK_NULL_HANDLE)
    {
        vkDestroyShaderModule(device, module, nullptr);
        module = VK_NULL_HANDLE;
    }
}

VkCommandBuffer VulkanKernelDispatcher::acquireCommandBuffer()
{
    if (mCmdPool == VK_NULL_HANDLE || mCmdBuffers.empty())
    {
        return VK_NULL_HANDLE;
    }

    vkResetCommandPool(mContext->getDevice(), mCmdPool, 0);

    VkCommandBuffer cmdBuf = mCmdBuffers[0];
    vkResetCommandBuffer(cmdBuf, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    return cmdBuf;
}

void VulkanKernelDispatcher::submitAndFree(VkCommandBuffer cmdBuf)
{
    if (cmdBuf == VK_NULL_HANDLE)
    {
        return;
    }

    vkEndCommandBuffer(cmdBuf);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;

    VkFence fence = VK_NULL_HANDLE;
    if (mKernelRegistry)
    {
        auto runtime = VulkanRuntime::getInstance();
        if (runtime)
        {
            std::shared_ptr<GPUFence> fencePtr;
            runtime->allocateFence(&fencePtr);
            if (fencePtr)
            {
                fence = fencePtr->getFence();
            }
        }
    }

    vkQueueSubmit(mContext->getComputeQueue(0), 1, &submitInfo, fence);
    vkQueueWaitIdle(mContext->getComputeQueue(0));
}

VulkanResult VulkanKernelDispatcher::dispatchElementwiseAdd(
    void* a, void* b, void* output,
    size_t elementCount,
    uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    auto variant = mKernelRegistry->getBestVariant("elementwise_add");
    if (!variant || !variant->isValid())
    {
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
    {
        return VulkanResult::UNKNOWN_ERROR;
    }

    // Use pre-created pipeline from the variant
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    // Bind the three storage buffers (a, b, output) to descriptor set bindings
    // 0, 1, 2 declared by the shader. The opaque device pointers returned by
    // VulkanBackend::malloc are VkBuffer handles, so they are cast directly.
    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(a), reinterpret_cast<VkBuffer>(b),
                          reinterpret_cast<VkBuffer>(output)};
    VkDeviceSize range = static_cast<VkDeviceSize>(elementCount) * sizeof(float);

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[3]{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].offset = 0;
        bufInfos[i].range = range;
    }

    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 3, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);

    // Set up push constants
    struct PushConstants
    {
        uint32_t N;
    } pc{};
    pc.N = static_cast<uint32_t>(elementCount);

    // Push constants
    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

    uint32_t workGroupsX = (static_cast<uint32_t>(elementCount) + blockSize - 1) / blockSize;
    vkCmdDispatch(cmdBuf, workGroupsX, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchRmsNorm(
    void* input, void* gamma, void* beta, void* output,
    float eps, size_t hiddenDim, size_t tokenCount,
    uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    auto variant = mKernelRegistry->getBestVariant("rms_norm");
    if (!variant || !variant->isValid())
    {
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
    {
        return VulkanResult::UNKNOWN_ERROR;
    }

    uint32_t totalElements = static_cast<uint32_t>(hiddenDim * tokenCount);

    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(input), reinterpret_cast<VkBuffer>(gamma),
                          reinterpret_cast<VkBuffer>(beta), reinterpret_cast<VkBuffer>(output)};

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[4]{};
    bufInfos[0].buffer = buffers[0];
    bufInfos[1].buffer = buffers[1];
    bufInfos[2].buffer = buffers[2];
    bufInfos[3].buffer = buffers[3];
    bufInfos[0].range = static_cast<VkDeviceSize>(totalElements) * sizeof(float);
    bufInfos[1].range = static_cast<VkDeviceSize>(hiddenDim) * sizeof(float);
    bufInfos[2].range = static_cast<VkDeviceSize>(hiddenDim) * sizeof(float);
    bufInfos[3].range = static_cast<VkDeviceSize>(totalElements) * sizeof(float);
    for (uint32_t i = 0; i < 4; ++i)
    {
        bufInfos[i].offset = 0;
    }

    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 4, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants
    {
        uint32_t hiddenDim;
        uint32_t totalElements;
        float eps;
    } pc{};
    pc.hiddenDim = static_cast<uint32_t>(hiddenDim);
    pc.totalElements = totalElements;
    pc.eps = eps;

    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    uint32_t workGroupsX = (totalElements + blockSize - 1u) / blockSize;
    vkCmdDispatch(cmdBuf, workGroupsX, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchFp16Gemm(
    void* a, void* b, void* output,
    uint32_t M, uint32_t N, uint32_t K,
    bool aTransposed, bool bTransposed,
    uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    // This Vulkan variant implements the non-transposed GEMM
    // C[M,N] = A[M,K] * B[K,N]. Transposed operand paths are not supported.
    if (aTransposed || bTransposed)
    {
        TLLM_LOG_WARNING("Vulkan fp16_gemm: transposed operands are not supported by this variant");
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    auto variant = mKernelRegistry->getBestVariant("fp16_gemm");
    if (!variant || !variant->isValid())
    {
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
    {
        return VulkanResult::UNKNOWN_ERROR;
    }

    uint32_t totalElements = M * N;

    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(a), reinterpret_cast<VkBuffer>(b),
                          reinterpret_cast<VkBuffer>(output)};

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    // binding 0: A (M*K), binding 1: B (K*N), binding 2: output (M*N)
    VkDescriptorBufferInfo bufInfos[3]{};
    bufInfos[0].buffer = buffers[0];
    bufInfos[1].buffer = buffers[1];
    bufInfos[2].buffer = buffers[2];
    bufInfos[0].range = static_cast<VkDeviceSize>(M) * K * sizeof(float);
    bufInfos[1].range = static_cast<VkDeviceSize>(K) * N * sizeof(float);
    bufInfos[2].range = static_cast<VkDeviceSize>(M) * N * sizeof(float);
    for (uint32_t i = 0; i < 3; ++i)
    {
        bufInfos[i].offset = 0;
    }

    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 3, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants
    {
        uint32_t M;
        uint32_t N;
        uint32_t K;
    } pc{};
    pc.M = M;
    pc.N = N;
    pc.K = K;

    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    uint32_t workGroupsX = (totalElements + blockSize - 1u) / blockSize;
    vkCmdDispatch(cmdBuf, workGroupsX, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchQ8_0Gemm(
    void* weight, void* activation, void* output,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t blocksPerRow,
    uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    // Q8_0 blocks pack 32 int8 weights under one fp32 scale (36 bytes/block).
    // K must be a positive multiple of 32 and blocksPerRow must equal K / 32.
    if (K == 0 || N == 0 || M == 0 || blocksPerRow == 0 || (K % 32u) != 0u)
    {
        TLLM_LOG_WARNING("Vulkan q8_0_gemm: requires K>0, K %% 32 == 0, and blocksPerRow>0");
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    auto variant = mKernelRegistry->getBestVariant("q8_0_gemm");
    if (!variant || !variant->isValid())
    {
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
    {
        return VulkanResult::UNKNOWN_ERROR;
    }

    uint32_t totalElements = M * N;

    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(weight), reinterpret_cast<VkBuffer>(activation),
                          reinterpret_cast<VkBuffer>(output)};

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDeviceSize weightBytes = static_cast<VkDeviceSize>(N) * blocksPerRow * 36u; // 36-byte q8_0 blocks
    VkDeviceSize actBytes   = static_cast<VkDeviceSize>(M) * K * sizeof(float);  // fp32 activation
    VkDeviceSize outBytes   = static_cast<VkDeviceSize>(M) * N * sizeof(float); // fp32 output
    VkDescriptorBufferInfo bufInfos[3]{};
    bufInfos[0].buffer = buffers[0]; bufInfos[0].range = weightBytes; bufInfos[0].offset = 0;
    bufInfos[1].buffer = buffers[1]; bufInfos[1].range = actBytes;    bufInfos[1].offset = 0;
    bufInfos[2].buffer = buffers[2]; bufInfos[2].range = outBytes;    bufInfos[2].offset = 0;

    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 3, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants
    {
        uint32_t M;
        uint32_t N;
        uint32_t K;
        uint32_t blocksPerRow;
    } pc{};
    pc.M = M;
    pc.N = N;
    pc.K = K;
    pc.blocksPerRow = blocksPerRow;

    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    uint32_t workGroupsX = (totalElements + blockSize - 1u) / blockSize;
    vkCmdDispatch(cmdBuf, workGroupsX, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchSoftmax(
    void* input, void* output,
    uint32_t batchSize, uint32_t numHeads, uint32_t seqLen,
    uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    if (batchSize == 0 || numHeads == 0 || seqLen == 0)
    {
        TLLM_LOG_WARNING("Vulkan softmax: zero-dimension input (batchSize/numHeads/seqLen must be > 0)");
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    auto variant = mKernelRegistry->getBestVariant("softmax");
    if (!variant || !variant->isValid())
    {
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
    {
        return VulkanResult::UNKNOWN_ERROR;
    }

    uint32_t numRows = batchSize * numHeads;
    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(input), reinterpret_cast<VkBuffer>(output)};

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDeviceSize range = static_cast<VkDeviceSize>(numRows) * seqLen * sizeof(float);
    VkDescriptorBufferInfo bufInfos[2]{};
    bufInfos[0].buffer = buffers[0];
    bufInfos[1].buffer = buffers[1];
    bufInfos[0].range = range;
    bufInfos[1].range = range;
    for (uint32_t i = 0; i < 2; ++i)
    {
        bufInfos[i].offset = 0;
    }

    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 2, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants
    {
        uint32_t batchSize;
        uint32_t numHeads;
        uint32_t seqLen;
    } pc{};
    pc.batchSize = batchSize;
    pc.numHeads = numHeads;
    pc.seqLen = seqLen;

    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    // local_size_x = 1, one workgroup per (batch * head) row.
    (void)blockSize;
    vkCmdDispatch(cmdBuf, numRows, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchAttention(
    void* q, void* k, void* v, void* output,
    uint32_t batchSize, uint32_t numHeads,
    uint32_t seqLenQ, uint32_t seqLenK, uint32_t headDim,
    bool causal,
    uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    if (batchSize == 0 || numHeads == 0 || seqLenQ == 0 || seqLenK == 0 || headDim == 0)
    {
        TLLM_LOG_WARNING("Vulkan attention: zero-dimension input");
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    auto variant = mKernelRegistry->getBestVariant("attention");
    if (!variant || !variant->isValid())
    {
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
    {
        return VulkanResult::UNKNOWN_ERROR;
    }

    uint32_t numQueries = batchSize * numHeads * seqLenQ;
    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(q), reinterpret_cast<VkBuffer>(k),
                          reinterpret_cast<VkBuffer>(v), reinterpret_cast<VkBuffer>(output)};

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDeviceSize qBytes  = static_cast<VkDeviceSize>(batchSize) * numHeads * seqLenQ * headDim * sizeof(float);
    VkDeviceSize kvBytes = static_cast<VkDeviceSize>(batchSize) * numHeads * seqLenK * headDim * sizeof(float);
    VkDeviceSize oBytes  = static_cast<VkDeviceSize>(batchSize) * numHeads * seqLenQ * headDim * sizeof(float);
    VkDescriptorBufferInfo bufInfos[4]{};
    bufInfos[0].buffer = buffers[0]; bufInfos[0].range = qBytes;  bufInfos[0].offset = 0;
    bufInfos[1].buffer = buffers[1]; bufInfos[1].range = kvBytes; bufInfos[1].offset = 0;
    bufInfos[2].buffer = buffers[2]; bufInfos[2].range = kvBytes; bufInfos[2].offset = 0;
    bufInfos[3].buffer = buffers[3]; bufInfos[3].range = oBytes;  bufInfos[3].offset = 0;

    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 4, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants
    {
        uint32_t batchSize;
        uint32_t numHeads;
        uint32_t seqLenQ;
        uint32_t seqLenK;
        uint32_t headDim;
        uint32_t causal;
    } pc{};
    pc.batchSize = batchSize;
    pc.numHeads = numHeads;
    pc.seqLenQ = seqLenQ;
    pc.seqLenK = seqLenK;
    pc.headDim = headDim;
    pc.causal = causal ? 1u : 0u;

    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    // local_size_x = 1; one workgroup owns a single (batch, head, query) row.
    (void)blockSize;
    uint32_t workGroupsX = numQueries;
    vkCmdDispatch(cmdBuf, workGroupsX, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchTopk(
    void* scores, void* inputOffsets, void* outputOffsets, void* topkIndices,
    uint32_t topk, uint32_t numHeads, uint32_t batchSize,
    uint32_t totalTokens, uint32_t totalOutputTokens,
    uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    // topk must fit the shader's fixed-size register scratch (see topk.comp).
    constexpr uint32_t MAX_TOPK = 32;
    if (topk == 0 || topk > MAX_TOPK || numHeads == 0 || batchSize == 0 ||
        totalTokens == 0 || totalOutputTokens == 0)
    {
        TLLM_LOG_WARNING(
            "Vulkan topk: invalid dims (topk=%u heads=%u batch=%u tokens=%u out=%u, topk<=%u)",
            topk, numHeads, batchSize, totalTokens, totalOutputTokens, MAX_TOPK);
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    auto variant = mKernelRegistry->getBestVariant("topk");
    if (!variant || !variant->isValid())
    {
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
    {
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(scores),
                          reinterpret_cast<VkBuffer>(inputOffsets),
                          reinterpret_cast<VkBuffer>(outputOffsets),
                          reinterpret_cast<VkBuffer>(topkIndices)};

    VkDeviceSize scoresBytes = static_cast<VkDeviceSize>(numHeads) * totalTokens * sizeof(float);
    VkDeviceSize offBytes    = static_cast<VkDeviceSize>(batchSize + 1) * sizeof(uint32_t);
    VkDeviceSize idxBytes    = static_cast<VkDeviceSize>(numHeads) * totalOutputTokens * sizeof(int32_t);

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[4]{};
    bufInfos[0].buffer = buffers[0]; bufInfos[0].offset = 0; bufInfos[0].range = scoresBytes;
    bufInfos[1].buffer = buffers[1]; bufInfos[1].offset = 0; bufInfos[1].range = offBytes;
    bufInfos[2].buffer = buffers[2]; bufInfos[2].offset = 0; bufInfos[2].range = offBytes;
    bufInfos[3].buffer = buffers[3]; bufInfos[3].offset = 0; bufInfos[3].range = idxBytes;

    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 4, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants
    {
        uint32_t topk;
        uint32_t numHeads;
        uint32_t batchSize;
        uint32_t totalTokens;
        uint32_t totalOutputTokens;
    } pc{};
    pc.topk = topk;
    pc.numHeads = numHeads;
    pc.batchSize = batchSize;
    pc.totalTokens = totalTokens;
    pc.totalOutputTokens = totalOutputTokens;

    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    // One workgroup per (batch, head) row.
    (void)blockSize;
    vkCmdDispatch(cmdBuf, batchSize, numHeads, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchTreeSpecBuild(
    void* parentListUvec2, void* selectedIndexUvec2, void* treeMask,
    void* positions, void* retrieveIndex, void* retrieveNextToken,
    void* retrieveNextSibling,
    uint32_t batchSize, uint32_t draftTokenNum, uint32_t topK,
    uint32_t depth, uint32_t numInt32PerRow, uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
        return VulkanResult::INITIALIZATION_FAILED;
    if (batchSize == 0 || draftTokenNum == 0 || topK == 0 || depth == 0 || numInt32PerRow == 0)
        return VulkanResult::FEATURE_NOT_PRESENT;

    auto variant = mKernelRegistry->getBestVariant("tree_spec_build");
    if (!variant || !variant->isValid())
        return VulkanResult::FEATURE_NOT_PRESENT;

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
        return VulkanResult::UNKNOWN_ERROR;

    // parentList Uvec2 count = B * (topK * (depth-1) + 1); selectedIndex Uvec2 count = B*(dT-1).
    // treeMask uint count = B * dT * numInt32PerRow. Remaining buffers: B * dT int32.
    uint32_t parentElems  = batchSize * (topK * (depth - 1u) + 1u);
    uint32_t selElems     = batchSize * (draftTokenNum - 1u);
    uint32_t treeElems    = batchSize * draftTokenNum * numInt32PerRow;
    uint32_t arrElems     = batchSize * draftTokenNum;
    VkDeviceSize parentBytes  = VkDeviceSize(parentElems)  * 2u * sizeof(uint32_t); // uvec2
    VkDeviceSize selBytes     = VkDeviceSize(selElems)   * 2u * sizeof(uint32_t);
    VkDeviceSize treeBytes    = VkDeviceSize(treeElems)  * sizeof(uint32_t);
    VkDeviceSize arrBytes     = VkDeviceSize(arrElems)   * sizeof(int32_t);

    VkBuffer buffers[] = {
        reinterpret_cast<VkBuffer>(parentListUvec2),   reinterpret_cast<VkBuffer>(selectedIndexUvec2),
        reinterpret_cast<VkBuffer>(treeMask),          reinterpret_cast<VkBuffer>(positions),
        reinterpret_cast<VkBuffer>(retrieveIndex),      reinterpret_cast<VkBuffer>(retrieveNextToken),
        reinterpret_cast<VkBuffer>(retrieveNextSibling)};
    VkDeviceSize ranges[] = {parentBytes, selBytes, treeBytes, arrBytes, arrBytes, arrBytes, arrBytes};

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[7]{};
    VkWriteDescriptorSet writes[7]{};
    for (uint32_t i = 0; i < 7; ++i)
    {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].range = ranges[i];
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(mContext->getDevice(), 7, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants
    {
        uint32_t draftTokenNum;
        uint32_t topK;
        uint32_t depth;
        uint32_t numInt32PerRow;
    } pc{};
    pc.draftTokenNum = draftTokenNum;
    pc.topK = topK;
    pc.depth = depth;
    pc.numInt32PerRow = numInt32PerRow;
    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    // One workgroup per (batch, token) — 2D grid.
    (void)blockSize;
    vkCmdDispatch(cmdBuf, batchSize, draftTokenNum, 1);
    submitAndFree(cmdBuf);
    freeDescriptorSet(set);
    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchTreeSpecGreedyVerify(
    void* acceptIndex, void* acceptTokenNum, void* acceptToken,
    void* candidates, void* retrievePacked, void* targetPredict,
    void* treeValid,
    uint32_t batchSize, uint32_t numSpeculativeTokens, uint32_t numDraftTokens,
    uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
        return VulkanResult::INITIALIZATION_FAILED;
    if (batchSize == 0 || numSpeculativeTokens == 0 || numDraftTokens == 0)
        return VulkanResult::FEATURE_NOT_PRESENT;

    auto variant = mKernelRegistry->getBestVariant("tree_spec_greedy_verify");
    if (!variant || !variant->isValid())
        return VulkanResult::FEATURE_NOT_PRESENT;

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
        return VulkanResult::UNKNOWN_ERROR;

    VkDeviceSize idxBytes = VkDeviceSize(batchSize) * numSpeculativeTokens * sizeof(int32_t);
    VkDeviceSize numBytes = VkDeviceSize(batchSize) * sizeof(uint32_t);
    VkDeviceSize tokBytes = VkDeviceSize(batchSize) * numSpeculativeTokens * sizeof(int32_t);
    VkDeviceSize candBytes = VkDeviceSize(batchSize) * numDraftTokens * sizeof(int32_t);
    VkDeviceSize retBytes  = VkDeviceSize(batchSize) * numDraftTokens * 3u * sizeof(int32_t);
    VkDeviceSize tgtBytes  = VkDeviceSize(batchSize) * numDraftTokens * sizeof(int32_t);
    VkDeviceSize valBytes  = VkDeviceSize(batchSize) * sizeof(int32_t);

    VkBuffer buffers[] = {
        reinterpret_cast<VkBuffer>(acceptIndex), reinterpret_cast<VkBuffer>(acceptTokenNum),
        reinterpret_cast<VkBuffer>(acceptToken),   reinterpret_cast<VkBuffer>(candidates),
        reinterpret_cast<VkBuffer>(retrievePacked), reinterpret_cast<VkBuffer>(targetPredict),
        reinterpret_cast<VkBuffer>(treeValid)};
    VkDeviceSize ranges[] = {idxBytes, numBytes, tokBytes, candBytes, retBytes, tgtBytes, valBytes};

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[7]{};
    VkWriteDescriptorSet writes[7]{};
    for (uint32_t i = 0; i < 7; ++i)
    {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].range = ranges[i];
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(mContext->getDevice(), 7, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants
    {
        uint32_t numSpeculativeTokens;
        uint32_t numDraftTokens;
    } pc{};
    pc.numSpeculativeTokens = numSpeculativeTokens;
    pc.numDraftTokens = numDraftTokens;
    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    (void)blockSize;
    vkCmdDispatch(cmdBuf, batchSize, 1, 1);   // one WG per batch row
    submitAndFree(cmdBuf);
    freeDescriptorSet(set);
    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchTreeSpecRejection(
    void* acceptIndex, void* acceptTokenNum, void* acceptToken, void* draftTokens,
    void* targetProbs, void* retrieveNextToken, void* retrieveNextSibling,
    void* treeValid, void* rngSamples,
    uint32_t batchSize, uint32_t numSpeculativeTokens, uint32_t numDraftTokens,
    uint32_t vocabSize, uint32_t kMaxTriedPerLevel, uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
        return VulkanResult::INITIALIZATION_FAILED;
    if (batchSize == 0 || numSpeculativeTokens == 0 || numDraftTokens == 0 || vocabSize == 0)
        return VulkanResult::FEATURE_NOT_PRESENT;

    auto variant = mKernelRegistry->getBestVariant("tree_spec_rejection");
    if (!variant || !variant->isValid())
        return VulkanResult::FEATURE_NOT_PRESENT;

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
        return VulkanResult::UNKNOWN_ERROR;

    // acceptIndex/acceptToken are int64-as-uvec2 (2x uint32 per element)
    VkDeviceSize idxBytes = VkDeviceSize(batchSize) * numSpeculativeTokens * 2u * sizeof(uint32_t);
    VkDeviceSize numBytes = VkDeviceSize(batchSize) * sizeof(uint32_t);
    VkDeviceSize tokBytes = VkDeviceSize(batchSize) * numSpeculativeTokens * 2u * sizeof(uint32_t);
    VkDeviceSize candBytes = VkDeviceSize(batchSize) * (numDraftTokens - 1u) * sizeof(int32_t);
    VkDeviceSize tpBytes  = VkDeviceSize(batchSize) * numDraftTokens * vocabSize * sizeof(float);
    VkDeviceSize rtBytes  = VkDeviceSize(batchSize) * numDraftTokens * sizeof(int32_t);
    VkDeviceSize rsBytes  = rtBytes;
    VkDeviceSize valBytes = VkDeviceSize(batchSize) * sizeof(int32_t);
    VkDeviceSize rngBytes = VkDeviceSize(batchSize) * sizeof(float);

    VkBuffer buffers[] = {
        reinterpret_cast<VkBuffer>(acceptIndex),     reinterpret_cast<VkBuffer>(acceptTokenNum),
        reinterpret_cast<VkBuffer>(acceptToken),     reinterpret_cast<VkBuffer>(draftTokens),
        reinterpret_cast<VkBuffer>(targetProbs),     reinterpret_cast<VkBuffer>(retrieveNextToken),
        reinterpret_cast<VkBuffer>(retrieveNextSibling), reinterpret_cast<VkBuffer>(treeValid),
        reinterpret_cast<VkBuffer>(rngSamples)};
    VkDeviceSize ranges[] = {idxBytes, numBytes, tokBytes, candBytes, tpBytes,
                             rtBytes, rsBytes, valBytes, rngBytes};

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[9]{};
    VkWriteDescriptorSet writes[9]{};
    for (uint32_t i = 0; i < 9; ++i)
    {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].range = ranges[i];
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(mContext->getDevice(), 9, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants
    {
        uint32_t numSpeculativeTokens;
        uint32_t numDraftTokens;
        uint32_t vocabSize;
        uint32_t kMaxTriedPerLevel;
    } pc{};
    pc.numSpeculativeTokens = numSpeculativeTokens;
    pc.numDraftTokens = numDraftTokens;
    pc.vocabSize = vocabSize;
    pc.kMaxTriedPerLevel = kMaxTriedPerLevel;
    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    (void)blockSize;
    vkCmdDispatch(cmdBuf, batchSize, 1, 1);   // one 256-thread WG per batch row
    submitAndFree(cmdBuf);
    freeDescriptorSet(set);
    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchMlaFmha(
    void* q, void* kv, void* pageTable, void* cacheSeqs, void* output,
    uint32_t numHeads, uint32_t seqQLen, uint32_t batchSize,
    uint32_t dLatent, uint32_t dRope, uint32_t pageSize, uint32_t maxPages,
    float softmaxScale, uint32_t slidingWindow, uint32_t storageType,
    float kvScale, uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    if (numHeads == 0 || seqQLen == 0 || batchSize == 0 || dLatent == 0 ||
        dRope == 0 || pageSize == 0 || maxPages == 0)
    {
        TLLM_LOG_WARNING("Vulkan mla_fmha: zero-dimension input (H=%u S=%u B=%u L=%u R=%u ps=%u pg=%u)",
                         numHeads, seqQLen, batchSize, dLatent, dRope, pageSize, maxPages);
        return VulkanResult::FEATURE_NOT_PRESENT;
    }
    if (storageType > 3u)
    {
        TLLM_LOG_WARNING("Vulkan mla_fmha: unsupported storageType=%u (0=fp32,1=fp16,2=fp8,3=bf16)", storageType);
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    auto variant = mKernelRegistry->getBestVariant("mla_fmha");
    if (!variant || !variant->isValid())
    {
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
    {
        return VulkanResult::UNKNOWN_ERROR;
    }

    uint32_t D = dLatent + dRope;
    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(q), reinterpret_cast<VkBuffer>(kv),
                          reinterpret_cast<VkBuffer>(pageTable), reinterpret_cast<VkBuffer>(cacheSeqs),
                          reinterpret_cast<VkBuffer>(output)};

    // 16-bit-packed dtypes occupy half a 32-bit word per element; fp32 one word.
    uint32_t qStride = (storageType == 0u) ? D : (D + 1u) / 2u;
    uint32_t kvStride = (storageType == 0u) ? D : (D + 1u) / 2u;
    uint32_t outStride = (storageType == 0u) ? dLatent : (dLatent + 1u) / 2u;
    VkDeviceSize qBytes     = static_cast<VkDeviceSize>(batchSize) * seqQLen * numHeads * qStride * sizeof(uint32_t);
    VkDeviceSize kvBytes    = static_cast<VkDeviceSize>(maxPages) * pageSize * kvStride * sizeof(uint32_t);
    VkDeviceSize pagBytes   = static_cast<VkDeviceSize>(batchSize) * maxPages * sizeof(int32_t);
    VkDeviceSize seqBytes   = static_cast<VkDeviceSize>(batchSize) * sizeof(int32_t);
    VkDeviceSize outBytes   = static_cast<VkDeviceSize>(batchSize) * seqQLen * numHeads * outStride * sizeof(uint32_t);

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[5]{};
    bufInfos[0].buffer = buffers[0]; bufInfos[0].offset = 0; bufInfos[0].range = qBytes;
    bufInfos[1].buffer = buffers[1]; bufInfos[1].offset = 0; bufInfos[1].range = kvBytes;
    bufInfos[2].buffer = buffers[2]; bufInfos[2].offset = 0; bufInfos[2].range = pagBytes;
    bufInfos[3].buffer = buffers[3]; bufInfos[3].offset = 0; bufInfos[3].range = seqBytes;
    bufInfos[4].buffer = buffers[4]; bufInfos[4].offset = 0; bufInfos[4].range = outBytes;

    VkWriteDescriptorSet writes[5]{};
    for (uint32_t i = 0; i < 5; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 5, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants
    {
        uint32_t numHeads;
        uint32_t seqQLen;
        uint32_t batchSize;
        uint32_t dLatent;
        uint32_t dRope;
        uint32_t pageSize;
        uint32_t maxPages;
        float    softmaxScale;
        uint32_t slidingWindow;
        uint32_t storageType;
        float    kvScale;
    } pc{};
    pc.numHeads      = numHeads;
    pc.seqQLen       = seqQLen;
    pc.batchSize     = batchSize;
    pc.dLatent       = dLatent;
    pc.dRope         = dRope;
    pc.pageSize      = pageSize;
    pc.maxPages      = maxPages;
    pc.softmaxScale  = softmaxScale;
    pc.slidingWindow = slidingWindow;
    pc.storageType   = storageType;
    pc.kvScale       = kvScale;

    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    // One workgroup per query token.
    (void)blockSize;
    vkCmdDispatch(cmdBuf, batchSize, numHeads * seqQLen, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchSpecDecodeAccept(
    void* targetLogits, void* draftLogits, void* uniformRng, void* draftTokens,
    void* acceptCount, void* acceptedTokens, void* resampleProbs,
    uint32_t batchSize, uint32_t draftLen, uint32_t vocabSize,
    float temperature, float acceptProbFloor, uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    if (batchSize == 0 || draftLen == 0 || vocabSize == 0)
    {
        TLLM_LOG_WARNING("Vulkan spec_decode_accept: zero-dimension input (B=%u D=%u V=%u)",
                         batchSize, draftLen, vocabSize);
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    auto variant = mKernelRegistry->getBestVariant("spec_accept");
    if (!variant || !variant->isValid())
    {
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
    {
        return VulkanResult::UNKNOWN_ERROR;
    }

    uint32_t posStride = vocabSize;
    uint32_t posCount = draftLen + 1u; // includes bonus position

    VkDeviceSize logitsBytes  = static_cast<VkDeviceSize>(batchSize) * posCount * posStride * sizeof(float);
    VkDeviceSize uniformBytes = static_cast<VkDeviceSize>(batchSize) * draftLen * sizeof(float);
    VkDeviceSize tokensBytes  = static_cast<VkDeviceSize>(batchSize) * posCount * sizeof(int32_t);
    VkDeviceSize countBytes   = static_cast<VkDeviceSize>(batchSize) * sizeof(uint32_t);
    VkDeviceSize acceptOutBytes = static_cast<VkDeviceSize>(batchSize) * posCount * sizeof(int32_t);
    VkDeviceSize resampleBytes = static_cast<VkDeviceSize>(batchSize) * posStride * sizeof(float);

    VkBuffer buffers[] = {
        reinterpret_cast<VkBuffer>(targetLogits), reinterpret_cast<VkBuffer>(draftLogits),
        reinterpret_cast<VkBuffer>(uniformRng),   reinterpret_cast<VkBuffer>(draftTokens),
        reinterpret_cast<VkBuffer>(acceptCount),  reinterpret_cast<VkBuffer>(acceptedTokens),
        reinterpret_cast<VkBuffer>(resampleProbs)};

    VkDeviceSize ranges[] = {logitsBytes, logitsBytes, uniformBytes, tokensBytes,
                             countBytes, acceptOutBytes, resampleBytes};

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[7]{};
    VkWriteDescriptorSet writes[7]{};
    for (uint32_t i = 0; i < 7; ++i)
    {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].offset = 0;
        bufInfos[i].range = ranges[i];
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 7, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants
    {
        uint32_t draftLen;
        uint32_t vocabSize;
        float    temperature;
        float    sqrtNegLogAccept;
    } pc{};
    pc.draftLen = draftLen;
    pc.vocabSize = vocabSize;
    pc.temperature = temperature;
    pc.sqrtNegLogAccept = acceptProbFloor;

    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    // One workgroup per batch row.
    (void)blockSize;
    vkCmdDispatch(cmdBuf, batchSize, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchMlaFmhaPrefill(
    void* q, void* kv, void* pageTable, void* cacheSeqs, void* output,
    uint32_t numHeads, uint32_t seqQLen, uint32_t batchSize,
    uint32_t dLatent, uint32_t dRope, uint32_t pageSize, uint32_t maxPages,
    bool causal, float softmaxScale, uint32_t slidingWindow, uint32_t storageType,
    float kvScale, uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
    {
        return VulkanResult::INITIALIZATION_FAILED;
    }

    if (numHeads == 0 || seqQLen == 0 || batchSize == 0 || dLatent == 0 ||
        dRope == 0 || pageSize == 0 || maxPages == 0)
    {
        TLLM_LOG_WARNING("Vulkan mla_fmha_prefill: zero-dimension input (H=%u S=%u B=%u L=%u R=%u ps=%u pg=%u)",
                         numHeads, seqQLen, batchSize, dLatent, dRope, pageSize, maxPages);
        return VulkanResult::FEATURE_NOT_PRESENT;
    }
    if (storageType > 3u)
    {
        TLLM_LOG_WARNING("Vulkan mla_fmha_prefill: unsupported storageType=%u (0=fp32,1=fp16,2=fp8,3=bf16)", storageType);
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    auto variant = mKernelRegistry->getBestVariant("mla_fmha_prefill");
    if (!variant || !variant->isValid())
    {
        return VulkanResult::FEATURE_NOT_PRESENT;
    }

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
    {
        return VulkanResult::UNKNOWN_ERROR;
    }

    uint32_t D = dLatent + dRope;
    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(q), reinterpret_cast<VkBuffer>(kv),
                          reinterpret_cast<VkBuffer>(pageTable), reinterpret_cast<VkBuffer>(cacheSeqs),
                          reinterpret_cast<VkBuffer>(output)};

    uint32_t qStride = (storageType == 0u) ? D : (D + 1u) / 2u;
    uint32_t kvStride = (storageType == 0u) ? D : (D + 1u) / 2u;
    uint32_t outStride = (storageType == 0u) ? dLatent : (dLatent + 1u) / 2u;
    VkDeviceSize qBytes     = static_cast<VkDeviceSize>(batchSize) * seqQLen * numHeads * qStride * sizeof(uint32_t);
    VkDeviceSize kvBytes    = static_cast<VkDeviceSize>(maxPages) * pageSize * kvStride * sizeof(uint32_t);
    VkDeviceSize pagBytes   = static_cast<VkDeviceSize>(batchSize) * maxPages * sizeof(int32_t);
    VkDeviceSize seqBytes   = static_cast<VkDeviceSize>(batchSize) * sizeof(int32_t);
    VkDeviceSize outBytes   = static_cast<VkDeviceSize>(batchSize) * seqQLen * numHeads * outStride * sizeof(uint32_t);

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[5]{};
    bufInfos[0].buffer = buffers[0]; bufInfos[0].offset = 0; bufInfos[0].range = qBytes;
    bufInfos[1].buffer = buffers[1]; bufInfos[1].offset = 0; bufInfos[1].range = kvBytes;
    bufInfos[2].buffer = buffers[2]; bufInfos[2].offset = 0; bufInfos[2].range = pagBytes;
    bufInfos[3].buffer = buffers[3]; bufInfos[3].offset = 0; bufInfos[3].range = seqBytes;
    bufInfos[4].buffer = buffers[4]; bufInfos[4].offset = 0; bufInfos[4].range = outBytes;

    VkWriteDescriptorSet writes[5]{};
    for (uint32_t i = 0; i < 5; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 5, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants
    {
        uint32_t numHeads;
        uint32_t seqQLen;
        uint32_t batchSize;
        uint32_t dLatent;
        uint32_t dRope;
        uint32_t pageSize;
        uint32_t maxPages;
        float    softmaxScale;
        uint32_t causal;
        uint32_t slidingWindow;
        uint32_t storageType;
        float    kvScale;
    } pc{};
    pc.numHeads      = numHeads;
    pc.seqQLen       = seqQLen;
    pc.batchSize     = batchSize;
    pc.dLatent       = dLatent;
    pc.dRope         = dRope;
    pc.pageSize      = pageSize;
    pc.maxPages      = maxPages;
    pc.softmaxScale  = softmaxScale;
    pc.causal        = causal ? 1u : 0u;
    pc.slidingWindow = slidingWindow;
    pc.storageType   = storageType;
    pc.kvScale       = kvScale;

    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    // One workgroup per query token (b, h, s).
    (void)blockSize;
    vkCmdDispatch(cmdBuf, batchSize, numHeads, seqQLen);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchFill(
    void* output, float value, size_t elementCount,
    uint32_t blockSize)
{
    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
    {
        return VulkanResult::UNKNOWN_ERROR;
    }

    // Use vkCmdFillBuffer for filling device memory
    vkCmdFillBuffer(cmdBuf, (VkBuffer)output, 0, elementCount * sizeof(float),
        *reinterpret_cast<uint32_t*>(&value));

    submitAndFree(cmdBuf);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchSilu(
    void* input, void* output,
    size_t elementCount,
    uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
        return VulkanResult::INITIALIZATION_FAILED;
    if (elementCount == 0)
        return VulkanResult::FEATURE_NOT_PRESENT;

    auto variant = mKernelRegistry->getBestVariant("silu");
    if (!variant || !variant->isValid())
        return VulkanResult::FEATURE_NOT_PRESENT;

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
        return VulkanResult::UNKNOWN_ERROR;

    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(input),
                           reinterpret_cast<VkBuffer>(output)};
    VkDeviceSize range = static_cast<VkDeviceSize>(elementCount) * sizeof(float);

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].range = range;
        bufInfos[i].offset = 0;
    }

    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 2, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants { uint32_t N; } pc{};
    pc.N = static_cast<uint32_t>(elementCount);
    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    uint32_t workGroupsX = (static_cast<uint32_t>(elementCount) + blockSize - 1u) / blockSize;
    vkCmdDispatch(cmdBuf, workGroupsX, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchGelu(
    void* input, void* output,
    size_t elementCount,
    uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
        return VulkanResult::INITIALIZATION_FAILED;
    if (elementCount == 0)
        return VulkanResult::FEATURE_NOT_PRESENT;

    auto variant = mKernelRegistry->getBestVariant("gelu");
    if (!variant || !variant->isValid())
        return VulkanResult::FEATURE_NOT_PRESENT;

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
        return VulkanResult::UNKNOWN_ERROR;

    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(input),
                           reinterpret_cast<VkBuffer>(output)};
    VkDeviceSize range = static_cast<VkDeviceSize>(elementCount) * sizeof(float);

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].range = range;
        bufInfos[i].offset = 0;
    }

    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 2, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants { uint32_t N; } pc{};
    pc.N = static_cast<uint32_t>(elementCount);
    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    uint32_t workGroupsX = (static_cast<uint32_t>(elementCount) + blockSize - 1u) / blockSize;
    vkCmdDispatch(cmdBuf, workGroupsX, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchSwiglu(
    void* input, void* output,
    uint32_t hiddenDim, uint32_t tokenCount,
    uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
        return VulkanResult::INITIALIZATION_FAILED;
    if (hiddenDim == 0 || tokenCount == 0)
        return VulkanResult::FEATURE_NOT_PRESENT;

    auto variant = mKernelRegistry->getBestVariant("swiglu");
    if (!variant || !variant->isValid())
        return VulkanResult::FEATURE_NOT_PRESENT;

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
        return VulkanResult::UNKNOWN_ERROR;

    // Input has tokenCount * 2 * hiddenDim elements (up + gate concatenated)
    // Output has tokenCount * hiddenDim elements
    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(input),
                           reinterpret_cast<VkBuffer>(output)};
    VkDeviceSize inRange  = static_cast<VkDeviceSize>(tokenCount) * hiddenDim * 2u * sizeof(float);
    VkDeviceSize outRange = static_cast<VkDeviceSize>(tokenCount) * hiddenDim * sizeof(float);

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[2]{};
    bufInfos[0].buffer = buffers[0]; bufInfos[0].range = inRange;  bufInfos[0].offset = 0;
    bufInfos[1].buffer = buffers[1]; bufInfos[1].range = outRange; bufInfos[1].offset = 0;

    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 2, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants { uint32_t N; } pc{};
    pc.N = tokenCount * hiddenDim; // number of output elements
    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    uint32_t totalOutputElems = tokenCount * hiddenDim;
    uint32_t workGroupsX = (totalOutputElems + blockSize - 1u) / blockSize;
    vkCmdDispatch(cmdBuf, workGroupsX, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchSigmoid(
    void* input, void* output,
    size_t elementCount, uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
        return VulkanResult::INITIALIZATION_FAILED;
    if (elementCount == 0)
        return VulkanResult::FEATURE_NOT_PRESENT;

    auto variant = mKernelRegistry->getBestVariant("sigmoid");
    if (!variant || !variant->isValid())
        return VulkanResult::FEATURE_NOT_PRESENT;

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
        return VulkanResult::UNKNOWN_ERROR;

    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(input),
                           reinterpret_cast<VkBuffer>(output)};
    VkDeviceSize range = static_cast<VkDeviceSize>(elementCount) * sizeof(float);

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].range = range;
        bufInfos[i].offset = 0;
    }

    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 2, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants { uint32_t N; } pc{};
    pc.N = static_cast<uint32_t>(elementCount);
    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    uint32_t workGroupsX = (static_cast<uint32_t>(elementCount) + blockSize - 1u) / blockSize;
    vkCmdDispatch(cmdBuf, workGroupsX, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchRelu(
    void* input, void* output,
    size_t elementCount, uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
        return VulkanResult::INITIALIZATION_FAILED;
    if (elementCount == 0)
        return VulkanResult::FEATURE_NOT_PRESENT;

    auto variant = mKernelRegistry->getBestVariant("relu");
    if (!variant || !variant->isValid())
        return VulkanResult::FEATURE_NOT_PRESENT;

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
        return VulkanResult::UNKNOWN_ERROR;

    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(input),
                           reinterpret_cast<VkBuffer>(output)};
    VkDeviceSize range = static_cast<VkDeviceSize>(elementCount) * sizeof(float);

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].range = range;
        bufInfos[i].offset = 0;
    }

    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 2, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants { uint32_t N; } pc{};
    pc.N = static_cast<uint32_t>(elementCount);
    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    uint32_t workGroupsX = (static_cast<uint32_t>(elementCount) + blockSize - 1u) / blockSize;
    vkCmdDispatch(cmdBuf, workGroupsX, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchKVCacheUpdate2D(
    void* kvCacheK, void* kvCacheV,
    void* acceptedDraftTokensIndices2D, void* numAcceptedTokens,
    void* pastKeyValueLengths, void* rewindDraftTokenSeparateAdjustments,
    void* seqSlotRemapping,
    uint32_t batchSize, uint32_t numKVHeads, uint32_t maxKVCacheLen,
    uint32_t headDim, uint32_t maxDraftLen, int32_t rewindDraftTokenCommonCount,
    uint32_t layerCount, uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
        return VulkanResult::INITIALIZATION_FAILED;
    if (batchSize == 0 || numKVHeads == 0 || maxKVCacheLen == 0 || headDim == 0 || maxDraftLen == 0)
        return VulkanResult::FEATURE_NOT_PRESENT;

    auto variant = mKernelRegistry->getBestVariant("kv_cache_update_2d");
    if (!variant || !variant->isValid())
        return VulkanResult::FEATURE_NOT_PRESENT;

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
        return VulkanResult::UNKNOWN_ERROR;

    VkDeviceSize kvBytes    = static_cast<VkDeviceSize>(layerCount) * batchSize * numKVHeads * maxKVCacheLen * headDim * sizeof(float);
    VkDeviceSize accBytes   = static_cast<VkDeviceSize>(batchSize) * maxDraftLen * sizeof(int32_t);
    VkDeviceSize arrBytes   = static_cast<VkDeviceSize>(batchSize) * sizeof(int32_t);

    VkBuffer buffers[] = {
        reinterpret_cast<VkBuffer>(kvCacheK),                   reinterpret_cast<VkBuffer>(kvCacheV),
        reinterpret_cast<VkBuffer>(acceptedDraftTokensIndices2D), reinterpret_cast<VkBuffer>(numAcceptedTokens),
        reinterpret_cast<VkBuffer>(pastKeyValueLengths),         reinterpret_cast<VkBuffer>(rewindDraftTokenSeparateAdjustments),
        reinterpret_cast<VkBuffer>(seqSlotRemapping)};
    VkDeviceSize ranges[] = {kvBytes, kvBytes, accBytes, arrBytes, arrBytes, arrBytes, arrBytes};

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[7]{};
    VkWriteDescriptorSet writes[7]{};
    for (uint32_t i = 0; i < 7; ++i)
    {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].offset = 0;
        bufInfos[i].range = ranges[i];
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(mContext->getDevice(), 7, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants
    {
        uint32_t maxDraftLen;
        uint32_t numKVHeads;
        uint32_t maxKVCacheLen;
        uint32_t headDim;
        int32_t  rewindDraftTokenCommonCount;
        uint32_t layerCount;
        uint32_t seqCount;
    } pc{};
    pc.maxDraftLen = maxDraftLen;
    pc.numKVHeads = numKVHeads;
    pc.maxKVCacheLen = maxKVCacheLen;
    pc.headDim = headDim;
    pc.rewindDraftTokenCommonCount = rewindDraftTokenCommonCount;
    pc.layerCount = layerCount;
    pc.seqCount = batchSize;

    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    (void)blockSize;
    vkCmdDispatch(cmdBuf, batchSize, numKVHeads, layerCount);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchTopKGeneral(
    void* input, void* outputIndices, void* outputValues,
    uint32_t rows, uint32_t cols, uint32_t topk, uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
        return VulkanResult::INITIALIZATION_FAILED;
    if (rows == 0 || cols == 0 || topk == 0 || topk > cols)
        return VulkanResult::FEATURE_NOT_PRESENT;

    auto variant = mKernelRegistry->getBestVariant("topk_general");
    if (!variant || !variant->isValid())
        return VulkanResult::FEATURE_NOT_PRESENT;

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
        return VulkanResult::UNKNOWN_ERROR;

    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(input),
                           reinterpret_cast<VkBuffer>(outputIndices),
                           reinterpret_cast<VkBuffer>(outputValues)};
    VkDeviceSize inRange  = static_cast<VkDeviceSize>(rows) * cols * sizeof(float);
    VkDeviceSize idxRange = static_cast<VkDeviceSize>(rows) * topk * sizeof(uint32_t);
    VkDeviceSize valRange = static_cast<VkDeviceSize>(rows) * topk * sizeof(float);

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[3]{};
    bufInfos[0].buffer = buffers[0]; bufInfos[0].range = inRange;  bufInfos[0].offset = 0;
    bufInfos[1].buffer = buffers[1]; bufInfos[1].range = idxRange; bufInfos[1].offset = 0;
    bufInfos[2].buffer = buffers[2]; bufInfos[2].range = valRange; bufInfos[2].offset = 0;

    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 3, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants { uint32_t rows; uint32_t cols; uint32_t topk; } pc{};
    pc.rows = rows;
    pc.cols = cols;
    pc.topk = topk;
    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    (void)blockSize;
    vkCmdDispatch(cmdBuf, rows, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchSigmoidMul(
    void* a, void* b, void* output,
    size_t elementCount, uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
        return VulkanResult::INITIALIZATION_FAILED;
    if (elementCount == 0)
        return VulkanResult::FEATURE_NOT_PRESENT;

    auto variant = mKernelRegistry->getBestVariant("sigmoid_mul");
    if (!variant || !variant->isValid())
        return VulkanResult::FEATURE_NOT_PRESENT;

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
        return VulkanResult::UNKNOWN_ERROR;

    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(a),
                          reinterpret_cast<VkBuffer>(b),
                          reinterpret_cast<VkBuffer>(output)};
    VkDeviceSize range = static_cast<VkDeviceSize>(elementCount) * sizeof(float);

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[3]{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].range = range;
        bufInfos[i].offset = 0;
    }

    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 3, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants { uint32_t N; } pc{};
    pc.N = static_cast<uint32_t>(elementCount);
    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    uint32_t workGroupsX = (static_cast<uint32_t>(elementCount) + blockSize - 1u) / blockSize;
    vkCmdDispatch(cmdBuf, workGroupsX, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchElementwiseMul(
    void* a, void* b, void* output,
    size_t elementCount, uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
        return VulkanResult::INITIALIZATION_FAILED;
    if (elementCount == 0)
        return VulkanResult::FEATURE_NOT_PRESENT;

    auto variant = mKernelRegistry->getBestVariant("elementwise_mul");
    if (!variant || !variant->isValid())
        return VulkanResult::FEATURE_NOT_PRESENT;

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
        return VulkanResult::UNKNOWN_ERROR;

    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(a),
                          reinterpret_cast<VkBuffer>(b),
                          reinterpret_cast<VkBuffer>(output)};
    VkDeviceSize range = static_cast<VkDeviceSize>(elementCount) * sizeof(float);

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[3]{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].range = range;
        bufInfos[i].offset = 0;
    }

    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 3, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants { uint32_t N; } pc{};
    pc.N = static_cast<uint32_t>(elementCount);
    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    uint32_t workGroupsX = (static_cast<uint32_t>(elementCount) + blockSize - 1u) / blockSize;
    vkCmdDispatch(cmdBuf, workGroupsX, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchScaleRows(
    void* input, void* scale, void* output,
    uint32_t rows, uint32_t cols, uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
        return VulkanResult::INITIALIZATION_FAILED;
    if (rows == 0 || cols == 0)
        return VulkanResult::FEATURE_NOT_PRESENT;

    auto variant = mKernelRegistry->getBestVariant("scale_rows");
    if (!variant || !variant->isValid())
        return VulkanResult::FEATURE_NOT_PRESENT;

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
        return VulkanResult::UNKNOWN_ERROR;

    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(input),
                          reinterpret_cast<VkBuffer>(scale),
                          reinterpret_cast<VkBuffer>(output)};
    VkDeviceSize inRange = static_cast<VkDeviceSize>(rows) * cols * sizeof(float);
    VkDeviceSize scaleRange = static_cast<VkDeviceSize>(rows) * sizeof(float);

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[3]{};
    bufInfos[0].buffer = buffers[0]; bufInfos[0].range = inRange;  bufInfos[0].offset = 0;
    bufInfos[1].buffer = buffers[1]; bufInfos[1].range = scaleRange; bufInfos[1].offset = 0;
    bufInfos[2].buffer = buffers[2]; bufInfos[2].range = inRange;  bufInfos[2].offset = 0;

    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 3, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants { uint32_t rows; uint32_t cols; } pc{};
    pc.rows = rows;
    pc.cols = cols;
    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    uint32_t totalElems = rows * cols;
    uint32_t workGroupsX = (totalElems + blockSize - 1u) / blockSize;
    vkCmdDispatch(cmdBuf, workGroupsX, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchCast(
    void* input, void* output,
    size_t elementCount, int32_t targetDtype, uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
        return VulkanResult::INITIALIZATION_FAILED;
    if (elementCount == 0)
        return VulkanResult::FEATURE_NOT_PRESENT;

    auto variant = mKernelRegistry->getBestVariant("cast");
    if (!variant || !variant->isValid())
        return VulkanResult::FEATURE_NOT_PRESENT;

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
        return VulkanResult::UNKNOWN_ERROR;

    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(input),
                          reinterpret_cast<VkBuffer>(output)};
    VkDeviceSize range = static_cast<VkDeviceSize>(elementCount) * sizeof(float);

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].range = range;
        bufInfos[i].offset = 0;
    }

    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 2, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants { uint32_t N; int32_t dtype; } pc{};
    pc.N = static_cast<uint32_t>(elementCount);
    pc.dtype = targetDtype;
    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    uint32_t workGroupsX = (static_cast<uint32_t>(elementCount) + blockSize - 1u) / blockSize;
    vkCmdDispatch(cmdBuf, workGroupsX, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanKernelDispatcher::dispatchIndexAdd(
    void* output, void* indices, void* values,
    uint32_t outputRows, uint32_t valueRows, uint32_t cols, uint32_t blockSize)
{
    if (!mContext || !mKernelRegistry)
        return VulkanResult::INITIALIZATION_FAILED;
    if (valueRows == 0 || outputRows == 0 || cols == 0)
        return VulkanResult::FEATURE_NOT_PRESENT;

    auto variant = mKernelRegistry->getBestVariant("index_add");
    if (!variant || !variant->isValid())
        return VulkanResult::FEATURE_NOT_PRESENT;

    VkCommandBuffer cmdBuf = acquireCommandBuffer();
    if (cmdBuf == VK_NULL_HANDLE)
        return VulkanResult::UNKNOWN_ERROR;

    VkBuffer buffers[] = {reinterpret_cast<VkBuffer>(output),
                          reinterpret_cast<VkBuffer>(indices),
                          reinterpret_cast<VkBuffer>(values)};
    VkDeviceSize outRange = static_cast<VkDeviceSize>(outputRows) * cols * sizeof(float);
    VkDeviceSize idxRange = static_cast<VkDeviceSize>(valueRows) * sizeof(uint32_t);
    VkDeviceSize valRange = static_cast<VkDeviceSize>(valueRows) * cols * sizeof(float);

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocateDescriptorSet(variant->setLayout, &set))
    {
        submitAndFree(cmdBuf);
        return VulkanResult::UNKNOWN_ERROR;
    }

    VkDescriptorBufferInfo bufInfos[3]{};
    bufInfos[0].buffer = buffers[0]; bufInfos[0].range = outRange; bufInfos[0].offset = 0;
    bufInfos[1].buffer = buffers[1]; bufInfos[1].range = idxRange; bufInfos[1].offset = 0;
    bufInfos[2].buffer = buffers[2]; bufInfos[2].range = valRange; bufInfos[2].offset = 0;

    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(mContext->getDevice(), 3, writes, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            variant->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);

    struct PushConstants { uint32_t rows; uint32_t cols; } pc{};
    pc.rows = valueRows;
    pc.cols = cols;
    vkCmdPushConstants(cmdBuf, variant->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    (void)blockSize;
    uint32_t totalElems = valueRows * cols;
    uint32_t workGroupsX = (totalElems + 255u) / 256u;
    vkCmdDispatch(cmdBuf, workGroupsX, 1, 1);

    submitAndFree(cmdBuf);
    freeDescriptorSet(set);

    return VulkanResult::SUCCESS;
}

} // namespace kernels
TRTLLM_NAMESPACE_END
