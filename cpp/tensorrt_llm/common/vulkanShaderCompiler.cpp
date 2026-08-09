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

#include "tensorrt_llm/common/vulkanShaderCompiler.h"
#include "tensorrt_llm/common/vulkanRuntime.h"

#include <vulkan/vulkan.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

namespace
{

// Helper to execute shell command and capture output
int executeCommand(std::string const& cmd, std::string& output)
{
#ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe)
    {
        return -1;
    }

    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        output += buffer;
    }

#ifdef _WIN32
    int status = _pclose(pipe);
    return status;
#else
    int status = pclose(pipe);
    return WEXITSTATUS(status);
#endif
}

// Find the glslc compiler path
std::string findGlslcPath()
{
    // Check common locations
    std::vector<std::string> candidates = {"glslc", "/usr/bin/glslc", "/usr/local/bin/glslc",
        "C:/VulkanSDK/1.4.357.0/Bin/glslc.exe", "C:/VulkanSDK/Bin/glslc.exe"};

    for (auto const& path : candidates)
    {
        std::string result;
        if (executeCommand(path + " --version 2>&1", result) == 0)
        {
            return path;
        }
    }

    return "";
}

} // namespace

// ==================== Utility Functions ====================

std::string gpuTargetToString(GPUTarget::Architecture arch)
{
    switch (arch)
    {
    case GPUTarget::Architecture::AMD_RDNA1: return "AMD RDNA1 (gfx10xx, wave64)";
    case GPUTarget::Architecture::AMD_RDNA2: return "AMD RDNA2 (gfx103x, wave64)";
    case GPUTarget::Architecture::AMD_RDNA3: return "AMD RDNA3 (gfx11xx, wave32)";
    case GPUTarget::Architecture::AMD_RDNA4: return "AMD RDNA4 (gfx12xx, wave32)";
    case GPUTarget::Architecture::AMD_CDNA1: return "AMD CDNA1 (gfx90a, wave64)";
    case GPUTarget::Architecture::AMD_CDNA2: return "AMD CDNA2 (gfx908, wave64)";
    case GPUTarget::Architecture::AMD_CDNA3: return "AMD CDNA3 (gfx94x, wave64)";
    case GPUTarget::Architecture::AMD_CDNA4: return "AMD CDNA4 (gfx950, wave64)";
    case GPUTarget::Architecture::NVIDIA_TURING: return "NVIDIA Turing";
    case GPUTarget::Architecture::NVIDIA_AMPERE: return "NVIDIA Ampere";
    case GPUTarget::Architecture::NVIDIA_HOPPER: return "NVIDIA Hopper";
    case GPUTarget::Architecture::NVIDIA_BLACKWELL: return "NVIDIA Blackwell";
    default: return "Unknown";
    }
}

// ==================== VulkanShaderCompiler Implementation ====================

std::shared_ptr<VulkanShaderCompiler> VulkanShaderCompiler::create(std::shared_ptr<VulkanContext> const& ctx)
{
    auto compiler = std::shared_ptr<VulkanShaderCompiler>(new VulkanShaderCompiler(ctx));
    if (!compiler->initialize())
    {
        return nullptr;
    }
    return compiler;
}

VulkanShaderCompiler::VulkanShaderCompiler(std::shared_ptr<VulkanContext> const& ctx)
    : mContext(ctx)
{
}

bool VulkanShaderCompiler::initialize()
{
    if (!mContext || mContext->getDevice() == VK_NULL_HANDLE)
    {
        return false;
    }

    mDetectedTarget = detectTarget();
    mCompilerPath = findGlslcPath();

    if (mCompilerPath.empty())
    {
        TLLM_LOG_WARNING("glslc compiler not found - shader compilation will be limited");
        // Try glslangValidator as fallback
        std::string glslangOutput;
        if (executeCommand("glslangValidator --version 2>&1", glslangOutput) == 0)
        {
            mCompilerPath = "glslangValidator";
        }
    }

    TLLM_LOG_INFO("Shader compiler initialized with target: %s", gpuTargetToString(mDetectedTarget.arch).c_str());
    TLLM_LOG_INFO("Shader compiler path: %s", mCompilerPath.c_str());

    return true;
}

VulkanShaderCompiler::~VulkanShaderCompiler()
{
    clearCache();
}

GPUTarget VulkanShaderCompiler::detectTarget() const
{
    GPUTarget target{};

    if (!mContext)
    {
        target.arch = GPUTarget::Architecture::UNKNOWN;
        return target;
    }

    VulkanDeviceInfo const& info = mContext->getDeviceInfo();
    target.majorVersion = VK_VERSION_MAJOR(info.apiVersion);
    target.minorVersion = VK_VERSION_MINOR(info.apiVersion);
    target.gpuID = info.vendorID;
    target.hasCooperativeMatrix = info.hasCooperativeMatrix;
    target.hasFP16 = info.hasFP16;
    target.hasBF16 = info.hasBF16;
    target.subgroupSize = info.subgroupSize;

    // Detect AMD architecture based on device ID
    if (info.vendorID == 0x1002 /* AMD */)
    {
        target.archName = info.deviceName;

        // Extract device family from PCI device ID
        uint32_t deviceID = info.deviceID;

        if ((deviceID & 0xFFFF) == 0x9690 || // RX 9070 XT (RDNA4)
            (deviceID & 0xFFFF) == 0x9691 || // RX 9070 (RDNA4)
            (deviceID & 0xFFFF) == 0x9692 || // RX 9050 (RDNA4)
            (deviceID & 0xFF00) == 0x9600)   // RDNA4 family
        {
            target.arch = GPUTarget::Architecture::AMD_RDNA4;
            target.archName = "RDNA4";
            target.subgroupSize = 32;
        }
        else if ((deviceID & 0xFF00) == 0x7400) // RDNA3 family
        {
            target.arch = GPUTarget::Architecture::AMD_RDNA3;
            target.archName = "RDNA3";
            target.subgroupSize = 32;
        }
        else if ((deviceID & 0xFF00) == 0x7300) // RDNA2 family
        {
            target.arch = GPUTarget::Architecture::AMD_RDNA2;
            target.archName = "RDNA2";
            target.subgroupSize = 64;
        }
        else if ((deviceID & 0xFF00) == 0x9700 || (deviceID & 0xFF00) == 0x9800) // RDNA1/Vega
        {
            target.arch = GPUTarget::Architecture::AMD_RDNA1;
            target.archName = "RDNA1";
            target.subgroupSize = 32;           // RDNA1 supports both wave sizes
        }
        else if ((deviceID & 0xFF00) == 0x9800) // CDNA1
        {
            target.arch = GPUTarget::Architecture::AMD_CDNA1;
            target.archName = "CDNA1";
            target.subgroupSize = 64;
        }
        else if ((deviceID & 0xFF00) == 0x9C00) // CDNA2
        {
            target.arch = GPUTarget::Architecture::AMD_CDNA2;
            target.archName = "CDNA2";
            target.subgroupSize = 64;
        }
        else if ((deviceID & 0xFF00) == 0x9D00 || (deviceID & 0xFF00) == 0x9E00) // CDNA3
        {
            target.arch = GPUTarget::Architecture::AMD_CDNA3;
            target.archName = "CDNA3";
            target.subgroupSize = 64;
        }
        else if ((deviceID & 0xFFFF) == 0x9510) // MI300/CDNA4
        {
            target.arch = GPUTarget::Architecture::AMD_CDNA4;
            target.archName = "CDNA4";
            target.subgroupSize = 64;
        }
        else
        {
            // Default to RDNA2 behavior
            target.arch = GPUTarget::Architecture::AMD_RDNA2;
            target.archName = "Unknown AMD";
            target.subgroupSize = 64;
        }
    }
    else if (info.vendorID == 0x10DE /* NVIDIA */)
    {
        // Parse NVIDIA GPU from device name
        if (info.deviceName.find("RTX 6000") != std::string::npos
            || info.deviceName.find("RTX 5000") != std::string::npos)
        {
            target.arch = GPUTarget::Architecture::NVIDIA_BLACKWELL;
            target.archName = "Blackwell";
        }
        else if (info.deviceName.find("H100") != std::string::npos || info.deviceName.find("H200") != std::string::npos)
        {
            target.arch = GPUTarget::Architecture::NVIDIA_HOPPER;
            target.archName = "Hopper";
        }
        else if (info.deviceName.find("A100") != std::string::npos || info.deviceName.find("A10") != std::string::npos)
        {
            target.arch = GPUTarget::Architecture::NVIDIA_AMPERE;
            target.archName = "Ampere";
        }
        else if (info.deviceName.find("RTX 6000") != std::string::npos)
        {
            target.arch = GPUTarget::Architecture::NVIDIA_BLACKWELL;
            target.archName = "Blackwell";
        }
        else
        {
            target.arch = GPUTarget::Architecture::NVIDIA_AMPERE;
            target.archName = "Unknown NVIDIA";
        }
        target.subgroupSize = 32; // NVIDIA always uses 32-wide warps
    }
    else
    {
        target.arch = GPUTarget::Architecture::UNKNOWN;
        target.archName = "Unknown GPU";
        target.subgroupSize = 32;
    }

    return target;
}

std::string VulkanShaderCompiler::generateCompileOptions(GPUTarget const& target, ShaderFeatureFlags const& features)
{
    std::stringstream opts;

    // The Vulkan target SPIR-V version is selected by --target-env=vulkan1.3 below.
    // The bare -V flag is not portable across glslc/shaderc distributions, so it
    // is omitted; defines use the attached -DNAME form which all glslc/shaderc
    // builds accept.

    // Define architecture-specific macros
    switch (target.arch)
    {
    case GPUTarget::Architecture::AMD_RDNA1: opts << " -DRDNA1"; break;
    case GPUTarget::Architecture::AMD_RDNA2: opts << " -DRDNA2 -DWAVE64"; break;
    case GPUTarget::Architecture::AMD_RDNA3: opts << " -DRDNA3 -DWAVE32"; break;
    case GPUTarget::Architecture::AMD_RDNA4: opts << " -DRDNA4 -DWAVE32"; break;
    case GPUTarget::Architecture::AMD_CDNA1: opts << " -DCDNA1"; break;
    case GPUTarget::Architecture::AMD_CDNA2: opts << " -DCDNA2"; break;
    case GPUTarget::Architecture::AMD_CDNA3: opts << " -DCDNA3"; break;
    case GPUTarget::Architecture::AMD_CDNA4: opts << " -DCDNA4"; break;
    case GPUTarget::Architecture::NVIDIA_TURING: opts << " -DTURING"; break;
    case GPUTarget::Architecture::NVIDIA_AMPERE: opts << " -DAMPERE"; break;
    case GPUTarget::Architecture::NVIDIA_HOPPER: opts << " -DHOPPER"; break;
    case GPUTarget::Architecture::NVIDIA_BLACKWELL: opts << " -DBLACKWELL"; break;
    default: break;
    }

    // Feature-specific defines
    if (features.enableFP16)
    {
        opts << " -DFP16_SUPPORT";
    }
    if (features.enableBF16)
    {
        opts << " -DBF16_SUPPORT";
    }
    if (features.enableCooperativeMatrix)
    {
        opts << " -DCOOP_MATRIX";
    }
    if (features.enableTensorCore)
    {
        opts << " -DTENSOR_CORE";
    }
    if (features.enableTMA)
    {
        opts << " -DTMA_SUPPORT";
    }

    // Set optimization level
    opts << " -O";

    // Target Vulkan 1.3 SPIR-V environment
    opts << " --target-env=vulkan1.3";

    return opts.str();
}

void VulkanShaderCompiler::processSpirV(std::vector<uint32_t>& spirv, GPUTarget const& target)
{
    // In a full implementation, this would patch the SPIR-V to inject
    // target-specific constants like wave size, subgroup size, etc.
    // For now, the shader source handles these via preprocessor defines
}

std::string VulkanShaderCompiler::generateCacheKey(
    std::string const& source, GPUTarget const& target, ShaderFeatureFlags const& features)
{
    std::stringstream key;
    key << target.archName << "_" << target.subgroupSize;
    key << "_fp16:" << features.enableFP16;
    key << "_bf16:" << features.enableBF16;
    key << "_coop:" << features.enableCooperativeMatrix;
    key << "_tc:" << features.enableTensorCore;
    key << "_tma:" << features.enableTMA;
    key << "_" << std::hash<std::string>{}(source);
    return key.str();
}

VulkanResult VulkanShaderCompiler::compileInternal(std::string const& source, std::string const& entryPoint,
    GPUTarget const& target, ShaderFeatureFlags const& features, std::vector<uint32_t>* pSpirvOut, bool isHLSL)
{
    if (!pSpirvOut || source.empty() || mCompilerPath.empty())
    {
        return VulkanResult::INVALID_VALUE;
    }

    // Check cache first
    std::string cacheKey = generateCacheKey(source, target, features);
    auto it = mShaderCache.find(cacheKey);
    if (it != mShaderCache.end())
    {
        *pSpirvOut = it->second;
        return VulkanResult::SUCCESS;
    }

    // Write source to a temporary file with a .comp extension so glslc can
    // infer the shader stage from the filename; a nameless temp file is rejected
    // as "file format not recognized".
    std::string tempFile = std::string(std::tmpnam(nullptr)) + ".comp";
    std::string outputFile = tempFile + ".spv";

    {
        std::ofstream outFile(tempFile);
        if (!outFile.is_open())
        {
            return VulkanResult::UNKNOWN_ERROR;
        }
        outFile << source;
    }

    // Build compilation command. Note: glslc/shaderc default the entry point to
    // `main`, which is what these compute shaders expose, so no -e flag is passed.
    std::stringstream cmd;
    cmd << mCompilerPath;
    cmd << generateCompileOptions(target, features);
    cmd << " -o " << outputFile;
    cmd << " " << tempFile;

    // Execute compilation
    std::string output;
    int ret = executeCommand(cmd.str(), output);

    // Clean up temp source file
    std::remove(tempFile.c_str());

    if (ret != 0)
    {
        TLLM_LOG_ERROR("Shader compilation failed:\nCommand: %s\nOutput: %s", cmd.str().c_str(), output.c_str());
        std::remove(outputFile.c_str());
        return VulkanResult::UNKNOWN_ERROR;
    }

    // Read compiled SPIR-V
    std::ifstream spvFile(outputFile, std::ios::binary | std::ios::ate);
    if (!spvFile.is_open())
    {
        return VulkanResult::UNKNOWN_ERROR;
    }

    size_t fileSize = spvFile.tellg();
    spvFile.seekg(0, std::ios::beg);

    pSpirvOut->resize(fileSize / sizeof(uint32_t));
    spvFile.read(reinterpret_cast<char*>(pSpirvOut->data()), fileSize);
    spvFile.close();

    // Process SPIR-V (patch constants, etc.)
    processSpirV(*pSpirvOut, target);

    // Cache the result
    mShaderCache[cacheKey] = *pSpirvOut;

    // Clean up output file
    std::remove(outputFile.c_str());

    return VulkanResult::SUCCESS;
}

VulkanResult VulkanShaderCompiler::compileGLSL(std::string const& glslSource, std::string const& entryPoint,
    GPUTarget const& target, ShaderFeatureFlags const& features, std::vector<uint32_t>* pSpirvOut)
{
    return compileInternal(glslSource, entryPoint, target, features, pSpirvOut, false);
}

VulkanResult VulkanShaderCompiler::compileHLSL(std::string const& hlslSource, std::string const& entryPoint,
    GPUTarget const& target, ShaderFeatureFlags const& features, std::vector<uint32_t>* pSpirvOut)
{
    return compileInternal(hlslSource, entryPoint, target, features, pSpirvOut, true);
}

VulkanResult VulkanShaderCompiler::compile(std::string const& source, std::string const& entryPoint,
    std::vector<uint32_t>* pSpirvOut, bool isHLSL, ShaderFeatureFlags const* pFeatures, GPUTarget const* pTarget)
{
    GPUTarget target = pTarget ? *pTarget : mDetectedTarget;
    ShaderFeatureFlags features = pFeatures ? *pFeatures : getFeatureFlags();

    if (isHLSL)
    {
        return compileHLSL(source, entryPoint, target, features, pSpirvOut);
    }
    else
    {
        return compileGLSL(source, entryPoint, target, features, pSpirvOut);
    }
}

VulkanResult VulkanShaderCompiler::createShaderModule(std::vector<uint32_t> const& spirv, VkShaderModule* pModule)
{
    if (!mContext || !pModule || spirv.empty())
    {
        return VulkanResult::INVALID_VALUE;
    }

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirv.size() * sizeof(uint32_t);
    createInfo.pCode = spirv.data();

    VkResult result = vkCreateShaderModule(mContext->getDevice(), &createInfo, nullptr, pModule);
    return VulkanRuntime::translateVkResult(result);
}

VulkanResult VulkanShaderCompiler::compileAndCreateShader(std::string const& source, std::string const& entryPoint,
    VkShaderModule* pModule, bool isHLSL, ShaderFeatureFlags const* pFeatures, GPUTarget const* pTarget)
{
    std::vector<uint32_t> spirv;
    VulkanResult result = compile(source, entryPoint, &spirv, isHLSL, pFeatures, pTarget);
    if (result != VulkanResult::SUCCESS)
    {
        return result;
    }

    return createShaderModule(spirv, pModule);
}

VkExtent3D VulkanShaderCompiler::getOptimalLocalSize(VkExtent3D requestedSize) const
{
    // Adjust local size based on GPU capabilities
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(mContext->getPhysicalDevice(), &props);

    uint32_t maxSize = props.limits.maxComputeWorkGroupInvocations;
    uint32_t maxX = props.limits.maxComputeWorkGroupSize[0];
    uint32_t maxY = props.limits.maxComputeWorkGroupSize[1];
    uint32_t maxZ = props.limits.maxComputeWorkGroupSize[2];

    VkExtent3D result = requestedSize;

    // Clamp to device limits (VkExtent3D uses width/height/depth, not x/y/z)
    result.width = std::min(result.width, maxX);
    result.height = std::min(result.height, maxY);
    result.depth = std::min(result.depth, maxZ);

    // Optimize for wave/wavefront size
    uint32_t waveSize = mDetectedTarget.subgroupSize;
    if (waveSize > 1)
    {
        // Round up to nearest multiple of wave size for better occupancy
        result.width = ((result.width + waveSize - 1) / waveSize) * waveSize;
    }

    // Ensure total doesn't exceed max
    uint32_t total = result.width * result.height * result.depth;
    if (total > maxSize)
    {
        // Scale down proportionally
        float scale = static_cast<float>(maxSize) / static_cast<float>(total);
        result.width = static_cast<uint32_t>(static_cast<float>(result.width) * scale);
        result.height = static_cast<uint32_t>(static_cast<float>(result.height) * scale);
        result.depth = static_cast<uint32_t>(static_cast<float>(result.depth) * scale);

        // Round to wave size again
        result.width = (result.width / waveSize) * waveSize;
    }

    return result;
}

ShaderFeatureFlags VulkanShaderCompiler::getFeatureFlags() const
{
    ShaderFeatureFlags flags{};

    if (mContext)
    {
        VulkanDeviceInfo const& info = mContext->getDeviceInfo();
        flags.enableFP16 = info.hasFP16;
        flags.enableCooperativeMatrix = info.hasCooperativeMatrix;
        flags.enableTF32 = true;  // Always available
        flags.enableFP8 = false;  // Would need extension check
        flags.enableInt4 = false; // Would need custom support check
        flags.enableInt8 = true;  // Standard support
    }

    return flags;
}

void VulkanShaderCompiler::clearCache()
{
    mShaderCache.clear();
}

} // namespace common

TRTLLM_NAMESPACE_END
