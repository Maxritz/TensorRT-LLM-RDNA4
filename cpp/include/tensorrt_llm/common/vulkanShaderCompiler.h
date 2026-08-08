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

#ifndef VULKAN_SHADER_COMPILER_H
#define VULKAN_SHADER_COMPILER_H

#include "tensorrt_llm/common/vulkanContext.h"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

// ShaderVariant: Represents a compiled shader for a specific GPU target
struct ShaderVariant
{
    VkShaderModule module;
    std::string entryPoint;
    uint32_t localSizeX = 0;
    uint32_t localSizeY = 0;
    uint32_t localSizeZ = 0;
    uint32_t subgroupSize = 32; // Default wavefront/wave size
    bool hasCooperativeMatrix = false;
    bool hasFP16 = false;
    bool hasBF16 = false;

    // Performance characteristics of this variant
    float estimatedOccupancy = 0.0f;
    uint32_t vgprCount = 0;
    uint32_t ldsSize = 0;
};

// GPUTarget: Describes a specific GPU architecture for shader compilation
struct GPUTarget
{
    enum class Architecture
    {
        UNKNOWN,
        AMD_RDNA1,   // gfx10xx
        AMD_RDNA2,   // gfx103x, wave64
        AMD_RDNA3,   // gfx11xx, wave32
        AMD_RDNA4,   // gfx12xx, wave32
        AMD_CDNA1,   // gfx90a
        AMD_CDNA2,   // gfx908
        AMD_CDNA3,   // gfx940/941/942
        AMD_CDNA4,   // gfx950
        NVIDIA_TURING,
        NVIDIA_AMPERE,
        NVIDIA_HOPPER,
        NVIDIA_BLACKWELL
    };

    Architecture arch;
    uint32_t majorVersion;
    uint32_t minorVersion;
    uint32_t subgroupSize;     // Wave size (32 or 64)
    bool hasCooperativeMatrix; // Tensor cores support
    bool hasFP16;              // FP16 support
    bool hasBF16;              // BF16 support
    std::string archName;
    uint32_t gpuID;
};

// ShaderFeatureFlags: Feature flags for conditional compilation
struct ShaderFeatureFlags
{
    bool enableFP16 : 1;
    bool enableBF16 : 1;
    bool enableCooperativeMatrix : 1;
    bool enableTensorCore : 1;
    bool enableSparse : 1;
    bool enableTMA : 1;         // Tensor Memory Access (CDNA3+/Blackwell)
    bool enableFloat64 : 1;
    bool enableInt8 : 1;
    bool enableTF32 : 1;
    bool enableFP8 : 1;
    bool enableInt4 : 1;

    // Padding for ABI stability
    uint32_t _reserved : 21;

    ShaderFeatureFlags() : enableFP16(false), enableBF16(false), enableCooperativeMatrix(false),
        enableTensorCore(false), enableSparse(false), enableTMA(false),
        enableFloat64(false), enableInt8(false), enableTF32(false),
        enableFP8(false), enableInt4(false), _reserved(0) {}
};

// VulkanShaderCompiler: Handles shader compilation with target-specific optimizations
class VulkanShaderCompiler
{
public:
    static std::shared_ptr<VulkanShaderCompiler> create(std::shared_ptr<VulkanContext> const& ctx);

    ~VulkanShaderCompiler();

    // Detect GPU target from Vulkan device properties
    GPUTarget detectTarget() const;

    // Compile GLSL source to SPIR-V with target-specific optimizations
    VulkanResult compileGLSL(
        std::string const& glslSource,
        std::string const& entryPoint,
        GPUTarget const& target,
        ShaderFeatureFlags const& features,
        std::vector<uint32_t>* pSpirvOut);

    // Compile HLSL source to SPIR-V
    VulkanResult compileHLSL(
        std::string const& hlslSource,
        std::string const& entryPoint,
        GPUTarget const& target,
        ShaderFeatureFlags const& features,
        std::vector<uint32_t>* pSpirvOut);

    // Compile with automatic variant selection
    VulkanResult compile(
        std::string const& source,
        std::string const& entryPoint,
        std::vector<uint32_t>* pSpirvOut,
        bool isHLSL = false,
        ShaderFeatureFlags const* pFeatures = nullptr,
        GPUTarget const* pTarget = nullptr);

    // Create a shader module from SPIR-V
    VulkanResult createShaderModule(std::vector<uint32_t> const& spirv, VkShaderModule* pModule);

    // Compile and create shader module in one call
    VulkanResult compileAndCreateShader(
        std::string const& source,
        std::string const& entryPoint,
        VkShaderModule* pModule,
        bool isHLSL = false,
        ShaderFeatureFlags const* pFeatures = nullptr,
        GPUTarget const* pTarget = nullptr);

    // Get the optimal local size for a compute shader based on GPU capabilities
    VkExtent3D getOptimalLocalSize(VkExtent3D requestedSize) const;

    // Get feature flags for the current device
    ShaderFeatureFlags getFeatureFlags() const;

    // Cache management
    void clearCache();

private:
    explicit VulkanShaderCompiler(std::shared_ptr<VulkanContext> const& ctx);
    bool initialize();

    // Internal compilation methods
    VulkanResult compileInternal(
        std::string const& source,
        std::string const& entryPoint,
        GPUTarget const& target,
        ShaderFeatureFlags const& features,
        std::vector<uint32_t>* pSpirvOut,
        bool isHLSL);

    // Generate compiler options string
    std::string generateCompileOptions(GPUTarget const& target, ShaderFeatureFlags const& features);

    // Process SPIR-V to inject target-specific constants
    void processSpirV(std::vector<uint32_t>& spirv, GPUTarget const& target);

    // Cache key generation
    std::string generateCacheKey(
        std::string const& source,
        GPUTarget const& target,
        ShaderFeatureFlags const& features);

    std::shared_ptr<VulkanContext> mContext;
    GPUTarget mDetectedTarget;
    bool mInitialized = false;

    // Shader cache
    std::map<std::string, std::vector<uint32_t>> mShaderCache;

    // glslc/glslang binary path
    std::string mCompilerPath;
};

// Utility function to get human-readable GPU target name
std::string gpuTargetToString(GPUTarget::Architecture arch);

} // namespace common
TRTLLM_NAMESPACE_END

#endif // VULKAN_SHADER_COMPILER_H
