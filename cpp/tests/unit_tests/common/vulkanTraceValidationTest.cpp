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

#include "tensorrt_llm/common/vulkanBackend.h"
#include "tensorrt_llm/common/vulkanContext.h"
#include "tensorrt_llm/common/vulkanRuntime.h"
#include "tensorrt_llm/common/vulkanMemoryAllocator.h"
#include "tensorrt_llm/common/vulkanShaderCompiler.h"
#include "tensorrt_llm/kernels/vulkanKernelRegistry.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>
#include <vector>

using namespace tensorrt_llm::common;

// Comprehensive trace validation test suite
class VulkanTraceValidation : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Note: These tests require ENABLE_VULKAN cmake flag
        // They will be skipped if Vulkan is not available
    }

    void TearDown() override
    {
    }
};

// ==================== Complete System Integration Trace ====================

// This test traces the complete path from initialization to kernel dispatch
TEST_F(VulkanTraceValidation, FullSystemInitializationTrace)
{
    // Step 1: Get Vulkan backend singleton
    auto backend = VulkanBackend::getInstance();
    ASSERT_NE(backend, nullptr);

    // Step 2: Initialize Vulkan runtime
    // This triggers:
    //   VulkanRuntime::getInstance()
    //   → VulkanContext::create(gpuID)
    //     → vkCreateInstance, vkEnumeratePhysicalDevices, vkGetPhysicalDeviceProperties
    //     → vkCreateDevice, vkGetDeviceQueue
    //   → GPUDeviceInfo population (vendor detection, feature flagging)
    bool initResult = backend->initialize(0);

    if (!initResult)
    {
        // Vulkan not available - skip this test
        GTEST_SKIP() << "Vulkan backend not available on this system";
    }

    // Step 3: Verify all components are properly initialized
    EXPECT_TRUE(backend->isActive());
    EXPECT_NE(backend->getRuntime(), nullptr);
    EXPECT_NE(backend->getContext(), nullptr);
    EXPECT_NE(backend->getMemoryManager(), nullptr);

    // Step 4: Verify device info is populated
    auto deviceInfo = backend->getContext()->getDeviceInfo();
    EXPECT_FALSE(deviceInfo.deviceName.empty());
    EXPECT_GT(deviceInfo.limits.maxComputeWorkGroupInvocations, 0);

    // Step 5: Verify shader compiler is working
    auto compiler = VulkanShaderCompiler::create(backend->getContext());
    ASSERT_NE(compiler, nullptr);

    GPUTarget target = compiler->detectTarget();
    EXPECT_NE(target.arch, GPUTarget::Architecture::UNKNOWN);
}

// ==================== Memory Allocation Trace ====================

TEST_F(VulkanTraceValidation, MemoryAllocationTrace)
{
    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    // CUDA equivalent: cudaMalloc(&ptr, size)
    const size_t allocationSize = 1024;
    void* devicePtr = VulkanBackend::malloc(allocationSize);
    EXPECT_NE(devicePtr, nullptr);

    if (!devicePtr)
    {
        FAIL() << "Failed to allocate device memory";
        return;
    }

    // CUDA equivalent: cudaMemset(ptr, 0, size)
    VulkanBackend::memset(devicePtr, 0, allocationSize);

    // CUDA equivalent: cudaMemcpy(ptr, hostData, size, cudaMemcpyHostToDevice)
    std::vector<float> hostData(256, 3.14159f);
    void* result = VulkanBackend::memcpyHostToDevice(
        devicePtr, hostData.data(), hostData.size() * sizeof(float));
    EXPECT_NE(result, nullptr);

    // CUDA equivalent: cudaFree(ptr)
    VulkanBackend::free(devicePtr);
    SUCCEED() << "Memory allocation trace completed successfully";
}

// ==================== Kernel Dispatch Trace ====================

TEST_F(VulkanTraceValidation, ElementwiseAddDispatchTrace)
{
    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    const size_t elementCount = 1024;
    const size_t bufferSize = elementCount * sizeof(float);

    // Allocate device memory for tensors (mirrors cudaMalloc)
    void* a = VulkanBackend::malloc(bufferSize);
    void* b = VulkanBackend::malloc(bufferSize);
    void* output = VulkanBackend::malloc(bufferSize);

    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(output, nullptr);

    // Fill input with test data (mirrors cudaMemset + cudaMemcpyHtoD)
    std::vector<float> hostA(elementCount, 1.0f);
    std::vector<float> hostB(elementCount, 2.0f);

    VulkanBackend::memcpyHostToDevice(a, hostA.data(), bufferSize);
    VulkanBackend::memcpyHostToDevice(b, hostB.data(), bufferSize);

    // Launch elementwise add kernel
    // This triggers:
    //   VulkanBackend::launchElementwiseAdd
    //   → VulkanKernelDispatcher::dispatchElementwiseAdd
    //     → VulkanKernelRegistry::getBestVariant("elementwise_add")
    //       → compileAndCreateShader if not cached
    //         → VulkanShaderCompiler::compile (GLSL → SPIR-V via glslc)
    //         → vkCreateShaderModule
    //     → acquireCommandBuffer
    //       → vkAllocateCommandBuffers, vkBeginCommandBuffer
    //     → vkCmdBindPipeline, vkCmdBindDescriptorSets
    //     → vkCmdPushConstants
    //     → vkCmdDispatch
    //     → submitAndFree → vkQueueSubmit, vkQueueWaitIdle
    bool launchResult = VulkanBackend::launchElementwiseAdd(a, b, output, elementCount);
    EXPECT_TRUE(launchResult);

    // Copy result back and verify
    std::vector<float> result(elementCount, 0.0f);
    VulkanBackend::memcpyDeviceToHost(result.data(), output, bufferSize);

    bool allCorrect = true;
    for (size_t i = 0; i < elementCount; ++i)
    {
        if (std::abs(result[i] - 3.0f) > 0.01f)
        {
            allCorrect = false;
            break;
        }
    }
    EXPECT_TRUE(allCorrect) << "Elementwise add produced incorrect results";

    // Cleanup
    VulkanBackend::free(a);
    VulkanBackend::free(b);
    VulkanBackend::free(output);
}

// ==================== GPU Utilization Trace ====================

TEST_F(VulkanTraceValidation, GpuUtilizationTrace)
{
    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    // CUDA equivalent: cuCtxSetCurrent, cuCtxGetDeviceInfo-like queries
    float initialUtil = backend->getGpuUtilization();
    EXPECT_GE(initialUtil, 0.0f);
    EXPECT_LE(initialUtil, 1.0f);

    // Do some work
    const size_t size = 1024;
    void* ptr = VulkanBackend::malloc(size);
    ASSERT_NE(ptr, nullptr);

    VulkanBackend::memset(ptr, 42, size);

    // Wait for completion (mirrors cudaStreamSynchronize)
    VulkanBackend::deviceSynchronize();

    float postWorkUtil = backend->getGpuUtilization();
    EXPECT_GE(postWorkUtil, 0.0f);
    EXPECT_LE(postWorkUtil, 1.0f);

    VulkanBackend::free(ptr);
}

// ==================== Error Handling Trace ====================

TEST_F(VulkanTraceValidation, ErrorHandlingTrace)
{
    // Test NULL pointers and invalid operations
    VulkanBackend::free(nullptr); // Should not crash

    // Test with invalid sizes
    void* ptr = VulkanBackend::malloc(0);
    EXPECT_EQ(ptr, nullptr);

    // Test memcpy with null pointers
    void* nullPtr = nullptr;
    std::vector<char> dummyHost(100);
    void* copyResult = VulkanBackend::memcpyHostToDevice(nullPtr, dummyHost.data(), 100);
    EXPECT_EQ(copyResult, nullptr);
}

// ==================== Wave Size Adaptation Trace ====================

TEST_F(VulkanTraceValidation, WaveSizeAdaptationTrace)
{
    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    auto compiler = VulkanShaderCompiler::create(backend->getContext());
    ASSERT_NE(compiler, nullptr);

    GPUTarget target = compiler->detectTarget();

    // Verify wave size is correctly detected
    switch (target.arch)
    {
        case GPUTarget::Architecture::AMD_RDNA2:
        case GPUTarget::Architecture::AMD_CDNA1:
        case GPUTarget::Architecture::AMD_CDNA2:
        case GPUTarget::Architecture::AMD_CDNA3:
        case GPUTarget::Architecture::AMD_CDNA4:
            // CDNA and RDNA2 typically use wave64
            EXPECT_EQ(target.subgroupSize, 64)
                << "Expected wave64 for " << gpuTargetToString(target.arch);
            break;

        case GPUTarget::Architecture::AMD_RDNA3:
        case GPUTarget::Architecture::AMD_RDNA4:
        case GPUTarget::Architecture::AMD_RDNA1:
            // RDNA3/4 and RDNA1 use wave32
            EXPECT_EQ(target.subgroupSize, 32)
                << "Expected wave32 for " << gpuTargetToString(target.arch);
            break;

        case GPUTarget::Architecture::NVIDIA_TURING:
        case GPUTarget::Architecture::NVIDIA_AMPERE:
        case GPUTarget::Architecture::NVIDIA_HOPPER:
        case GPUTarget::Architecture::NVIDIA_BLACKWELL:
            // NVIDIA always uses 32-wide warps
            EXPECT_EQ(target.subgroupSize, 32);
            break;

        default:
            // Unknown GPU - just verify we got something
            EXPECT_GT(target.subgroupSize, 0);
            break;
    }

    // Verify local size adaptation
    VkExtent3D optimalSize = compiler->getOptimalLocalSize({256, 1, 1});
    EXPECT_GT(optimalSize.x, 0);
}

// ==================== Feature Availability Trace ====================

TEST_F(VulkanTraceValidation, FeatureAvailabilityTrace)
{
    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    auto compiler = VulkanShaderCompiler::create(backend->getContext());
    ASSERT_NE(compiler, nullptr);

    ShaderFeatureFlags flags = compiler->getFeatureFlags();

    // All features should have been probed
    // (actual values depend on GPU capabilities)
    bool anyFeature = flags.enableFP16 ||
                      flags.enableBF16 ||
                      flags.enableCooperativeMatrix ||
                      flags.enableTensorCore ||
                      flags.enableFloat64 ||
                      flags.enableInt8 ||
                      flags.enableTF32 ||
                      flags.enableFP8;

    EXPECT_TRUE(anyFeature) << "At least one feature should be available";

    // Verify cooperative matrix availability
    GPUTarget target = compiler->detectTarget();
    if (target.hasCooperativeMatrix)
    {
        EXPECT_TRUE(flags.enableCooperativeMatrix);
    }
}

// ==================== Kernel Registry Trace ====================

TEST_F(VulkanTraceValidation, KernelRegistryTrace)
{
    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    auto kernelRegistry = kernels::VulkanKernelRegistry::getInstance();
    ASSERT_NE(kernelRegistry, nullptr);

    // Initialize and compile all kernels
    bool initSuccess = kernelRegistry->initialize();
    EXPECT_TRUE(initSuccess);

    bool compileSuccess = kernelRegistry->compileAll();
    EXPECT_TRUE(compileSuccess);

    // Verify we can get kernel names
    auto kernelNames = kernelRegistry->getKernelNames();
    EXPECT_FALSE(kernelNames.empty());

    // Verify specific kernels are registered
    std::set<std::string> nameSet(kernelNames.begin(), kernelNames.end());
    EXPECT_TRUE(nameSet.count("elementwise_add") > 0);
    EXPECT_TRUE(nameSet.count("rms_norm") > 0);
    // Q8_0 GEMM requires cooperative matrix - may not be available on all GPUs
}

// ==================== Memory Layout Verification ====================

TEST_F(VulkanTraceValidation, MemoryAlignmentTrace)
{
    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    auto ctx = backend->getContext();
    auto const& limits = ctx->getDeviceInfo().limits;

    // Verify alignment requirements match expectations
    EXPECT_GT(limits.minStorageBufferOffsetAlignment, 0);
    EXPECT_GT(limits.minUniformBufferOffsetAlignment, 0);

    // Verify alignment values are powers of 2
    EXPECT_TRUE((limits.minStorageBufferOffsetAlignment & (limits.minStorageBufferOffsetAlignment - 1)) == 0);
    EXPECT_TRUE((limits.minUniformBufferOffsetAlignment & (limits.minUniformBufferOffsetAlignment - 1)) == 0);
}

// ==================== Resource Leak Prevention Trace ====================

TEST_F(VulkanTraceValidation, ResourceLeakPreventionTrace)
{
    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    const size_t testSize = 1024;
    const int allocationCount = 100;

    // Allocate many buffers
    std::vector<void*> ptrs;
    ptrs.reserve(allocationCount);

    for (int i = 0; i < allocationCount; ++i)
    {
        void* ptr = VulkanBackend::malloc(testSize);
        EXPECT_NE(ptr, nullptr);
        if (ptr)
        {
            ptrs.push_back(ptr);
        }
    }

    // Free all buffers
    for (void* ptr : ptrs)
    {
        VulkanBackend::free(ptr);
    }

    // Verify we can still allocate (no pool exhaustion)
    void* finalPtr = VulkanBackend::malloc(testSize);
    EXPECT_NE(finalPtr, nullptr);

    if (finalPtr)
    {
        VulkanBackend::free(finalPtr);
    }
}
