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

#include "tensorrt_llm/common/vulkanContext.h"
#include "tensorrt_llm/common/vulkanFence.h"
#include "tensorrt_llm/common/vulkanRuntime.h"
#include "tensorrt_llm/common/vulkanMemoryAllocator.h"
#include "tensorrt_llm/common/vulkanShaderCompiler.h"
#include "tensorrt_llm/common/vulkanBackend.h"
#include "tensorrt_llm/kernels/vulkanKernelRegistry.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

// Simple assertion macros for test reporting
#define TRACE_TEST(name) std::cout << "[TRACE] " << #name << "... "; std::cout.flush()
#define TRACE_PASS() std::cout << "PASS" << std::endl
#define TRACE_FAIL(msg) std::cout << "FAIL: " << msg << std::endl

bool test_context_initialization()
{
    TRACE_TEST("VulkanContext Initialization");

    auto ctx = VulkanContext::create(0);
    if (!ctx || ctx->getDevice() == VK_NULL_HANDLE)
    {
        TRACE_FAIL("Failed to create Vulkan context");
        return false;
    }

    auto const& info = ctx->getDeviceInfo();
    std::cout << "      Device: " << info.deviceName << std::endl;
    std::cout << "      Vendor: 0x" << std::hex << info.vendorID << std::dec << std::endl;
    std::cout << "      Subgroup size: " << info.subgroupSize << std::endl;
    std::cout << "      FP16: " << (info.hasFP16 ? "Yes" : "No") << std::endl;
    std::cout << "      Cooperative Matrix: " << (info.hasCooperativeMatrix ? "Yes" : "No") << std::endl;

    TRACE_PASS();
    return true;
}

bool test_gpu_detection()
{
    TRACE_TEST("GPU Architecture Detection");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        TRACE_FAIL("Backend not available");
        return false;
    }

    auto compiler = VulkanShaderCompiler::create(backend->getContext());
    if (!compiler)
    {
        TRACE_FAIL("Shader compiler creation failed");
        return false;
    }

    GPUTarget target = compiler->detectTarget();
    std::cout << "      Detected: " << gpuTargetToString(target.arch) << std::endl;
    std::cout << "      Wave size: " << target.subgroupSize << std::endl;
    std::cout << "      FP16 support: " << (target.hasFP16 ? "Yes" : "No") << std::endl;
    std::cout << "      CoopMatrix support: " << (target.hasCooperativeMatrix ? "Yes" : "No") << std::endl;

    TRACE_PASS();
    return true;
}

bool test_memory_allocation()
{
    TRACE_TEST("Memory Allocation (malloc/free)");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive())
    {
        TRACE_FAIL("Backend not active");
        return false;
    }

    auto memMgr = VulkanMemoryManager::create(backend->getContext());
    if (!memMgr)
    {
        TRACE_FAIL("Memory manager creation failed");
        return false;
    }

    const size_t testSize = 4096;
    void* ptr = VulkanBackend::malloc(testSize);
    if (!ptr)
    {
        TRACE_FAIL("Allocation failed");
        return false;
    }

    // Test memory set
    VulkanBackend::memset(ptr, 0xAA, testSize);

    // Test memory info
    size_t freeMem, totalMem;
    memMgr->getMemoryInfo(&freeMem, &totalMem);
    std::cout << "      Total memory: " << totalMem << " bytes" << std::endl;
    std::cout << "      Free memory: " << freeMem << " bytes" << std::endl;

    VulkanBackend::free(ptr);

    TRACE_PASS();
    return true;
}

bool test_kernel_registry()
{
    TRACE_TEST("Kernel Registry & Compilation");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive())
    {
        TRACE_FAIL("Backend not active");
        return false;
    }

    auto registry = VulkanKernelRegistry::getInstance();
    if (!registry->initialize())
    {
        TRACE_FAIL("Registry initialization failed");
        return false;
    }

    bool compileResult = registry->compileAll();
    if (!compileResult)
    {
        TRACE_FAIL("Kernel compilation failed");
        return false;
    }

    auto kernelNames = registry->getKernelNames();
    std::cout << "      Registered kernels: " << kernelNames.size() << std::endl;
    for (auto const& name : kernelNames)
    {
        std::cout << "        - " << name << std::endl;
    }

    TRACE_PASS();
    return true;
}

bool test_elementwise_kernel()
{
    TRACE_TEST("Elementwise Add Kernel Dispatch");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive())
    {
        TRACE_FAIL("Backend not active");
        return false;
    }

    const size_t elementCount = 1024;
    const size_t bufferSize = elementCount * sizeof(float);

    // Allocate device memory
    void* a = VulkanBackend::malloc(bufferSize);
    void* b = VulkanBackend::malloc(bufferSize);
    void* output = VulkanBackend::malloc(bufferSize);

    if (!a || !b || !output)
    {
        TRACE_FAIL("Device memory allocation failed");
        if (a) VulkanBackend::free(a);
        if (b) VulkanBackend::free(b);
        if (output) VulkanBackend::free(output);
        return false;
    }

    // Create test data
    std::vector<float> hostA(elementCount, 1.0f);
    std::vector<float> hostB(elementCount, 2.0f);

    // Copy to device
    if (!VulkanBackend::memcpyHostToDevice(a, hostA.data(), bufferSize) ||
        !VulkanBackend::memcpyHostToDevice(b, hostB.data(), bufferSize))
    {
        TRACE_FAIL("HostToDevice memcpy failed");
        VulkanBackend::free(a);
        VulkanBackend::free(b);
        VulkanBackend::free(output);
        return false;
    }

    // Launch kernel
    auto start = std::chrono::high_resolution_clock::now();
    bool launchResult = VulkanBackend::launchElementwiseAdd(a, b, output, elementCount);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "      Launch time: " << duration.count() << " us" << std::endl;

    if (!launchResult)
    {
        TRACE_FAIL("Kernel launch failed");
        VulkanBackend::free(a);
        VulkanBackend::free(b);
        VulkanBackend::free(output);
        return false;
    }

    // Copy back and verify
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

    if (!allCorrect)
    {
        TRACE_FAIL("Incorrect results");
    }
    else
    {
        std::cout << "      Result verified: all 1024 elements == 3.0f" << std::endl;
        TRACE_PASS();
    }

    // Cleanup
    VulkanBackend::free(a);
    VulkanBackend::free(b);
    VulkanBackend::free(output);

    return allCorrect;
}

bool test_resource_leak_prevention()
{
    TRACE_TEST("Resource Leak Prevention");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive())
    {
        TRACE_FAIL("Backend not active");
        return false;
    }

    const size_t testSize = 1024;
    const int allocationCount = 100;

    std::vector<void*> ptrs;
    ptrs.reserve(allocationCount);

    // Allocate many buffers
    for (int i = 0; i < allocationCount; ++i)
    {
        void* ptr = VulkanBackend::malloc(testSize);
        if (!ptr)
        {
            TRACE_FAIL("Allocation failed at index " + std::to_string(i));
            // Cleanup what we have so far
            for (auto* p : ptrs)
            {
                VulkanBackend::free(p);
            }
            return false;
        }
        ptrs.push_back(ptr);
    }

    // Free all
    for (void* ptr : ptrs)
    {
        VulkanBackend::free(ptr);
    }

    // Verify we can still allocate
    void* finalPtr = VulkanBackend::malloc(testSize);
    if (!finalPtr)
    {
        TRACE_FAIL("Pool exhaustion after cleanup");
        return false;
    }

    VulkanBackend::free(finalPtr);
    TRACE_PASS();
    return true;
}

bool test_utilization_tracking()
{
    TRACE_TEST("GPU Utilization Tracking");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive())
    {
        TRACE_FAIL("Backend not active");
        return false;
    }

    float util = backend->getGpuUtilization();
    std::cout << "      Initial utilization: " << std::fixed << std::setprecision(2)
              << (util * 100.0f) << "%" << std::endl;

    // Do some work
    const size_t size = 1024 * sizeof(float);
    void* ptr = VulkanBackend::malloc(size);
    VulkanBackend::memset(ptr, 42, size);
    VulkanBackend::deviceSynchronize();

    float postUtil = backend->getGpuUtilization();
    std::cout << "      Post-work utilization: " << std::fixed << std::setprecision(2)
              << (postUtil * 100.0f) << "%" << std::endl;

    VulkanBackend::free(ptr);
    TRACE_PASS();
    return true;
}

bool test_rms_norm()
{
    TRACE_TEST("RMS Norm Kernel Dispatch");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        TRACE_FAIL("Backend not available");
        return false;
    }

    const size_t tokenCount = 8;
    const size_t hiddenDim = 128;
    const size_t totalElements = tokenCount * hiddenDim;
    const size_t inputBytes = totalElements * sizeof(float);
    const size_t featBytes = hiddenDim * sizeof(float);
    const float eps = 1e-6f;

    void* input = VulkanBackend::malloc(inputBytes);
    void* gamma = VulkanBackend::malloc(featBytes);
    void* beta = VulkanBackend::malloc(featBytes);
    void* output = VulkanBackend::malloc(inputBytes);

    if (!input || !gamma || !beta || !output)
    {
        TRACE_FAIL("Device memory allocation failed");
        if (input) VulkanBackend::free(input);
        if (gamma) VulkanBackend::free(gamma);
        if (beta) VulkanBackend::free(beta);
        if (output) VulkanBackend::free(output);
        return false;
    }

    std::vector<float> hostIn(totalElements), hostGamma(hiddenDim), hostBeta(hiddenDim), result(totalElements, 0.0f);
    for (size_t i = 0; i < totalElements; ++i)
    {
        hostIn[i] = 1.0f + (float)(i % 50) * 0.1f;
    }
    for (size_t d = 0; d < hiddenDim; ++d)
    {
        hostGamma[d] = 0.5f + (float)d * 0.01f;
        hostBeta[d] = (float)d * 0.1f;
    }

    if (!VulkanBackend::memcpyHostToDevice(input, hostIn.data(), inputBytes) ||
        !VulkanBackend::memcpyHostToDevice(gamma, hostGamma.data(), featBytes) ||
        !VulkanBackend::memcpyHostToDevice(beta, hostBeta.data(), featBytes))
    {
        TRACE_FAIL("HostToDevice memcpy failed");
        VulkanBackend::free(input);
        VulkanBackend::free(gamma);
        VulkanBackend::free(beta);
        VulkanBackend::free(output);
        return false;
    }

    bool launchResult = VulkanBackend::launchRmsNorm(input, gamma, beta, output, eps, hiddenDim, tokenCount);
    if (!launchResult)
    {
        TRACE_FAIL(std::string("RMS norm launch failed: ") + backend->getLastError());
        VulkanBackend::free(input);
        VulkanBackend::free(gamma);
        VulkanBackend::free(beta);
        VulkanBackend::free(output);
        return false;
    }

    if (!VulkanBackend::memcpyDeviceToHost(result.data(), output, inputBytes))
    {
        TRACE_FAIL("DeviceToHost memcpy failed");
        VulkanBackend::free(input);
        VulkanBackend::free(gamma);
        VulkanBackend::free(beta);
        VulkanBackend::free(output);
        return false;
    }

    // CPU reference: out[tid] = in[tid] * invRms * gamma[dim] + beta[dim]
    bool allCorrect = true;
    for (size_t t = 0; t < tokenCount; ++t)
    {
        size_t base = t * hiddenDim;
        float sumSq = 0.0f;
        for (size_t k = 0; k < hiddenDim; ++k)
        {
            float v = hostIn[base + k];
            sumSq += v * v;
        }
        float invRms = 1.0f / std::sqrt(sumSq / (float)hiddenDim + eps);
        for (size_t d = 0; d < hiddenDim; ++d)
        {
            size_t idx = base + d;
            float ref = hostIn[idx] * invRms * hostGamma[d] + hostBeta[d];
            if (std::abs(result[idx] - ref) > 1e-3f)
            {
                allCorrect = false;
                std::cout << "      mismatch t=" << t << " d=" << d << " got=" << result[idx]
                          << " ref=" << ref << std::endl;
                break;
            }
        }
        if (!allCorrect)
        {
            break;
        }
    }

    if (!allCorrect)
    {
        TRACE_FAIL("Incorrect RMS norm results");
    }
    else
    {
        std::cout << "      Result verified: RMS norm matches CPU reference" << std::endl;
        TRACE_PASS();
    }

    VulkanBackend::free(input);
    VulkanBackend::free(gamma);
    VulkanBackend::free(beta);
    VulkanBackend::free(output);

    return allCorrect;
}

int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "Vulkan Compatibility Layer Trace Tests" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;

    int passed = 0;
    int total = 0;

    auto runTest = [&](auto testFunc, const char* testName) {
        total++;
        std::cout << "[TEST " << std::setw(2) << total << "] " << testName << std::endl;
        try
        {
            if (testFunc())
            {
                passed++;
            }
        }
        catch (std::exception const& e)
        {
            TRACE_FAIL(e.what());
        }
        std::cout << std::endl;
    };

    runTest(test_context_initialization, "Context Initialization");
    runTest(test_gpu_detection, "GPU Architecture Detection");
    runTest(test_memory_allocation, "Memory Allocation");
    runTest(test_kernel_registry, "Kernel Registry");
    runTest(test_elementwise_kernel, "Elementwise Add Kernel");
    runTest(test_rms_norm, "RMS Norm Kernel");
    runTest(test_resource_leak_prevention, "Resource Leak Prevention");
    runTest(test_utilization_tracking, "GPU Utilization Tracking");

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
