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
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <utility>
#include <cstring>
#include <random>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

// ---- Host-side fp16/bf16/fp8 round-trip helpers (mirror the GLSL dequant) ----
static uint32_t fp32ToFp16Bits(float v)
{
    uint32_t bits; std::memcpy(&bits, &v, sizeof(bits));
    uint32_t sign = (bits >> 31u) & 0x1u;
    int32_t  exp  = int32_t((bits >> 23u) & 0xFFu) - 127;
    uint32_t mant = bits & 0x7FFFFFu;
    uint32_t e = uint32_t(exp + 15);
    if (exp + 15 <= 0) return sign << 15u;
    if (exp + 15 >= 31) return (sign << 15u) | 0x7C00u;
    uint32_t m = (mant >> 13u) & 0x3FFu;
    m += (mant >> 12u) & 0x1u;
    if (m > 0x3FFu) { m = 0u; e += 1u; }
    if (e >= 31u) e = 31u;
    return (sign << 15u) | (e << 10u) | m;
}
static float fp16BitsToFp32(uint32_t bits)
{
    uint32_t sign = (bits >> 15u) & 0x1u;
    uint32_t exp  = (bits >> 10u) & 0x1Fu;
    uint32_t mant = bits & 0x3FFu;
    uint32_t f;
    if (exp == 0u) f = 0u;
    else f = ((sign << 31u) | (((exp + 112u) & 0xFFu) << 23u) | (mant << 13u));
    float v; std::memcpy(&v, &f, sizeof(v)); return v;
}
static uint32_t fp32ToBf16Bits(float v)
{
    uint32_t bits; std::memcpy(&bits, &v, sizeof(bits));
    // bf16 = top 16 bits of fp32, round-to-nearest (matches GLSL fp32ToBf16).
    uint32_t bf16 = (bits + 0x8000u) >> 16u;
    return bf16 & 0xFFFFu;
}
static float bf16BitsToFp32(uint32_t bits)
{
    uint32_t f = (bits & 0xFFFFu) << 16u;
    float v; std::memcpy(&v, &f, sizeof(v)); return v;
}
static float fp8BitsToFp32(uint32_t bits, float scale)
{
    uint32_t sign = (bits >> 7u) & 0x1u;
    uint32_t exp  = (bits >> 2u) & 0x1Fu;
    uint32_t mant = bits & 0x3u;
    uint32_t f;
    if (exp == 0u) f = 0u;
    else f = ((sign << 31u) | (((exp + 120u) & 0xFFu) << 23u) | (mant << 21u));
    float v; std::memcpy(&v, &f, sizeof(v)); return v * scale;
}

// Real-valued DeepSeek MLA test-data loader (see scripts/mla_real_data.py).
// Binary layout: magic('MLA1'), then uint32 dims, then raw float Q bytes,
// then raw float KV bytes. Returns false if the file is absent or malformed,
// leaving the caller to fall back to the synthetic RNG stream.
bool loadRealMlaData(std::string const& path,
                     std::vector<float>& hostQ,
                     std::vector<float>& hostKv)
{
    // Resolve the real-weights asset relative to this source file first (so the test
    // passes from any working directory), then fall back to the literal path (CWD).
    std::vector<std::string> candidates;
    std::string sd(__FILE__);
    auto pos = sd.find_last_of("\\/");
    if (pos != std::string::npos)
    {
        std::string base = sd.substr(0, pos);
        candidates.push_back(base + "/" + path);
        candidates.push_back(base + "/tests/" + path);
    }
    candidates.push_back(path);

    FILE* f = nullptr;
    for (auto const& c : candidates)
    {
#if defined(_WIN32)
        fopen_s(&f, c.c_str(), "rb");
#else
        f = std::fopen(c.c_str(), "rb");
#endif
        if (f) break;
    }
    if (!f) return false;

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "MLA1", 4) != 0)
    {
        fclose(f);
        return false;
    }
    uint32_t batchSize = 0, maxPages = 0, seqQLen = 0, numHeads = 0, pageSize = 0, D = 0;
    if (fread(&batchSize, sizeof(uint32_t), 1, f) != 1 ||
        fread(&maxPages,  sizeof(uint32_t), 1, f) != 1 ||
        fread(&seqQLen,   sizeof(uint32_t), 1, f) != 1 ||
        fread(&numHeads,  sizeof(uint32_t), 1, f) != 1 ||
        fread(&pageSize,  sizeof(uint32_t), 1, f) != 1 ||
        fread(&D,         sizeof(uint32_t), 1, f) != 1)
    {
        fclose(f);
        return false;
    }
    uint32_t qCount  = batchSize * seqQLen * numHeads * D;
    uint32_t kvCount = maxPages * pageSize * D;
    if (qCount != hostQ.size() || kvCount != hostKv.size())
    {
        fclose(f);
        return false;
    }
    if (fread(hostQ.data(),  sizeof(float), qCount,  f) != qCount  ||
        fread(hostKv.data(), sizeof(float), kvCount, f) != kvCount)
    {
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

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

// ==================== fp16 GEMM (naive) ====================
// C[M,N] = A[M,K] * B[K,N], fp32 accumulation. Verified against a CPU reference.
bool test_fp16_gemm()
{
    TRACE_TEST("FP16 GEMM Kernel Dispatch");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        TRACE_FAIL("Backend not available");
        return false;
    }

    const uint32_t M = 16;
    const uint32_t N = 16;
    const uint32_t K = 16;
    const size_t aBytes = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bBytes = static_cast<size_t>(K) * N * sizeof(float);
    const size_t oBytes = static_cast<size_t>(M) * N * sizeof(float);

    void* a = VulkanBackend::malloc(aBytes);
    void* b = VulkanBackend::malloc(bBytes);
    void* output = VulkanBackend::malloc(oBytes);
    if (!a || !b || !output)
    {
        TRACE_FAIL("Device memory allocation failed");
        if (a) VulkanBackend::free(a);
        if (b) VulkanBackend::free(b);
        if (output) VulkanBackend::free(output);
        return false;
    }

    std::minstd_rand rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> hostA(M * K), hostB(K * N), result(M * N, 0.0f);
    for (auto& v : hostA)
    {
        v = dist(rng);
    }
    for (auto& v : hostB)
    {
        v = dist(rng);
    }

    if (!VulkanBackend::memcpyHostToDevice(a, hostA.data(), aBytes) ||
        !VulkanBackend::memcpyHostToDevice(b, hostB.data(), bBytes))
    {
        TRACE_FAIL("HostToDevice memcpy failed");
        VulkanBackend::free(a);
        VulkanBackend::free(b);
        VulkanBackend::free(output);
        return false;
    }

    if (!VulkanBackend::launchFp16Gemm(a, b, output, M, N, K))
    {
        TRACE_FAIL(std::string("FP16 GEMM launch failed: ") + backend->getLastError());
        VulkanBackend::free(a);
        VulkanBackend::free(b);
        VulkanBackend::free(output);
        return false;
    }

    VulkanBackend::memcpyDeviceToHost(result.data(), output, oBytes);

    bool allCorrect = true;
    for (uint32_t r = 0; r < M && allCorrect; ++r)
    {
        for (uint32_t c = 0; c < N; ++c)
        {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
            {
                ref += hostA[r * K + k] * hostB[k * N + c];
            }
            if (std::abs(result[r * N + c] - ref) > 1e-3f)
            {
                allCorrect = false;
                std::cout << "      mismatch r=" << r << " c=" << c << " got=" << result[r * N + c]
                          << " ref=" << ref << std::endl;
                break;
            }
        }
    }

    VulkanBackend::free(a);
    VulkanBackend::free(b);
    VulkanBackend::free(output);

    if (!allCorrect)
    {
        TRACE_FAIL("Incorrect fp16 GEMM results");
    }
    else
    {
        std::cout << "      Result verified: fp16 GEMM matches CPU reference" << std::endl;
        TRACE_PASS();
    }

    return allCorrect;
}

// ==================== Q8_0 GEMM (dequantized) ====================
// C[M,N] = A[M,K] * W_dequant[N,K]^T  with Q8_0 blocks (1 fp32 scale + 32 int8).
bool test_q8_0_gemm()
{
    TRACE_TEST("Q8_0 GEMM Kernel Dispatch");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        TRACE_FAIL("Backend not available");
        return false;
    }

    const uint32_t M = 8;
    const uint32_t N = 12;
    const uint32_t K = 64; // must be a multiple of 32
    const uint32_t blocksPerRow = K / 32u;
    const size_t weightBytes = static_cast<size_t>(N) * blocksPerRow * 36u;
    const size_t aBytes = static_cast<size_t>(M) * K * sizeof(float);
    const size_t oBytes = static_cast<size_t>(M) * N * sizeof(float);

    void* weight = VulkanBackend::malloc(weightBytes);
    void* activation = VulkanBackend::malloc(aBytes);
    void* output = VulkanBackend::malloc(oBytes);
    if (!weight || !activation || !output)
    {
        TRACE_FAIL("Device memory allocation failed");
        if (weight) VulkanBackend::free(weight);
        if (activation) VulkanBackend::free(activation);
        if (output) VulkanBackend::free(output);
        return false;
    }

    std::minstd_rand rng(98765);
    std::uniform_real_distribution<float> dAct(0.0f, 1.0f);
    std::uniform_int_distribution<int> dInt(-4, 4);

    std::vector<uint8_t> hostW(weightBytes);
    std::vector<float> hostA(M * K), hostWdeq(N * K, 0.0f), result(M * N, 0.0f);
    for (uint32_t n = 0; n < N; ++n)
    {
        for (uint32_t blk = 0; blk < blocksPerRow; ++blk)
        {
            uint32_t base = (n * blocksPerRow + blk) * 36u;
            float scale = 1.0f;
            uint32_t bits;
            std::memcpy(&bits, &scale, sizeof(bits));
            hostW[base + 0] = static_cast<uint8_t>(bits & 0xFFu);
            hostW[base + 1] = static_cast<uint8_t>((bits >> 8) & 0xFFu);
            hostW[base + 2] = static_cast<uint8_t>((bits >> 16) & 0xFFu);
            hostW[base + 3] = static_cast<uint8_t>((bits >> 24) & 0xFFu);
            for (int w = 0; w < 32; ++w)
            {
                int8_t iv = static_cast<int8_t>(dInt(rng));
                hostW[base + 4 + static_cast<uint32_t>(w)] = static_cast<uint8_t>(iv);
                uint32_t k = blk * 32u + static_cast<uint32_t>(w);
                if (k < K)
                {
                    hostWdeq[n * K + k] = scale * static_cast<float>(iv);
                }
            }
        }
    }
    for (auto& v : hostA)
    {
        v = dAct(rng);
    }

    if (!VulkanBackend::memcpyHostToDevice(weight, hostW.data(), weightBytes) ||
        !VulkanBackend::memcpyHostToDevice(activation, hostA.data(), aBytes))
    {
        TRACE_FAIL("HostToDevice memcpy failed");
        VulkanBackend::free(weight);
        VulkanBackend::free(activation);
        VulkanBackend::free(output);
        return false;
    }

    if (!VulkanBackend::launchQ8_0Gemm(weight, activation, output, M, N, K, blocksPerRow))
    {
        TRACE_FAIL(std::string("Q8_0 GEMM launch failed: ") + backend->getLastError());
        VulkanBackend::free(weight);
        VulkanBackend::free(activation);
        VulkanBackend::free(output);
        return false;
    }

    VulkanBackend::memcpyDeviceToHost(result.data(), output, oBytes);

    bool allCorrect = true;
    for (uint32_t m = 0; m < M && allCorrect; ++m)
    {
        for (uint32_t n = 0; n < N; ++n)
        {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
            {
                ref += hostA[m * K + k] * hostWdeq[n * K + k];
            }
            if (std::abs(result[m * N + n] - ref) > 1e-2f)
            {
                allCorrect = false;
                std::cout << "      mismatch m=" << m << " n=" << n << " got=" << result[m * N + n]
                          << " ref=" << ref << std::endl;
                break;
            }
        }
    }

    VulkanBackend::free(weight);
    VulkanBackend::free(activation);
    VulkanBackend::free(output);

    if (!allCorrect)
    {
        TRACE_FAIL("Incorrect Q8_0 GEMM results");
    }
    else
    {
        std::cout << "      Result verified: Q8_0 GEMM matches CPU reference" << std::endl;
        TRACE_PASS();
    }

    VulkanBackend::free(weight);
    VulkanBackend::free(activation);
    VulkanBackend::free(output);

    return allCorrect;
}

// ==================== Softmax (per-row) ====================
// out[b,h,s] = exp(in - max) / sum(exp(in - max)), computed per (batch, head) row.
bool test_softmax()
{
    TRACE_TEST("Softmax Kernel Dispatch");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        TRACE_FAIL("Backend not available");
        return false;
    }

    const uint32_t batchSize = 2;
    const uint32_t numHeads = 2;
    const uint32_t seqLen = 8;
    const uint32_t rows = batchSize * numHeads;
    const uint32_t total = rows * seqLen;
    const size_t bytes = static_cast<size_t>(total) * sizeof(float);

    void* input = VulkanBackend::malloc(bytes);
    void* output = VulkanBackend::malloc(bytes);
    if (!input || !output)
    {
        TRACE_FAIL("Device memory allocation failed");
        if (input) VulkanBackend::free(input);
        if (output) VulkanBackend::free(output);
        return false;
    }

    std::minstd_rand rng(424242);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> hostIn(total), result(total, 0.0f);
    for (auto& v : hostIn)
    {
        v = dist(rng);
    }

    if (!VulkanBackend::memcpyHostToDevice(input, hostIn.data(), bytes))
    {
        TRACE_FAIL("HostToDevice memcpy failed");
        VulkanBackend::free(input);
        VulkanBackend::free(output);
        return false;
    }

    if (!VulkanBackend::launchSoftmax(input, output, batchSize, numHeads, seqLen))
    {
        TRACE_FAIL(std::string("Softmax launch failed: ") + backend->getLastError());
        VulkanBackend::free(input);
        VulkanBackend::free(output);
        return false;
    }

    VulkanBackend::memcpyDeviceToHost(result.data(), output, bytes);

    bool allCorrect = true;
    for (uint32_t r = 0; r < rows && allCorrect; ++r)
    {
        uint32_t base = r * seqLen;

        float maxVal = hostIn[base];
        for (uint32_t k = 1; k < seqLen; ++k)
        {
            if (hostIn[base + k] > maxVal) maxVal = hostIn[base + k];
        }

        float sum = 0.0f;
        std::vector<float> ref(seqLen);
        for (uint32_t k = 0; k < seqLen; ++k)
        {
            ref[k] = std::exp(hostIn[base + k] - maxVal);
            sum += ref[k];
        }
        float inv = 1.0f / sum;

        float rowSum = 0.0f;
        for (uint32_t k = 0; k < seqLen; ++k)
        {
            ref[k] *= inv;
            rowSum += result[base + k];
            if (std::abs(result[base + k] - ref[k]) > 1e-4f)
            {
                allCorrect = false;
                std::cout << "      mismatch r=" << r << " c=" << k << " got=" << result[base + k]
                          << " ref=" << ref[k] << std::endl;
                break;
            }
        }

        if (allCorrect && std::abs(rowSum - 1.0f) > 1e-3f)
        {
            allCorrect = false;
            std::cout << "      row sum " << rowSum << " != 1.0 for r=" << r << std::endl;
        }
    }

    VulkanBackend::free(input);
    VulkanBackend::free(output);

    if (!allCorrect)
    {
        TRACE_FAIL("Incorrect softmax results");
    }
    else
    {
        std::cout << "      Result verified: softmax matches CPU reference" << std::endl;
        TRACE_PASS();
    }

    return allCorrect;
}

// ==================== Attention (naive causal SDPA) ====================
// out[b,h,i,t] = sum_j softmax_causal((Q[i] . K[j]) / sqrt(headDim)) * V[j,t]
bool test_attention()
{
    TRACE_TEST("Attention Kernel Dispatch");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        TRACE_FAIL("Backend not available");
        return false;
    }

    const uint32_t batchSize = 1;
    const uint32_t numHeads = 2;
    const uint32_t seqLenQ = 3;
    const uint32_t seqLenK = 4;
    const uint32_t headDim = 4;
    const bool causal = true;
    const size_t qBytes  = static_cast<size_t>(batchSize) * numHeads * seqLenQ * headDim * sizeof(float);
    const size_t kvBytes = static_cast<size_t>(batchSize) * numHeads * seqLenK * headDim * sizeof(float);
    const size_t oBytes  = qBytes;

    void* q = VulkanBackend::malloc(qBytes);
    void* k = VulkanBackend::malloc(kvBytes);
    void* v = VulkanBackend::malloc(kvBytes);
    void* output = VulkanBackend::malloc(oBytes);
    if (!q || !k || !v || !output)
    {
        TRACE_FAIL("Device memory allocation failed");
        if (q) VulkanBackend::free(q);
        if (k) VulkanBackend::free(k);
        if (v) VulkanBackend::free(v);
        if (output) VulkanBackend::free(output);
        return false;
    }

    std::minstd_rand rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> hostQ(qBytes / sizeof(float)), hostK(kvBytes / sizeof(float)),
        hostV(kvBytes / sizeof(float)), result(oBytes / sizeof(float), 0.0f);
    for (auto& val : hostQ)
    {
        val = dist(rng);
    }
    for (auto& val : hostK)
    {
        val = dist(rng);
    }
    for (auto& val : hostV)
    {
        val = dist(rng);
    }

    if (!VulkanBackend::memcpyHostToDevice(q, hostQ.data(), qBytes) ||
        !VulkanBackend::memcpyHostToDevice(k, hostK.data(), kvBytes) ||
        !VulkanBackend::memcpyHostToDevice(v, hostV.data(), kvBytes))
    {
        TRACE_FAIL("HostToDevice memcpy failed");
        VulkanBackend::free(q);
        VulkanBackend::free(k);
        VulkanBackend::free(v);
        VulkanBackend::free(output);
        return false;
    }

    if (!VulkanBackend::launchAttention(q, k, v, output, batchSize, numHeads, seqLenQ, seqLenK,
                                         headDim, causal))
    {
        TRACE_FAIL(std::string("Attention launch failed: ") + backend->getLastError());
        VulkanBackend::free(q);
        VulkanBackend::free(k);
        VulkanBackend::free(v);
        VulkanBackend::free(output);
        return false;
    }

    VulkanBackend::memcpyDeviceToHost(result.data(), output, oBytes);

    auto qIdx = [&](uint32_t b, uint32_t h, uint32_t i, uint32_t d) {
        return ((b * numHeads + h) * seqLenQ + i) * headDim + d;
    };
    auto kIdx = [&](uint32_t b, uint32_t h, uint32_t j, uint32_t d) {
        return ((b * numHeads + h) * seqLenK + j) * headDim + d;
    };

    float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    bool allCorrect = true;

    for (uint32_t b = 0; b < batchSize && allCorrect; ++b)
    {
        for (uint32_t h = 0; h < numHeads && allCorrect; ++h)
        {
            for (uint32_t i = 0; i < seqLenQ && allCorrect; ++i)
            {
                auto score = [&](uint32_t j) -> float {
                    if (causal && j > i)
                    {
                        return -1.0e30f;
                    }
                    float s = 0.0f;
                    for (uint32_t d = 0; d < headDim; ++d)
                    {
                        s += hostQ[qIdx(b, h, i, d)] * hostK[kIdx(b, h, j, d)];
                    }
                    return scale * s;
                };

                float maxVal = -1.0e30f;
                for (uint32_t j = 0; j < seqLenK; ++j)
                {
                    float s = score(j);
                    if (s > maxVal) maxVal = s;
                }

                float sum = 0.0f;
                for (uint32_t j = 0; j < seqLenK; ++j)
                {
                    sum += std::exp(score(j) - maxVal);
                }
                float inv = 1.0f / sum;

                for (uint32_t t = 0; t < headDim && allCorrect; ++t)
                {
                    float acc = 0.0f;
                    for (uint32_t j = 0; j < seqLenK; ++j)
                    {
                        float s = score(j);
                        acc += std::exp(s - maxVal) * inv * hostV[kIdx(b, h, j, t)];
                    }
                    uint32_t oi = qIdx(b, h, i, t);
                    if (std::abs(result[oi] - acc) > 1e-4f)
                    {
                        allCorrect = false;
                        std::cout << "      mismatch b=" << b << " h=" << h << " i=" << i
                                  << " t=" << t << " got=" << result[oi] << " ref=" << acc << std::endl;
                    }
                }
            }
        }
    }

    VulkanBackend::free(q);
    VulkanBackend::free(k);
    VulkanBackend::free(v);
    VulkanBackend::free(output);

    if (!allCorrect)
    {
        TRACE_FAIL("Incorrect attention results");
    }
    else
    {
        std::cout << "      Result verified: attention matches CPU reference" << std::endl;
        TRACE_PASS();
    }

    return allCorrect;
}

// ==================== Top-K (sparse token selection) ====================
// Per-(batch, kv-head) top-k of attention scores. Mirrors topk_kernel in
// tensorrt-llm/_torch/attention_backend/sparse/kernel.py (triton_topk).
bool test_topk()
{
    TRACE_TEST("Top-K Kernel Dispatch");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        TRACE_FAIL("Backend not available");
        return false;
    }

    const uint32_t batchSize      = 2;
    const uint32_t numHeads      = 3;
    const uint32_t topk          = 4;
    const uint32_t totalTokens   = 16;
    // b0: tokens [0,6)  len 6 ; b1: tokens [6,16) len 10
    const uint32_t inputOffsets[3]  = {0, 6, 16};
    // output per row = min(input_len, topk) => 4, 4 => 8 per head row
    const uint32_t outputOffsets[3] = {0, 4, 8};
    const uint32_t totalOutputTokens = 8;

    const size_t scoresBytes = static_cast<size_t>(numHeads) * totalTokens * sizeof(float);
    const size_t offBytes    = static_cast<size_t>(batchSize + 1) * sizeof(uint32_t);
    const size_t idxBytes    = static_cast<size_t>(numHeads) * totalOutputTokens * sizeof(int32_t);

    void* scoresDev      = VulkanBackend::malloc(scoresBytes);
    void* inOffDev       = VulkanBackend::malloc(offBytes);
    void* outOffDev      = VulkanBackend::malloc(offBytes);
    void* topkIdxDev     = VulkanBackend::malloc(idxBytes);
    if (!scoresDev || !inOffDev || !outOffDev || !topkIdxDev)
    {
        TRACE_FAIL("Device memory allocation failed");
        if (scoresDev) VulkanBackend::free(scoresDev);
        if (inOffDev)  VulkanBackend::free(inOffDev);
        if (outOffDev) VulkanBackend::free(outOffDev);
        if (topkIdxDev) VulkanBackend::free(topkIdxDev);
        return false;
    }

    std::minstd_rand rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> hostScores(scoresBytes / sizeof(float));
    for (auto& val : hostScores)
    {
        val = dist(rng);
    }
    std::vector<uint32_t> hostInOff(inputOffsets, inputOffsets + batchSize + 1);
    std::vector<uint32_t> hostOutOff(outputOffsets, outputOffsets + batchSize + 1);
    std::vector<int32_t> result(idxBytes / sizeof(int32_t), -123);

    if (!VulkanBackend::memcpyHostToDevice(scoresDev, hostScores.data(), scoresBytes) ||
        !VulkanBackend::memcpyHostToDevice(inOffDev,  hostInOff.data(),  offBytes) ||
        !VulkanBackend::memcpyHostToDevice(outOffDev, hostOutOff.data(), offBytes))
    {
        TRACE_FAIL("HostToDevice memcpy failed");
        VulkanBackend::free(scoresDev);
        VulkanBackend::free(inOffDev);
        VulkanBackend::free(outOffDev);
        VulkanBackend::free(topkIdxDev);
        return false;
    }

    if (!VulkanBackend::launchTopk(scoresDev, inOffDev, outOffDev, topkIdxDev,
                                   topk, numHeads, batchSize, totalTokens,
                                   totalOutputTokens))
    {
        TRACE_FAIL(std::string("Top-K launch failed: ") + backend->getLastError());
        VulkanBackend::free(scoresDev);
        VulkanBackend::free(inOffDev);
        VulkanBackend::free(outOffDev);
        VulkanBackend::free(topkIdxDev);
        return false;
    }

    VulkanBackend::memcpyDeviceToHost(result.data(), topkIdxDev, idxBytes);

    bool allCorrect = true;

    // CPU reference: per (batch,head) row, top-k indices by score desc, first-max ties.
    for (uint32_t b = 0; b < batchSize && allCorrect; ++b)
    {
        uint32_t inStart  = inputOffsets[b];
        uint32_t inEnd    = inputOffsets[b + 1];
        uint32_t outStart = outputOffsets[b];
        uint32_t outEnd   = outputOffsets[b + 1];
        uint32_t inputLen  = inEnd - inStart;
        uint32_t outputLen = outEnd - outStart;
        uint32_t picks     = std::min(outputLen, topk);

        for (uint32_t h = 0; h < numHeads && allCorrect; ++h)
        {
            uint32_t rowBase = h * totalTokens + inStart;

            // Gather (value, localIndex) candidates, sort desc with first-max tie-break.
            std::vector<std::pair<float, uint32_t>> cand;
            cand.reserve(inputLen);
            for (uint32_t i = 0; i < inputLen; ++i)
            {
                cand.emplace_back(hostScores[rowBase + i], i);
            }
            std::sort(cand.begin(), cand.end(),
                      [](std::pair<float, uint32_t> const& a, std::pair<float, uint32_t> const& b) {
                          if (a.first != b.first)
                          {
                              return a.first > b.first;
                          }
                          return a.second < b.second;
                      });

            uint32_t outBase = h * totalOutputTokens + outStart;
            for (uint32_t s = 0; s < picks; ++s)
            {
                int32_t got = result[outBase + s];
                int32_t ref = static_cast<int32_t>(cand[s].second);
                if (got != ref)
                {
                    allCorrect = false;
                    std::cout << "      mismatch b=" << b << " h=" << h << " s=" << s
                              << " got=" << got << " ref=" << ref << std::endl;
                }
            }
            // Any slots beyond picks must be left untouched (-123 sentinel); the kernel
            // only writes `picks` entries, so the rest stays as host init.
        }
    }

    VulkanBackend::free(scoresDev);
    VulkanBackend::free(inOffDev);
    VulkanBackend::free(outOffDev);
    VulkanBackend::free(topkIdxDev);

    if (!allCorrect)
    {
        TRACE_FAIL("Incorrect top-k results");
    }
    else
    {
        std::cout << "      Result verified: top-k matches CPU reference" << std::endl;
        TRACE_PASS();
    }

    return allCorrect;
}

bool test_spec_decode_accept()
{
    TRACE_TEST("Spec-Decode Acceptance Kernel Dispatch");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        TRACE_FAIL("Backend not available");
        return false;
    }

    const uint32_t B         = 2;
    const uint32_t draftLen  = 4;
    const uint32_t V         = 8;
    const uint32_t posCount  = draftLen + 1u;
    const float    temperature = 1.0f;
    const float    acceptProbFloor = 1.0f; // sqrt(negLogAccept) — no floor effect

    // ---- Deterministic host data (seed=99) ----
    std::minstd_rand rng(99);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // targetLogits: all draftLen+1 positions valid. b0 fully accepted path,
    // b1 rejects at position 1.
    std::vector<float> hostTarget(B * posCount * V);
    std::vector<float> hostDraft(B * posCount * V);
    std::vector<float> hostUniform(B * draftLen, 0.0f);
    std::vector<int32_t> hostDraftTokens(B * posCount);
    // draft tokens chosen from vocab (0..V-1)
    for (uint32_t b = 0; b < B; ++b)
        for (uint32_t p = 0; p < posCount; ++p)
            hostDraftTokens[b * posCount + p] = static_cast<int32_t>((b + p) % V);

    // b0: target >> draft at every position => ratios huge => uniform < 1 => accept all
    // b1: position 0 accept, position 1 draft prob >> target prob => ratio tiny => reject
    for (uint32_t i = 0; i < hostTarget.size(); ++i) hostTarget[i] = 1.0f;
    for (uint32_t i = 0; i < hostDraft.size(); ++i)  hostDraft[i]  = 1.0f;
    for (uint32_t b = 0; b < B; ++b)
        for (uint32_t t = 0; t < draftLen; ++t)
            hostUniform[b * draftLen + t] = 0.5f; // middle of [0,1)

    // b0: make ratios large everywhere (1.0 each). b1: at t=1, make target tiny => reject.
    hostUniform[1 * draftLen + 1] = 0.01f; // b1 rejects at pos 1 regardless
    // Force b1 to reject at t=1: set target logit at the draft token low, draft high.
    {
        uint32_t pos = 1;
        int32_t tok = hostDraftTokens[1 * posCount + pos];
        uint32_t row = 1 * posCount * V + pos * V;
        hostTarget[row + tok] = -20.0f;  // tiny target prob => ratio ~0 => reject
        hostDraft[row + tok]  =  20.0f; // large draft prob
    }

    const size_t logitsBytes   = B * posCount * V * sizeof(float);
    const size_t uniformBytes  = B * draftLen * sizeof(float);
    const size_t tokensBytes   = B * posCount * sizeof(int32_t);
    const size_t countBytes    = B * sizeof(uint32_t);
    const size_t acceptOutBytes= B * posCount * sizeof(int32_t);
    const size_t resampleBytes = B * V * sizeof(float);

    void* targetDev  = VulkanBackend::malloc(logitsBytes);
    void* draftDev   = VulkanBackend::malloc(logitsBytes);
    void* uniDev     = VulkanBackend::malloc(uniformBytes);
    void* tokDev     = VulkanBackend::malloc(tokensBytes);
    void* cntDev     = VulkanBackend::malloc(countBytes);
    void* accDev     = VulkanBackend::malloc(acceptOutBytes);
    void* resDev     = VulkanBackend::malloc(resampleBytes);
    if (!targetDev || !draftDev || !uniDev || !tokDev || !cntDev || !accDev || !resDev)
    {
        TRACE_FAIL("Device memory allocation failed");
        for (void* p : {targetDev,draftDev,uniDev,tokDev,cntDev,accDev,resDev})
            if (p) VulkanBackend::free(p);
        return false;
    }

    if (!VulkanBackend::memcpyHostToDevice(targetDev,  hostTarget.data(), logitsBytes) ||
        !VulkanBackend::memcpyHostToDevice(draftDev,   hostDraft.data(),  logitsBytes) ||
        !VulkanBackend::memcpyHostToDevice(uniDev,     hostUniform.data(), uniformBytes) ||
        !VulkanBackend::memcpyHostToDevice(tokDev,     hostDraftTokens.data(), tokensBytes))
    {
        TRACE_FAIL("HostToDevice memcpy failed");
        VulkanBackend::free(targetDev); VulkanBackend::free(draftDev);
        VulkanBackend::free(uniDev); VulkanBackend::free(tokDev); VulkanBackend::free(cntDev);
        VulkanBackend::free(accDev); VulkanBackend::free(resDev);
        return false;
    }

    if (!VulkanBackend::launchSpecDecodeAccept(
            targetDev, draftDev, uniDev, tokDev, cntDev, accDev, resDev,
            B, draftLen, V, temperature, acceptProbFloor))
    {
        TRACE_FAIL(std::string("SpecDecodeAccept launch failed: ") + backend->getLastError());
        VulkanBackend::free(targetDev); VulkanBackend::free(draftDev);
        VulkanBackend::free(uniDev); VulkanBackend::free(tokDev); VulkanBackend::free(cntDev);
        VulkanBackend::free(accDev); VulkanBackend::free(resDev);
        return false;
    }

    std::vector<uint32_t> hostCnt(B, 0u);
    std::vector<int32_t> hostAcc(B * posCount, -123);
    std::vector<float> hostRes(B * V, -999.0f);
    VulkanBackend::memcpyDeviceToHost(hostCnt.data(), cntDev, countBytes);
    VulkanBackend::memcpyDeviceToHost(hostAcc.data(), accDev, acceptOutBytes);
    VulkanBackend::memcpyDeviceToHost(hostRes.data(), resDev, resampleBytes);

    // ---- CPU reference ----
    auto logSoft = [](float const* row, int tok, uint32_t V) {
        float mx = -1e30f;
        for (uint32_t i = 0; i < V; ++i) if (row[i] > mx) mx = row[i];
        float s = 0;
        for (uint32_t i = 0; i < V; ++i) s += expf(row[i] - mx);
        return row[tok] - mx - logf(s);
    };

    bool allCorrect = true;
    for (uint32_t b = 0; b < B && allCorrect; ++b)
    {
        uint32_t accepted = 0u;
        bool rejected = false;
        for (uint32_t t = 0; t < draftLen; ++t)
        {
            if (rejected) break;
            int32_t tok = hostDraftTokens[b * posCount + t];
            float const* tr = &hostTarget[b * posCount * V + t * V];
            float const* dr = &hostDraft[b * posCount * V + t * V];
            float ratio = expf(logSoft(tr, tok, V) - logSoft(dr, tok, V));
            if (ratio > 1.0f) ratio = 1.0f;
            if (hostUniform[b * draftLen + t] < ratio)
            {
                if (hostAcc[b * posCount + accepted] != tok) { allCorrect = false; break; }
                ++accepted;
            }
            else
            {
                rejected = true;
            }
        }

        if (hostCnt[b] != accepted) { allCorrect = false; std::cout << "accCount mismatch b=" << b << std::endl; }

        if (!rejected && accepted == draftLen)
        {
            // bonus token path
            int32_t refBonus = hostDraftTokens[b * posCount + draftLen];
            if (hostAcc[b * posCount + accepted] != refBonus) { allCorrect = false; std::cout << "bonus mismatch b=" << b << std::endl; }
            for (uint32_t i = 0; i < V; ++i) if (hostRes[b * V + i] != 0.0f) { allCorrect = false; std::cout << "resample nonzero b=" << b << std::endl; }
        }
        else
        {
            // resample path: verify distribution sums to ~1 and is non-negative
            float sum = 0;
            for (uint32_t i = 0; i < V; ++i) { if (hostRes[b * V + i] < 0.0f) { allCorrect = false; std::cout << "neg prob b=" << b << std::endl; } sum += hostRes[b * V + i]; }
            if (fabs(sum - 1.0f) > 1e-4f) { allCorrect = false; std::cout << "dist sum=" << sum << " b=" << b << std::endl; }
            float const* tgtRow = &hostTarget[b * posCount * V + accepted * V];
            float mx = -1e30f;
            for (uint32_t i = 0; i < V; ++i) if (tgtRow[i] > mx) mx = tgtRow[i];
            float s = 0;
            for (uint32_t i = 0; i < V; ++i) s += expf(tgtRow[i] - mx);
            // spot check a known index
            // (no exact token check here — caller samples rng)
        }
    }

    VulkanBackend::free(targetDev); VulkanBackend::free(draftDev);
    VulkanBackend::free(uniDev); VulkanBackend::free(tokDev); VulkanBackend::free(cntDev);
    VulkanBackend::free(accDev); VulkanBackend::free(resDev);

    if (!allCorrect)
    {
        TRACE_FAIL("Incorrect spec-decode acceptance results");
    }
    else
    {
        std::cout << "      Result verified: acceptance matches CPU reference" << std::endl;
        TRACE_PASS();
    }

    return allCorrect;
}

bool test_tree_spec_decode()
{
    TRACE_TEST("Tree Spec-Decode Build + Greedy Verify Dispatch");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        TRACE_FAIL("Backend not available");
        return false;
    }

    // ---- Deterministic small problem (mirrors dynamicTreeKernels test vectors) ----
    // Tree: depth=3, topK=2 → parentList has topK*(depth-1)+1 = 5 entries per batch.
    const uint32_t B = 2;
    const uint32_t dT = 4;          // draftTokenNum (tokens incl root)
    const uint32_t topK = 2;
    const uint32_t depth = 3;
    const uint32_t numSpec = 3;     // numSpeculativeTokens
    const uint32_t numInt32PerRow = (dT + 31u) / 32u; // padded to 32-bool rows → 1 uint32

    // parentList [B, 5] int64 → packed as uvec2 {lo,hi}
    // selectedIndex [B, dT-1=3] int64 → packed uvec2
    // We construct a valid tree:
    //   b=0: tokens [0(root),1,2,3], parents: t1←root(0), t2←t1, t3←t1
    //     selectedIndex = [0,1,1]  (parentTbIdx = sel/topK)
    //     parentList = [0, 1, 1, 2, 3]  (parent token id per parentIdx)
    //   b=1: valid tree, token 3 has no parent (invalid → greedy falls through)
    std::vector<int64_t> hostParent(B * (topK * (depth - 1) + 1));
    std::vector<int64_t> hostSel(B * (dT - 1));
    // b=0: root=0; child1 parent=root(0), child2 parent=child1(1), child3 parent=child1(1)
    hostParent[0] = 0; hostParent[1] = 1; hostParent[2] = 1; hostParent[3] = 2; hostParent[4] = 3;
    hostSel[0] = 0; hostSel[1] = 1; hostSel[2] = 1;  // b0
    // b=1: valid chain 0→1→2→3
    hostParent[5] = 0; hostParent[6] = 1; hostParent[7] = 2; hostParent[8] = 2; hostParent[9] = 3;
    hostSel[3] = 0; hostSel[4] = 1; hostSel[5] = 2;  // b1: sel/topK = [0,0,1] → parents 0,0,1

    // Pack int64 → uvec2 host buffer
    const size_t parentUvec2Bytes = B * (topK * (depth - 1) + 1) * 2 * sizeof(uint32_t);
    const size_t selUvec2Bytes    = B * (dT - 1) * 2 * sizeof(uint32_t);
    std::vector<uint32_t> hostParentUvec2(B * (topK * (depth - 1) + 1) * 2);
    std::vector<uint32_t> hostSelUvec2(B * (dT - 1) * 2);
    for (size_t i = 0; i < hostParent.size(); ++i)
    {
        hostParentUvec2[i*2] = uint32_t(hostParent[i] & 0xFFFFFFFF);
        hostParentUvec2[i*2+1] = uint32_t((hostParent[i] >> 32) & 0xFFFFFFFF);
    }
    for (size_t i = 0; i < hostSel.size(); ++i)
    {
        hostSelUvec2[i*2] = uint32_t(hostSel[i] & 0xFFFFFFFF);
        hostSelUvec2[i*2+1] = uint32_t((hostSel[i] >> 32) & 0xFFFFFFFF);
    }

    // Output buffers (host-side references computed after launch)
    const size_t treeBytes = B * dT * numInt32PerRow * sizeof(uint32_t);
    const size_t arrBytes  = B * dT * sizeof(int32_t);
    const size_t treeValidBytes = B * sizeof(int32_t);
    const size_t idxBytes  = B * numSpec * sizeof(int32_t);
    const size_t tokBytes  = B * numSpec * sizeof(int32_t);
    const size_t candBytes = B * dT * sizeof(int32_t);
    const size_t retBytes  = B * dT * 3u * sizeof(int32_t);
    const size_t tgtBytes  = B * dT * sizeof(int32_t);

    void* parentDev   = VulkanBackend::malloc(parentUvec2Bytes);
    void* selDev      = VulkanBackend::malloc(selUvec2Bytes);
    void* treeDev     = VulkanBackend::malloc(treeBytes);
    void* posDev      = VulkanBackend::malloc(arrBytes);
    void* retIdxDev   = VulkanBackend::malloc(arrBytes);
    void* nextTokDev  = VulkanBackend::malloc(arrBytes);
    void* nextSibDev  = VulkanBackend::malloc(arrBytes);
    void* acceptIdxDev = VulkanBackend::malloc(idxBytes);
    void* acceptNumDev = VulkanBackend::malloc(B * sizeof(uint32_t));
    void* acceptTokDev = VulkanBackend::malloc(tokBytes);
    void* candDev     = VulkanBackend::malloc(candBytes);
    void* retPackDev  = VulkanBackend::malloc(retBytes);
    void* tgtDev      = VulkanBackend::malloc(tgtBytes);
    void* validDev    = VulkanBackend::malloc(treeValidBytes);
    if (!parentDev || !selDev || !treeDev || !posDev || !retIdxDev || !nextTokDev || !nextSibDev ||
        !acceptIdxDev || !acceptNumDev || !acceptTokDev || !candDev || !retPackDev || !tgtDev || !validDev)
    {
        TRACE_FAIL("Device memory allocation failed");
        for (void* p : {parentDev,selDev,treeDev,posDev,retIdxDev,nextTokDev,nextSibDev,
                        acceptIdxDev,acceptNumDev,acceptTokDev,candDev,retPackDev,tgtDev,validDev})
            if (p) VulkanBackend::free(p);
        return false;
    }

    // Host candidate token ids and target predictions
    std::vector<int32_t> hostCand(B * dT);
    std::vector<int32_t> hostTgt(B * dT);
    // b0: tokens [0,1,2,3] ; target picks token 2 (matches draft) at pos1, then 3
    hostCand[0]=0; hostCand[1]=1; hostCand[2]=2; hostCand[3]=3;
    hostTgt[0]=0; hostTgt[1]=2; hostTgt[2]=2; hostTgt[3]=3;
    // b1: tokens [0,1,2,3] ; target picks 3 (no draft match) at pos1 → reject
    hostCand[4]=0; hostCand[5]=1; hostCand[6]=2; hostCand[7]=3;
    hostTgt[4]=0; hostTgt[5]=3; hostTgt[6]=3; hostTgt[7]=3;

    std::vector<int32_t> hostValid(B, 1);  // both valid trees

    // Clear output buffers to sentinel
    std::vector<uint32_t> hostTree(B * dT * numInt32PerRow, 0xFFFFFFFFu);
    std::vector<int32_t> hostPos(B * dT, -99);
    std::vector<int32_t> hostRetIdx(B * dT, -99);
    std::vector<int32_t> hostNextTok(B * dT, -99);
    std::vector<int32_t> hostNextSib(B * dT, -99);
    VulkanBackend::memcpyHostToDevice(treeDev, hostTree.data(), treeBytes);
    VulkanBackend::memcpyHostToDevice(posDev, hostPos.data(), arrBytes);
    VulkanBackend::memcpyHostToDevice(retIdxDev, hostRetIdx.data(), arrBytes);
    VulkanBackend::memcpyHostToDevice(nextTokDev, hostNextTok.data(), arrBytes);
    VulkanBackend::memcpyHostToDevice(nextSibDev, hostNextSib.data(), arrBytes);

    // Upload read-only inputs
    VulkanBackend::memcpyHostToDevice(parentDev, hostParentUvec2.data(), parentUvec2Bytes);
    VulkanBackend::memcpyHostToDevice(selDev, hostSelUvec2.data(), selUvec2Bytes);
    VulkanBackend::memcpyHostToDevice(candDev, hostCand.data(), candBytes);
    VulkanBackend::memcpyHostToDevice(tgtDev, hostTgt.data(), tgtBytes);
    VulkanBackend::memcpyHostToDevice(validDev, hostValid.data(), treeValidBytes);

    // ---- Build ----
    if (!VulkanBackend::launchTreeSpecBuild(parentDev, selDev, treeDev, posDev, retIdxDev,
                                             nextTokDev, nextSibDev, B, dT, topK, depth, numInt32PerRow))
    {
        TRACE_FAIL(std::string("TreeSpecBuild launch failed: ") + backend->getLastError());
        for (void* p : {parentDev,selDev,treeDev,posDev,retIdxDev,nextTokDev,nextSibDev,
                        acceptIdxDev,acceptNumDev,acceptTokDev,candDev,retPackDev,tgtDev,validDev})
            VulkanBackend::free(p);
        return false;
    }
    // ---- Retrieve build outputs + pack retrievePacked [B, dT, 3] ----
    std::vector<uint32_t> hostTreeBack(B * dT * numInt32PerRow);
    std::vector<int32_t> hostPosBack(B * dT);
    std::vector<int32_t> hostRetIdxBack(B * dT);
    std::vector<int32_t> hostNextTokBack(B * dT);
    std::vector<int32_t> hostNextSibBack(B * dT);
    VulkanBackend::memcpyDeviceToHost(hostTreeBack.data(), treeDev, treeBytes);
    VulkanBackend::memcpyDeviceToHost(hostPosBack.data(), posDev, arrBytes);
    VulkanBackend::memcpyDeviceToHost(hostRetIdxBack.data(), retIdxDev, arrBytes);
    VulkanBackend::memcpyDeviceToHost(hostNextTokBack.data(), nextTokDev, arrBytes);
    VulkanBackend::memcpyDeviceToHost(hostNextSibBack.data(), nextSibDev, arrBytes);

    // Build retrievePacked host-side from the three arrays
    std::vector<int32_t> hostRetPack(B * dT * 3);
    for (uint32_t b = 0; b < B; ++b)
        for (uint32_t n = 0; n < dT; ++n)
        {
            hostRetPack[(b*dT+n)*3 + 0] = hostRetIdxBack[b*dT+n];
            hostRetPack[(b*dT+n)*3 + 1] = hostNextTokBack[b*dT+n];
            hostRetPack[(b*dT+n)*3 + 2] = hostNextSibBack[b*dT+n];
        }
    VulkanBackend::memcpyHostToDevice(retPackDev, hostRetPack.data(), retBytes);

    // ---- Greedy verify ----
    if (!VulkanBackend::launchTreeSpecGreedyVerify(acceptIdxDev, acceptNumDev, acceptTokDev,
                                                   candDev, retPackDev, tgtDev, validDev,
                                                   B, numSpec, dT))
    {
        TRACE_FAIL(std::string("TreeSpecGreedyVerify launch failed: ") + backend->getLastError());
        for (void* p : {parentDev,selDev,treeDev,posDev,retIdxDev,nextTokDev,nextSibDev,
                        acceptIdxDev,acceptNumDev,acceptTokDev,candDev,retPackDev,tgtDev,validDev})
            VulkanBackend::free(p);
        return false;
    }

    std::vector<int32_t> hostAcceptIdx(B * numSpec, -123);
    std::vector<uint32_t> hostAcceptNum(B, 0u);
    std::vector<int32_t> hostAcceptTok(B * numSpec, -123);
    VulkanBackend::memcpyDeviceToHost(hostAcceptIdx.data(), acceptIdxDev, idxBytes);
    VulkanBackend::memcpyDeviceToHost(hostAcceptNum.data(), acceptNumDev, B * sizeof(uint32_t));
    VulkanBackend::memcpyDeviceToHost(hostAcceptTok.data(), acceptTokDev, tokBytes);

    backend->deviceSynchronize();
    bool allCorrect = true;
    for (uint32_t b = 0; b < B && allCorrect; ++b)
    {
        int32_t const* row = hostRetPack.data() + b * dT * 3;
        if (hostValid[b] == 0)
        {
            if (hostAcceptNum[b] != 0u) allCorrect = false;
            if (hostAcceptIdx[b * numSpec] != 0) allCorrect = false;
            if (hostAcceptTok[b * numSpec] != hostTgt[b * dT]) allCorrect = false;
            continue;
        }

        int32_t lastAccepted = row[0];
        uint32_t nAcc = 0u;
        int curIndex = 0;
        int32_t expTok0 = hostTgt[b * dT + uint32_t(lastAccepted)];
        if (hostAcceptTok[b * numSpec] != expTok0) { allCorrect = false; std::cout << "b" << b << " tok0 mismatch" << std::endl; }
        if (hostAcceptIdx[b * numSpec] != lastAccepted) { allCorrect = false; }

        for (uint32_t j = 1; j < numSpec; ++j)
        {
            curIndex = row[uint32_t(curIndex) * 3 + 1];
            bool matched = false;
            while (curIndex >= 0 && uint32_t(curIndex) < dT)
            {
                int draftLocalIdx = row[uint32_t(curIndex) * 3 + 0];
                int draftTok = hostCand[b * dT + uint32_t(curIndex)];
                int targetTok = hostTgt[b * dT + uint32_t(lastAccepted)];
                if (draftTok == targetTok)
                {
                    ++nAcc;
                    if (hostAcceptIdx[b * numSpec + nAcc] != draftLocalIdx) allCorrect = false;
                    if (hostAcceptTok[b * numSpec + nAcc] != hostTgt[b * dT + uint32_t(draftLocalIdx)]) allCorrect = false;
                    lastAccepted = draftLocalIdx;
                    matched = true;
                    break;
                }
                curIndex = row[uint32_t(curIndex) * 3 + 2];
            }
            if (!matched || curIndex < 0 || uint32_t(curIndex) >= dT) break;
        }
        if (hostAcceptNum[b] != nAcc) { allCorrect = false; std::cout << "b" << b << " nAcc mismatch " << hostAcceptNum[b] << "!=" << nAcc << std::endl; }
    }

    // Sanity-check treeMask bit-setting (root bit must be set for both rows)
    for (uint32_t b = 0; b < B && allCorrect; ++b)
    {
        bool rootBit = (hostTreeBack[b * dT * numInt32PerRow + 0] & 1u) != 0u;
        if (!rootBit) { allCorrect = false; std::cout << "b" << b << " root bit unset" << std::endl; }
    }

    if (!allCorrect)
    {
        TRACE_FAIL("Incorrect tree spec-decode results");
    }
    else
    {
        std::cout << "      Result verified: tree spec build + greedy verify match CPU reference" << std::endl;
        TRACE_PASS();
    }
    return allCorrect;
}

bool test_tree_spec_rejection()
{
    TRACE_TEST("Tree Spec-Decode Rejection Sampler Dispatch");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        TRACE_FAIL("Backend not available");
        return false;
    }

    const uint32_t B = 2;
    const uint32_t dT = 4;
    const uint32_t V = 8;
    const uint32_t J = 3;            // numSpeculativeTokens
    const uint32_t kMaxTried = 32u;
    // One 256-thread WG per batch row.

    // Flat tree under root: all children have sel < topK → parentTbIdx=0 (root).
    // With topK=3, sel values 0,1,2 all give parentTbIdx=0, producing a clean
    // sibling chain root→child1→child2→child3 via links.
    const uint32_t topK = 3;
    const uint32_t depth = 3;
    const uint32_t numInt32PerRow = (dT + 31u) / 32u; // 1 uint32 for dT<=32
    // b0: root(0) → child1(1 parent=root), child2(2 parent=root), child3(3 parent=child1)
    //   selectedIndex = [0, 0, 1]  → parentTbIdx per child = sel/topK = [0,0,0] but
    //   child at pos1 has parent=root, child at pos2 has parent=root, child at pos3 parent=child1.
    // We craft so greedy WOULD accept child1 (matches target), but force rejection path
    // by setting rngSamples high enough to reject all siblings → correction sample.
    std::vector<int64_t> hostParent(B * (topK * (depth - 1) + 1));
    std::vector<int64_t> hostSel(B * (dT - 1));
    // b0: parentList indices into token list
    // parentTokenIdx for child1=0 (root), child2=0(root), child3=1(child1)
    // selectedIndex: child1 sel=0, child2 sel=1, child3 sel=2 → parentTbIdx=sel/topK
    // We want parentTbIdx>0 for non-root children.
    // FLAT tree under root: root(0) with children 1,2,3 (all parentTbIdx=0).
    // This gives a clean sibling chain root→child1→child2→child3 via links.
    // sel values all < topK so parentTbIdx=sel/topK=0 (root) for every child.
    hostParent[0]=0; hostParent[1]=1; hostParent[2]=2; hostParent[3]=0; hostParent[4]=0;
    hostSel[0]=0; hostSel[1]=1; hostSel[2]=2;   // b0: all parentTbIdx=0 (root)
    hostParent[5]=0; hostParent[6]=1; hostParent[7]=2; hostParent[8]=0; hostParent[9]=0;
    hostSel[3]=0; hostSel[4]=1; hostSel[5]=2;  // b1: same

    // draft tokens: [root=0, child1=1, child2=2, child3=3]
    std::vector<int32_t> hostDraftTok(B * dT);
    for (uint32_t b = 0; b < B; ++b)
        for (uint32_t t = 0; t < dT; ++t)
            hostDraftTok[b * dT + t] = int32_t(t);

    // targetProbs [B, dT, V]:
    // b0: root(pos0)=tok0 only; child1 draft=1 gets prob 0 → REJECT → correction
    //     correction from root probs: mass on tok0, scaledCoin=0.95 → winner=0
    // b1: root(pos0)=tok1 only; child1 draft=1 gets prob 1.0 → coin(0.01)≤1.0 → ACCEPT child1
    //     then continue; for simplicity we expect b1 to hit bonus or max-depth.
    std::vector<float> hostTP(B * dT * V, 0.0f);
    for (uint32_t v = 0; v < V; ++v) hostTP[(0 * dT + 0) * V + v] = (v == 0) ? 1.0f : 0.0f;
    for (uint32_t v = 0; v < V; ++v) hostTP[(1 * dT + 0) * V + v] = (v == 1) ? 1.0f : 0.0f;
    for (uint32_t t = 1; t < dT; ++t)
        for (uint32_t v = 0; v < V; ++v)
        {
            hostTP[(0 * dT + t) * V + v] = 1.0f / float(V);
            hostTP[(1 * dT + t) * V + v] = 1.0f / float(V);
        }

    // b0 coin high (reject), b1 coin low (accept)
    std::vector<float> hostRng(B, 0.0f);
    hostRng[0] = 0.95f;
    hostRng[1] = 0.01f;

    // retrieveNextToken / retrieveNextSibling [B, dT] — will be set by build kernel,
    // but rejection sampler reads them directly as input. So we must build the tree
    // first (reuse tree_spec_build), then feed its outputs here.
    // Allocate build-side buffers
    std::vector<uint32_t> hostParentUvec2(B * (topK*(depth-1)+1) * 2);
    std::vector<uint32_t> hostSelUvec2(B * (dT - 1) * 2);
    for (size_t i = 0; i < hostParent.size(); ++i) { hostParentUvec2[i*2]=uint32_t(hostParent[i]&0xFFFFFFFF); hostParentUvec2[i*2+1]=uint32_t((hostParent[i]>>32)&0xFFFFFFFF); }
    for (size_t i = 0; i < hostSel.size(); ++i)    { hostSelUvec2[i*2]=uint32_t(hostSel[i]&0xFFFFFFFF);    hostSelUvec2[i*2+1]=uint32_t((hostSel[i]>>32)&0xFFFFFFFF); }

    std::vector<int32_t> hostTree(B * dT * numInt32PerRow, -1);
    std::vector<int32_t> hostPos(B * dT, -99), hostRetIdx(B * dT, -99),
                         hostNextTok(B * dT, -1), hostNextSib(B * dT, -1);

    size_t parentBytes = hostParentUvec2.size()*sizeof(uint32_t);
    size_t selBytes    = hostSelUvec2.size()*sizeof(uint32_t);
    size_t treeBytes   = hostTree.size()*sizeof(uint32_t);
    size_t arrBytes    = B*dT*sizeof(int32_t);
    size_t tpBytes     = B*dT*V*sizeof(float);
    size_t valBytes    = B*sizeof(uint32_t); // acceptTokenNum uint
    size_t idxBytes    = B*J*2*sizeof(uint32_t); // uvec2
    size_t tokBytes    = idxBytes;
    size_t candBytes   = B*(dT-1)*sizeof(int32_t);
    size_t rngBytes    = B*sizeof(float);

    void* parentDev   = VulkanBackend::malloc(parentBytes);
    void* selDev      = VulkanBackend::malloc(selBytes);
    void* treeDev     = VulkanBackend::malloc(treeBytes);
    void* posDev      = VulkanBackend::malloc(arrBytes);
    void* retIdxDev   = VulkanBackend::malloc(arrBytes);
    void* nextTokDev  = VulkanBackend::malloc(arrBytes);
    void* nextSibDev  = VulkanBackend::malloc(arrBytes);
    void* accIdxDev   = VulkanBackend::malloc(idxBytes);
    void* accNumDev   = VulkanBackend::malloc(valBytes);
    void* accTokDev   = VulkanBackend::malloc(tokBytes);
    void* candDev     = VulkanBackend::malloc(candBytes);
    void* tpDev       = VulkanBackend::malloc(tpBytes);
    void* validDev    = VulkanBackend::malloc(B*sizeof(int32_t));
    void* rngDev      = VulkanBackend::malloc(rngBytes);
    if (!parentDev||!selDev||!treeDev||!posDev||!retIdxDev||!nextTokDev||!nextSibDev||
        !accIdxDev||!accNumDev||!accTokDev||!candDev||!tpDev||!validDev||!rngDev)
    {
        TRACE_FAIL("Allocation failed");
        for (void* p:{parentDev,selDev,treeDev,posDev,retIdxDev,nextTokDev,nextSibDev,
                      accIdxDev,accNumDev,accTokDev,candDev,tpDev,validDev,rngDev})
            if(p) VulkanBackend::free(p);
        return false;
    }

    VulkanBackend::memcpyHostToDevice(parentDev, hostParentUvec2.data(), parentBytes);
    VulkanBackend::memcpyHostToDevice(selDev, hostSelUvec2.data(), selBytes);
    VulkanBackend::memcpyHostToDevice(treeDev, hostTree.data(), treeBytes);
    VulkanBackend::memcpyHostToDevice(posDev, hostPos.data(), arrBytes);
    VulkanBackend::memcpyHostToDevice(retIdxDev, hostRetIdx.data(), arrBytes);
    VulkanBackend::memcpyHostToDevice(nextTokDev, hostNextTok.data(), arrBytes);
    VulkanBackend::memcpyHostToDevice(nextSibDev, hostNextSib.data(), arrBytes);
    // Pack candidate draft tokens per-batch, skipping the root slot (index 0 per batch).
    // The device shader indexes draftTokens[bx*(dT-1) + (childIdx-1)], so we need
    // per-batch contiguous children — a flat hostDraftTok+1 copy misaligns b1's data.
    std::vector<int32_t> hostCandPacked(B * (dT - 1));
    for (uint32_t b = 0; b < B; ++b)
        for (uint32_t t = 1; t < dT; ++t)
            hostCandPacked[b * (dT - 1) + (t - 1)] = hostDraftTok[b * dT + t];
    VulkanBackend::memcpyHostToDevice(candDev, hostCandPacked.data(), candBytes); // skip root slot
    VulkanBackend::memcpyHostToDevice(tpDev, hostTP.data(), tpBytes);
    std::vector<int32_t> hostValid(B,1);
    VulkanBackend::memcpyHostToDevice(validDev, hostValid.data(), B*sizeof(int32_t));
    VulkanBackend::memcpyHostToDevice(rngDev, hostRng.data(), rngBytes);

    // ---- Step 1: build tree ----
    if (!VulkanBackend::launchTreeSpecBuild(parentDev, selDev, treeDev, posDev, retIdxDev,
                                             nextTokDev, nextSibDev, B, dT, topK, depth, numInt32PerRow))
    {
        TRACE_FAIL(std::string("TreeSpecBuild failed: ") + backend->getLastError());
        for (void* p:{parentDev,selDev,treeDev,posDev,retIdxDev,nextTokDev,nextSibDev,
                      accIdxDev,accNumDev,accTokDev,candDev,tpDev,validDev,rngDev})
            VulkanBackend::free(p);
        return false;
    }

    // retrievePacked = interleave(retIdx, nextTok, nextSib) [B, dT, 3]
    std::vector<int32_t> hostRetIdxB(B*dT), hostNextTokB(B*dT), hostNextSibB(B*dT), hostRetPack(B*dT*3);
    VulkanBackend::memcpyDeviceToHost(hostRetIdxB.data(), retIdxDev, arrBytes);
    VulkanBackend::memcpyDeviceToHost(hostNextTokB.data(), nextTokDev, arrBytes);
    VulkanBackend::memcpyDeviceToHost(hostNextSibB.data(), nextSibDev, arrBytes);
    for (uint32_t b=0;b<B;++b) for (uint32_t n=0;n<dT;++n) {
        hostRetPack[(b*dT+n)*3+0]=hostRetIdxB[b*dT+n];
        hostRetPack[(b*dT+n)*3+1]=hostNextTokB[b*dT+n];
        hostRetPack[(b*dT+n)*3+2]=hostNextSibB[b*dT+n];
    }

    // ---- Step 2: rejection sampler ----
    if (!VulkanBackend::launchTreeSpecRejection(accIdxDev, accNumDev, accTokDev,
                                                candDev, tpDev, nextTokDev, nextSibDev,
                                                validDev, rngDev,
                                                B, J, dT, V, kMaxTried))
    {
        TRACE_FAIL(std::string("TreeSpecRejection failed: ") + backend->getLastError());
        for (void* p:{parentDev,selDev,treeDev,posDev,retIdxDev,nextTokDev,nextSibDev,
                      accIdxDev,accNumDev,accTokDev,candDev,tpDev,validDev,rngDev})
            VulkanBackend::free(p);
        return false;
    }

    std::vector<uint32_t> hostNum(B,999u);
    std::vector<uint32_t> hostTokFull(B*J*2, 999u);
    std::vector<uint32_t> hostIdxFull(B*J*2, 999u);
    VulkanBackend::memcpyDeviceToHost(hostNum.data(), accNumDev, valBytes);
    VulkanBackend::memcpyDeviceToHost(hostTokFull.data(), accTokDev, tokBytes);
    VulkanBackend::memcpyDeviceToHost(hostIdxFull.data(), accIdxDev, idxBytes);
    for (void* p:{parentDev,selDev,treeDev,posDev,retIdxDev,nextTokDev,nextSibDev,
                  accIdxDev,accNumDev,accTokDev,candDev,tpDev,validDev,rngDev})
        VulkanBackend::free(p);

    // ---- CPU reference: emulate verifyDynamicTreeRejectionKernel exactly ----
    bool allCorrect = true;
    for (uint32_t b = 0; b < B && allCorrect; ++b)
    {
        int32_t const* row = hostRetPack.data() + b * dT * 3;
        int32_t lastAccepted = row[0];          // retrieveIndex[0]
        uint32_t nAcc = 0u;
        int curIndex = 0;
        int tried[32];
        uint32_t numTried = 0u;

        // treeValid guard
        if (hostValid[b] == 0)
        {
            // sample from root (not exercised in this test since both valid)
            continue;
        }

        for (uint32_t j = 1u; j < J; ++j)
        {
            curIndex = row[uint32_t(curIndex) * 3 + 1];  // first child
            bool matched = false;
            float probAcc = 0.0f;
            float coin = hostRng[b];
            uint32_t tpOffset = (b * dT + uint32_t(lastAccepted)) * V;

            // Walk sibling chain
            while (curIndex >= 0 && uint32_t(curIndex) < dT)
            {
                int draftTok = hostDraftTok[b * dT + uint32_t(curIndex)];
                float tProb = hostTP[tpOffset + uint32_t(draftTok)];
                probAcc += tProb;
                if (coin <= probAcc)
                {
                    nAcc++;
                    // CUDA writes: acceptToken[nAcc_before_inc]=token, nAcc++, acceptIndex[nAcc_after_inc]=childIdx
                    // Verify acceptIndex at position nAcc (post-increment) matches accepted childIdx
                    if (uint32_t(curIndex) < dT * 3)
                    {
                        int gotIdx = hostIdxFull[(b * J + nAcc) * 2];
                        if (gotIdx != curIndex)
                        {
                            allCorrect = false;
                            std::cout << "b=" << b << " acceptIndex mismatch at acc=" << nAcc
                                      << ": got " << gotIdx << " ref " << curIndex << std::endl;
                        }
                    }
                    // accepted this sibling
                    lastAccepted = draftTok;  // matches CUDA: lastAccepted = draftLocalIdx
                    matched = true;
                    break;
                }
                else
                {
                    if (numTried < 32u) tried[numTried++] = draftTok;
                    curIndex = row[uint32_t(curIndex) * 3 + 2];  // next sibling
                }
            }

            if (matched)
            {
                // CPU records acceptance; shader records accTok=tokenId, accIdx=childIdx
                continue;
            }

            // All siblings rejected → correction sample from residual
            if (curIndex < 0 || uint32_t(curIndex) >= dT)
            {
                float residual = 1.0f - probAcc;
                float scaledCoin = coin * residual;
                float cum = 0.0f;
                int refWinner = int(V - 1);
                int lastValid = -1;
                for (uint32_t v = 0u; v < V; ++v)
                {
                    float p = hostTP[tpOffset + v];
                    // zero out tried
                    bool inTried = false;
                    for (uint32_t k = 0u; k < numTried; ++k)
                        if (uint32_t(tried[k]) == v) { inTried = true; break; }
                    if (inTried) p = 0.0f;
                    cum += p;
                    if (p > 0.0f) lastValid = int(v);
                    if (scaledCoin < cum) { refWinner = int(v); break; }
                }
                if (refWinner == int(V - 1) && lastValid >= 0) refWinner = lastValid;
                // Shader emits: acceptToken[nAcc] = sampledToken, then terminal.
                int gotTok = hostTokFull[(b * J + nAcc) * 2];
                if (gotTok != refWinner)
                {
                    allCorrect = false;
                    std::cout << "b=" << b << " correction token mismatch: got " << gotTok
                              << " ref " << refWinner << std::endl;
                }
                // Terminal: no more accepted tokens
                break;
            }
        }

        // If no correction was hit (all siblings accepted to max depth), shader
        // emits bonus token. Verify numAccepted matches.
        if (allCorrect && hostNum[b] != nAcc)
        {
            allCorrect = false;
            std::cout << "b=" << b << " numAcc mismatch: got " << hostNum[b] << " ref " << nAcc << std::endl;
        }
    }

    if (!allCorrect)
    {
        TRACE_FAIL("Incorrect tree spec rejection results");
    }
    else
    {
        std::cout << "      Result verified: tree spec rejection matches CPU reference" << std::endl;
        TRACE_PASS();
    }
    return allCorrect;
}

// ==================== KV Cache Update (2D) ====================
// Vulkan port of updateKVCacheDraftTokenLocationBatchedKernel2D
// (kvCacheUpdateKernels.cu line 140). Compacts accepted draft tokens
// in the KV cache: copies scattered draft positions to contiguous positions.
// Validated against a simple CPU reference that performs the same copy.
bool test_kv_cache_update_2d()
{
    TRACE_TEST("KV Cache Update (2D) Kernel Dispatch");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        TRACE_FAIL("Backend not available");
        return false;
    }

    const uint32_t B = 2;       // seqCount
    const uint32_t H = 2;       // numKVHeads
    const uint32_t L = 1;       // layerCount
    const uint32_t D = 8;       // headDim
    const uint32_t S = 32;      // maxKVCacheLen
    const uint32_t draftLen = 6; // maxDraftLen
    const int32_t rewindCommon = 2;

    const size_t headBytes = static_cast<size_t>(S) * D * sizeof(float);
    const size_t kvBytes = static_cast<size_t>(L) * B * H * S * D * sizeof(float);
    const size_t accBytes = static_cast<size_t>(B) * draftLen * sizeof(int32_t);
    const size_t arrBytes = static_cast<size_t>(B) * sizeof(int32_t);

    // acceptedDraftTokensIndices2D: [B, draftLen], -1 padding
    // b0: accepted tokens at positions [3, 4, 1] (3 accepted, numAccepted=4)
    // b1: accepted tokens at positions [2, 1] (2 accepted, numAccepted=3)
    std::vector<int32_t> hostAccepted(B * draftLen, -1);
    hostAccepted[0 * draftLen + 0] = 3;
    hostAccepted[0 * draftLen + 1] = 4;
    hostAccepted[0 * draftLen + 2] = 1;
    // positions 3,4,5,6 are -1 (padding)
    hostAccepted[1 * draftLen + 0] = 2;
    hostAccepted[1 * draftLen + 1] = 1;

    std::vector<int32_t> hostNumAcc(B);
    hostNumAcc[0] = 4; // 3 accepted draft + 1 bonus = numAcceptedTokens
    hostNumAcc[1] = 3;

    std::vector<int32_t> hostPastKV(B, 20);  // all sequences have 20 KV tokens
    std::vector<int32_t> hostRewind(B, -1);  // not used
    std::vector<int32_t> hostSlotRemap(B, -1); // identity (not used)

    // KV cache: each (seq, head) has S*D floats. Draft tokens are at positions [pastKV, pastKV+draftLen).
    // We fill with distinct values so we can verify the copy.
    std::vector<float> hostK(L * B * H * S * D, 0.0f);
    std::vector<float> hostV(L * B * H * S * D, 0.0f);

    // Fill the KV region with identifiable values.
    // For each (layer, seq, head, pos, channel): value = layer*1000 + seq*100 + head*10 + pos*1 + channel*0.01
    for (uint32_t l = 0; l < L; ++l)
    for (uint32_t s = 0; s < B; ++s)
    for (uint32_t h = 0; h < H; ++h)
    for (uint32_t p = 0; p < S; ++p)
    for (uint32_t c = 0; c < D; ++c)
    {
        float val = float(l * 10000 + s * 1000 + h * 100 + p * 10 + c);
        uint32_t idx = (l * B * H + s * H + h) * S * D + p * D + c;
        hostK[idx] = val;
        hostV[idx] = val + 0.5f;
    }

    // Copy AFTER fill so ref buffers start with the initial (pre-update) state.
    std::vector<float> hostKRef = hostK;
    std::vector<float> hostVRef = hostV;

    // Compute expected result: for each accepted draft token, the source position
    // is pastKV - rewindCommon + acceptedDraftTokensIndices2D[seq, tokenIdx].
    // The target position is pastKV - rewindCommon + tokenIdx.
    // Key: load ALL source data first (shared-memory pattern), then store — prevents
    // read-after-write hazards when source and destination positions overlap.
    for (uint32_t b = 0; b < B; ++b)
    {
        int numAccepted = hostNumAcc[b];
        int seqDraftCount = numAccepted - 1; // exclude bonus token
        if (seqDraftCount <= 0) continue;

        int pastKV = hostPastKV[b];
        int tokenStartIdx = pastKV - rewindCommon;
        if (tokenStartIdx < 0) continue;

        // Load all source data into temporary buffers
        std::vector<float> tmpK(seqDraftCount * H * D);
        std::vector<float> tmpV(seqDraftCount * H * D);
        for (int tokenIdx = 0; tokenIdx < seqDraftCount; ++tokenIdx)
        {
            int srcPos = tokenStartIdx + hostAccepted[b * draftLen + tokenIdx];
            for (uint32_t h = 0; h < H; ++h)
            for (uint32_t c = 0; c < D; ++c)
            {
                uint32_t srcIdx = (b * H + h) * S * D + uint32_t(srcPos) * D + c;
                tmpK[(tokenIdx * H + h) * D + c] = hostK[srcIdx];
                tmpV[(tokenIdx * H + h) * D + c] = hostV[srcIdx];
            }
        }
        // Store from temp to compacted destination positions
        for (int tokenIdx = 0; tokenIdx < seqDraftCount; ++tokenIdx)
        {
            int dstPos = tokenStartIdx + tokenIdx;
            for (uint32_t h = 0; h < H; ++h)
            for (uint32_t c = 0; c < D; ++c)
            {
                uint32_t dstIdx = (b * H + h) * S * D + uint32_t(dstPos) * D + c;
                hostKRef[dstIdx] = tmpK[(tokenIdx * H + h) * D + c];
                hostVRef[dstIdx] = tmpV[(tokenIdx * H + h) * D + c];
            }
        }
    }

    void* kDev = VulkanBackend::malloc(kvBytes);
    void* vDev = VulkanBackend::malloc(kvBytes);
    void* accDev = VulkanBackend::malloc(accBytes);
    void* numAccDev = VulkanBackend::malloc(arrBytes);
    void* pastKVDev = VulkanBackend::malloc(arrBytes);
    void* rewindDev = VulkanBackend::malloc(arrBytes);
    void* slotRemapDev = VulkanBackend::malloc(arrBytes);

    if (!kDev || !vDev || !accDev || !numAccDev || !pastKVDev || !rewindDev || !slotRemapDev)
    {
        TRACE_FAIL("Allocation failed");
        for (void* p : {kDev, vDev, accDev, numAccDev, pastKVDev, rewindDev, slotRemapDev})
            if (p) VulkanBackend::free(p);
        return false;
    }

    VulkanBackend::memcpyHostToDevice(kDev, hostK.data(), kvBytes);
    VulkanBackend::memcpyHostToDevice(vDev, hostV.data(), kvBytes);
    VulkanBackend::memcpyHostToDevice(accDev, hostAccepted.data(), accBytes);
    VulkanBackend::memcpyHostToDevice(numAccDev, hostNumAcc.data(), arrBytes);
    VulkanBackend::memcpyHostToDevice(pastKVDev, hostPastKV.data(), arrBytes);
    VulkanBackend::memcpyHostToDevice(rewindDev, hostRewind.data(), arrBytes);
    VulkanBackend::memcpyHostToDevice(slotRemapDev, hostSlotRemap.data(), arrBytes);

    if (!VulkanBackend::launchKVCacheUpdate2D(kDev, vDev, accDev, numAccDev,
                                              pastKVDev, rewindDev, slotRemapDev,
                                              B, H, S, D, draftLen, rewindCommon, L))
    {
        TRACE_FAIL(std::string("KVCacheUpdate2D failed: ") + backend->getLastError());
        for (void* p : {kDev, vDev, accDev, numAccDev, pastKVDev, rewindDev, slotRemapDev})
            VulkanBackend::free(p);
        return false;
    }

    std::vector<float> hostKResult(hostK.size());
    std::vector<float> hostVResult(hostV.size());
    VulkanBackend::memcpyDeviceToHost(hostKResult.data(), kDev, kvBytes);
    VulkanBackend::memcpyDeviceToHost(hostVResult.data(), vDev, kvBytes);
    for (void* p : {kDev, vDev, accDev, numAccDev, pastKVDev, rewindDev, slotRemapDev})
        VulkanBackend::free(p);

    bool allCorrect = true;
    for (size_t i = 0; i < hostKRef.size(); ++i)
    {
        if (hostKResult[i] != hostKRef[i])
        {
            allCorrect = false;
            std::cout << "KV Cache K mismatch at idx " << i << ": got " << hostKResult[i]
                      << " ref " << hostKRef[i] << std::endl;
            break;
        }
        if (hostVResult[i] != hostVRef[i])
        {
            allCorrect = false;
            std::cout << "KV Cache V mismatch at idx " << i << ": got " << hostVResult[i]
                      << " ref " << hostVRef[i] << std::endl;
            break;
        }
    }

    if (!allCorrect)
    {
        TRACE_FAIL("Incorrect KV cache update results");
    }
    else
    {
        std::cout << "      Result verified: KV cache update matches CPU reference" << std::endl;
        TRACE_PASS();
    }
    return allCorrect;
}

// ==================== MLA FMHA (decode) ====================
// Vulkan port of the CuTe-DSL Blackwell MLA decode kernel. One thread per query
// token; paged KV gather via pageTable, split q_nope/q_rope key, value = latent.
// Validated against a CPU MLA reference with synthetic fp32 tensors.
bool test_mla_fmha()
{
    TRACE_TEST("MLA FMHA Kernel Dispatch");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        TRACE_FAIL("Backend not available");
        return false;
    }

    const uint32_t batchSize  = 2;
    const uint32_t numHeads  = 16;  // DeepSeek-V2-Lite MLA dims: num_attention_heads
    const uint32_t seqQLen   = 1;
    const uint32_t dLatent   = 128; // qk_nope_head_dim
    const uint32_t dRope     = 64;  // qk_rope_head_dim
    const uint32_t D         = dLatent + dRope; // 192
    const uint32_t pageSize  = 64;  // DeepSeek page size
    const uint32_t maxPages  = 2;
    const uint32_t numPages  = 2; // one page per request
    float softmaxScale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<int32_t> hostPageTable = {0, -1, 1, -1};  // b0->page0, b1->page1
    std::vector<int32_t> hostCacheSeqs = {50, 60};         // near-full pages (<= pageSize)

    const size_t qBytes     = static_cast<size_t>(batchSize) * seqQLen * numHeads * D * sizeof(float);
    const size_t kvBytes    = static_cast<size_t>(numPages) * pageSize * D * sizeof(float);
    const size_t pagBytes   = static_cast<size_t>(batchSize) * maxPages * sizeof(int32_t);
    const size_t seqBytes   = static_cast<size_t>(batchSize) * sizeof(int32_t);
    const size_t outBytes   = static_cast<size_t>(batchSize) * seqQLen * numHeads * dLatent * sizeof(float);

    void* qDev      = VulkanBackend::malloc(qBytes);
    void* kvDev     = VulkanBackend::malloc(kvBytes);
    void* ptDev     = VulkanBackend::malloc(pagBytes);
    void* csDev     = VulkanBackend::malloc(seqBytes);
    void* outDev    = VulkanBackend::malloc(outBytes);
    if (!qDev || !kvDev || !ptDev || !csDev || !outDev)
    {
        TRACE_FAIL("Device memory allocation failed");
        if (qDev)  VulkanBackend::free(qDev);
        if (kvDev) VulkanBackend::free(kvDev);
        if (ptDev) VulkanBackend::free(ptDev);
        if (csDev) VulkanBackend::free(csDev);
        if (outDev) VulkanBackend::free(outDev);
        return false;
    }

    std::vector<float> hostQ(qBytes / sizeof(float));
    std::vector<float> hostKv(kvBytes / sizeof(float));
    std::vector<float> hostOut(outBytes / sizeof(float), 0.0f);

    // Real-valued Q/KV derived from DeepSeek-Coder-V2-Lite weights (dequantized from GGUF).
    // If the generated asset 'mla_real.bin' is present, load it (real DeepSeek MLA data);
    // otherwise fall back to the original synthetic RNG stream so the test stays green in
    // environments without the model. The kernel-vs-CPU-reference validity check is
    // identical either way (both consume the same host bytes).
    std::string const realPath = "mla_real.bin";
    bool loadedReal = loadRealMlaData(realPath, hostQ, hostKv);
    if (!loadedReal)
    {
        std::cout << "      [warn] mla_real.bin not found; using synthetic RNG (fallback)."
                  << std::endl;
        std::minstd_rand rng(13);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& val : hostQ)  { val = dist(rng); }
        for (auto& val : hostKv) { val = dist(rng); }
    }

    if (!VulkanBackend::memcpyHostToDevice(qDev,  hostQ.data(),  qBytes) ||
        !VulkanBackend::memcpyHostToDevice(kvDev, hostKv.data(), kvBytes) ||
        !VulkanBackend::memcpyHostToDevice(ptDev, hostPageTable.data(), pagBytes) ||
        !VulkanBackend::memcpyHostToDevice(csDev, hostCacheSeqs.data(), seqBytes))
    {
        TRACE_FAIL("HostToDevice memcpy failed");
        VulkanBackend::free(qDev);
        VulkanBackend::free(kvDev);
        VulkanBackend::free(ptDev);
        VulkanBackend::free(csDev);
        VulkanBackend::free(outDev);
        return false;
    }

    if (!VulkanBackend::launchMlaFmha(qDev, kvDev, ptDev, csDev, outDev,
                                     numHeads, seqQLen, batchSize, dLatent, dRope,
                                     pageSize, maxPages, softmaxScale))
    {
        TRACE_FAIL(std::string("MLA FMHA launch failed: ") + backend->getLastError());
        VulkanBackend::free(qDev);
        VulkanBackend::free(kvDev);
        VulkanBackend::free(ptDev);
        VulkanBackend::free(csDev);
        VulkanBackend::free(outDev);
        return false;
    }

    VulkanBackend::memcpyDeviceToHost(hostOut.data(), outDev, outBytes);

    bool allCorrect = true;

    // CPU reference matching mla_fmha.comp exactly.
    for (uint32_t b = 0; b < batchSize && allCorrect; ++b)
    {
        int32_t kvLen = hostCacheSeqs[b];
        int32_t page0 = hostPageTable[b * maxPages]; // single page per request
        if (kvLen > static_cast<int32_t>(pageSize))
        {
            kvLen = pageSize;
        }
        for (uint32_t h = 0; h < numHeads && allCorrect; ++h)
        {
            uint32_t qOff  = (b * seqQLen * numHeads + h) * D;
            uint32_t oOff  = (b * seqQLen * numHeads + h) * dLatent;

            // gather scores over kvLen tokens
            float maxScore = -1.0e30f;
            std::vector<float> scores(kvLen, 0.0f);
            for (int32_t p = 0; p < kvLen; ++p)
            {
                uint32_t kvBase = (uint32_t(page0) * pageSize + uint32_t(p)) * D;
                float dn = 0.0f, dr = 0.0f;
                for (uint32_t d = 0; d < dLatent; ++d)
                {
                    dn += hostQ[qOff + d] * hostKv[kvBase + d];
                }
                for (uint32_t d = 0; d < dRope; ++d)
                {
                    dr += hostQ[qOff + dLatent + d] * hostKv[kvBase + dLatent + d];
                }
                scores[p] = softmaxScale * (dn + dr);
                if (scores[p] > maxScore)
                {
                    maxScore = scores[p];
                }
            }

            float sum = 0.0f;
            for (int32_t p = 0; p < kvLen; ++p)
            {
                sum += std::exp(scores[p] - maxScore);
            }
            float invSum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

            for (uint32_t d = 0; d < dLatent; ++d)
            {
                float acc = 0.0f;
                for (int32_t p = 0; p < kvLen; ++p)
                {
                    uint32_t kvBase = (uint32_t(page0) * pageSize + uint32_t(p)) * D;
                    float w = std::exp(scores[p] - maxScore) * invSum;
                    acc += w * hostKv[kvBase + d];
                }
                float got = hostOut[oOff + d];
                if (std::abs(got - acc) > 1e-3f)
                {
                    allCorrect = false;
                    std::cout << "      mismatch b=" << b << " h=" << h << " d=" << d
                              << " got=" << got << " ref=" << acc << std::endl;
                }
            }
        }
    }

    VulkanBackend::free(qDev);
    VulkanBackend::free(kvDev);
    VulkanBackend::free(ptDev);
    VulkanBackend::free(csDev);
    VulkanBackend::free(outDev);

    if (!allCorrect)
    {
        TRACE_FAIL("Incorrect MLA FMHA results");
    }
    else
    {
        std::cout << "      Result verified: MLA FMHA matches CPU reference" << std::endl;
        TRACE_PASS();
    }

    return allCorrect;
}

// ==================== MLA FMHA (prefill, causal) ====================
// Context-phase MLA FMHA: S_q>1 query tokens, causal (q[s] attends
// cacheSeqs[b] + s+1 tokens). Mirrors mla_fmha_prefill.comp: one workgroup
// per (b,h,s), paged KV gather, q_nope/q_rope split, value = latent.
bool test_mla_fmha_prefill()
{
    TRACE_TEST("MLA FMHA Prefill Kernel Dispatch");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        TRACE_FAIL("Backend not available");
        return false;
    }

    const uint32_t batchSize  = 2;
    const uint32_t numHeads  = 16;  // DeepSeek-V2-Lite MLA dims: num_attention_heads
    const uint32_t seqQLen   = 16;  // context tokens for this prefill chunk
    const uint32_t dLatent   = 128; // qk_nope_head_dim
    const uint32_t dRope     = 64;  // qk_rope_head_dim
    const uint32_t D         = dLatent + dRope; // 192
    const uint32_t pageSize  = 64;  // DeepSeek page size
    const uint32_t maxPages  = 4;   // b1 now spans page1+page2 (cross-page gather)
    const uint32_t numPages  = 4;   // 4-page pool
    float softmaxScale = 1.0f / std::sqrt(static_cast<float>(D));
    const bool   causal    = true;

    std::vector<int32_t> hostPageTable = {0, -1, -1, -1, 1, 2, -1, -1};  // b0->page0; b1->page1,page2
    std::vector<int32_t> hostCacheSeqs = {0, 80};                        // b0: cold; b1: 80 prior tokens (2 pages)

    const size_t qBytes   = static_cast<size_t>(batchSize) * seqQLen * numHeads * D * sizeof(float);
    const size_t kvBytes  = static_cast<size_t>(numPages) * pageSize * D * sizeof(float);
    const size_t pagBytes = static_cast<size_t>(batchSize) * maxPages * sizeof(int32_t);
    const size_t seqBytes = static_cast<size_t>(batchSize) * sizeof(int32_t);
    const size_t outBytes = static_cast<size_t>(batchSize) * seqQLen * numHeads * dLatent * sizeof(float);

    void* qDev   = VulkanBackend::malloc(qBytes);
    void* kvDev  = VulkanBackend::malloc(kvBytes);
    void* ptDev  = VulkanBackend::malloc(pagBytes);
    void* csDev  = VulkanBackend::malloc(seqBytes);
    void* outDev = VulkanBackend::malloc(outBytes);
    if (!qDev || !kvDev || !ptDev || !csDev || !outDev)
    {
        TRACE_FAIL("Device memory allocation failed");
        if (qDev)  VulkanBackend::free(qDev);
        if (kvDev) VulkanBackend::free(kvDev);
        if (ptDev) VulkanBackend::free(ptDev);
        if (csDev) VulkanBackend::free(csDev);
        if (outDev) VulkanBackend::free(outDev);
        return false;
    }

    std::vector<float> hostQ(qBytes / sizeof(float));
    std::vector<float> hostKv(kvBytes / sizeof(float));
    std::vector<float> hostOut(outBytes / sizeof(float), 0.0f);

    // Real-valued Q/KV derived from DeepSeek-Coder-V2-Lite weights (dequantized from GGUF).
    std::string const realPath = "mla_real_prefill.bin";
    bool loadedReal = loadRealMlaData(realPath, hostQ, hostKv);
    if (!loadedReal)
    {
        std::cout << "      [warn] mla_real_prefill.bin not found; using synthetic RNG (fallback)."
                  << std::endl;
        std::minstd_rand        rngF(7);
        std::uniform_real_distribution<float> distF(-1.0f, 1.0f);
        for (auto& val : hostQ)  { val = distF(rngF); }
        for (auto& val : hostKv) { val = distF(rngF); }
    }

    if (!VulkanBackend::memcpyHostToDevice(qDev,  hostQ.data(),  qBytes) ||
        !VulkanBackend::memcpyHostToDevice(kvDev, hostKv.data(), kvBytes) ||
        !VulkanBackend::memcpyHostToDevice(ptDev, hostPageTable.data(), pagBytes) ||
        !VulkanBackend::memcpyHostToDevice(csDev, hostCacheSeqs.data(), seqBytes))
    {
        TRACE_FAIL("HostToDevice memcpy failed");
        VulkanBackend::free(qDev); VulkanBackend::free(kvDev);
        VulkanBackend::free(ptDev); VulkanBackend::free(csDev);
        VulkanBackend::free(outDev);
        return false;
    }

    if (!VulkanBackend::launchMlaFmhaPrefill(qDev, kvDev, ptDev, csDev, outDev,
                                             numHeads, seqQLen, batchSize, dLatent, dRope,
                                             pageSize, maxPages, causal, softmaxScale))
    {
        TRACE_FAIL(std::string("MLA FMHA prefill launch failed: ") + backend->getLastError());
        VulkanBackend::free(qDev); VulkanBackend::free(kvDev);
        VulkanBackend::free(ptDev); VulkanBackend::free(csDev);
        VulkanBackend::free(outDev);
        return false;
    }

    VulkanBackend::memcpyDeviceToHost(hostOut.data(), outDev, outBytes);

    bool allCorrect = true;

    // CPU reference matching mla_fmha_prefill.comp exactly.
    for (uint32_t b = 0; b < batchSize && allCorrect; ++b)
    {
        int32_t baseKv = hostCacheSeqs[b];
        for (uint32_t s = 0; s < seqQLen && allCorrect; ++s)
        {
            int32_t kvLen = baseKv + static_cast<int32_t>(s) + 1; // causal window
            for (uint32_t h = 0; h < numHeads && allCorrect; ++h)
            {
                uint32_t qOff = (b * seqQLen * numHeads + s * numHeads + h) * D;
                uint32_t oOff = (b * seqQLen * numHeads + s * numHeads + h) * dLatent;

                float maxScore = -1.0e30f;
                std::vector<float> scores(kvLen, 0.0f);
                for (int32_t t = 0; t < kvLen; ++t)
                {
                    uint32_t page    = static_cast<uint32_t>(hostPageTable[b * maxPages + t / pageSize]);
                    uint32_t slot    = static_cast<uint32_t>(t % pageSize);
                    uint32_t kvBase  = (page * pageSize + slot) * D;
                    float dn = 0.0f, dr = 0.0f;
                    for (uint32_t d = 0; d < dLatent; ++d) { dn += hostQ[qOff + d] * hostKv[kvBase + d]; }
                    for (uint32_t d = 0; d < dRope; ++d)  { dr += hostQ[qOff + dLatent + d] * hostKv[kvBase + dLatent + d]; }
                    scores[t] = softmaxScale * (dn + dr);
                    if (scores[t] > maxScore) { maxScore = scores[t]; }
                }

                float sum = 0.0f;
                for (int32_t t = 0; t < kvLen; ++t) { sum += std::exp(scores[t] - maxScore); }
                float invSum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

                for (uint32_t d = 0; d < dLatent; ++d)
                {
                    float acc = 0.0f;
                    for (int32_t t = 0; t < kvLen; ++t)
                    {
                        uint32_t page    = static_cast<uint32_t>(hostPageTable[b * maxPages + t / pageSize]);
                        uint32_t slot    = static_cast<uint32_t>(t % pageSize);
                        uint32_t kvBase  = (page * pageSize + slot) * D;
                        float w = std::exp(scores[t] - maxScore) * invSum;
                        acc += w * hostKv[kvBase + d];
                    }
                    float got = hostOut[oOff + d];
                    if (std::abs(got - acc) > 1e-3f)
                    {
                        allCorrect = false;
                        std::cout << "      mismatch b=" << b << " s=" << s << " h=" << h << " d=" << d
                                  << " got=" << got << " ref=" << acc << std::endl;
                    }
                }
            }
        }
    }

    VulkanBackend::free(qDev);
    VulkanBackend::free(kvDev);
    VulkanBackend::free(ptDev);
    VulkanBackend::free(csDev);
    VulkanBackend::free(outDev);

    if (!allCorrect)
    {
        TRACE_FAIL("Incorrect MLA FMHA prefill results");
    }
    else
    {
        std::cout << "      Result verified: MLA FMHA prefill matches CPU reference" << std::endl;
        TRACE_PASS();
    }

    return allCorrect;
}

// ==================== MLA FMHA (sliding-window variant) ====================
// Exercises the new pc.slidingWindow push-constant path: each query attends only
// to the last W KV tokens (inclusive), clamped to kvLen. Uses real DeepSeek data
// via mla_real.bin when present, else synthetic RNG.
bool test_mla_fmha_sliding_window()
{
    TRACE_TEST("MLA FMHA Sliding-Window Kernel Dispatch");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        TRACE_FAIL("Backend not available");
        return false;
    }

    const uint32_t batchSize     = 2;
    const uint32_t numHeads     = 16;
    const uint32_t seqQLen      = 1;
    const uint32_t dLatent      = 128;
    const uint32_t dRope        = 64;
    const uint32_t D            = dLatent + dRope;
    const uint32_t pageSize     = 64;
    const uint32_t maxPages     = 2;
    const uint32_t numPages     = 2;
    const uint32_t slidingWindow = 32;
    float softmaxScale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<int32_t> hostPageTable = {0, -1, 1, -1};
    std::vector<int32_t> hostCacheSeqs = {50, 60};

    const size_t qBytes  = static_cast<size_t>(batchSize) * seqQLen * numHeads * D * sizeof(float);
    const size_t kvBytes = static_cast<size_t>(numPages) * pageSize * D * sizeof(float);
    const size_t pagBytes= static_cast<size_t>(batchSize) * maxPages * sizeof(int32_t);
    const size_t seqBytes= static_cast<size_t>(batchSize) * sizeof(int32_t);
    const size_t outBytes= static_cast<size_t>(batchSize) * seqQLen * numHeads * dLatent * sizeof(float);

    void* qDev   = VulkanBackend::malloc(qBytes);
    void* kvDev  = VulkanBackend::malloc(kvBytes);
    void* ptDev  = VulkanBackend::malloc(pagBytes);
    void* csDev  = VulkanBackend::malloc(seqBytes);
    void* outDev = VulkanBackend::malloc(outBytes);
    if (!qDev || !kvDev || !ptDev || !csDev || !outDev)
    {
        TRACE_FAIL("Device memory allocation failed");
        if (qDev) VulkanBackend::free(qDev);
        if (kvDev) VulkanBackend::free(kvDev);
        if (ptDev) VulkanBackend::free(ptDev);
        if (csDev) VulkanBackend::free(csDev);
        if (outDev) VulkanBackend::free(outDev);
        return false;
    }

    std::vector<float> hostQ(qBytes / sizeof(float));
    std::vector<float> hostKv(kvBytes / sizeof(float));
    std::vector<float> hostOut(outBytes / sizeof(float), 0.0f);

    std::string const realPath = "mla_real.bin";
    bool loadedReal = loadRealMlaData(realPath, hostQ, hostKv);
    if (!loadedReal)
    {
        std::cout << "      [warn] mla_real.bin not found; using synthetic RNG (fallback)." << std::endl;
        std::minstd_rand rng(13);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& val : hostQ)  { val = dist(rng); }
        for (auto& val : hostKv) { val = dist(rng); }
    }

    if (!VulkanBackend::memcpyHostToDevice(qDev, hostQ.data(), qBytes) ||
        !VulkanBackend::memcpyHostToDevice(kvDev, hostKv.data(), kvBytes) ||
        !VulkanBackend::memcpyHostToDevice(ptDev, hostPageTable.data(), pagBytes) ||
        !VulkanBackend::memcpyHostToDevice(csDev, hostCacheSeqs.data(), seqBytes))
    {
        TRACE_FAIL("HostToDevice memcpy failed");
        VulkanBackend::free(qDev); VulkanBackend::free(kvDev);
        VulkanBackend::free(ptDev); VulkanBackend::free(csDev); VulkanBackend::free(outDev);
        return false;
    }

    if (!VulkanBackend::launchMlaFmha(qDev, kvDev, ptDev, csDev, outDev,
                                       numHeads, seqQLen, batchSize, dLatent, dRope,
                                       pageSize, maxPages, softmaxScale,
                                       slidingWindow, 0u, 1.0f))
    {
        TRACE_FAIL(std::string("MLA FMHA sliding-window launch failed: ") + backend->getLastError());
        VulkanBackend::free(qDev); VulkanBackend::free(kvDev);
        VulkanBackend::free(ptDev); VulkanBackend::free(csDev); VulkanBackend::free(outDev);
        return false;
    }

    VulkanBackend::memcpyDeviceToHost(hostOut.data(), outDev, outBytes);

    bool allCorrect = true;
    for (uint32_t b = 0; b < batchSize && allCorrect; ++b)
    {
        int kvLen = hostCacheSeqs[b];
        if (kvLen > static_cast<int>(pageSize)) kvLen = pageSize;
        uint32_t winStart = (kvLen > static_cast<int>(slidingWindow)) ? uint32_t(kvLen) - slidingWindow : 0u;
        int page0 = hostPageTable[b * maxPages];
        for (uint32_t h = 0; h < numHeads && allCorrect; ++h)
        {
            uint32_t qOff = (b * seqQLen * numHeads + h) * D;
            uint32_t oOff = (b * seqQLen * numHeads + h) * dLatent;
            std::vector<float> scores(kvLen, 0.0f);
            for (int p = 0; p < kvLen; ++p)
            {
                uint32_t kvBase = (uint32_t(page0) * pageSize + uint32_t(p)) * D;
                float dn = 0.0f, dr = 0.0f;
                for (uint32_t d = 0; d < dLatent; ++d) dn += hostQ[qOff + d] * hostKv[kvBase + d];
                for (uint32_t d = 0; d < dRope; ++d)  dr += hostQ[qOff + dLatent + d] * hostKv[kvBase + dLatent + d];
                scores[p] = softmaxScale * (dn + dr);
            }
            float maxScore = -1.0e30f;
            for (int p = 0; p < kvLen; ++p) if (uint32_t(p) >= winStart) maxScore = std::max(maxScore, scores[p]);
            float sum = 0.0f;
            for (int p = 0; p < kvLen; ++p) if (uint32_t(p) >= winStart) sum += std::exp(scores[p] - maxScore);
            float invSum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
            for (uint32_t d = 0; d < dLatent; ++d)
            {
                float acc = 0.0f;
                for (int p = 0; p < kvLen; ++p)
                {
                    if (uint32_t(p) < winStart) continue;
                    uint32_t kvBase = (uint32_t(page0) * pageSize + uint32_t(p)) * D;
                    float w = std::exp(scores[p] - maxScore) * invSum;
                    acc += w * hostKv[kvBase + d];
                }
                float got = hostOut[oOff + d];
                if (std::abs(got - acc) > 1e-3f)
                {
                    allCorrect = false;
                    std::cout << "      [sw] mismatch b=" << b << " h=" << h << " d=" << d
                              << " got=" << got << " ref=" << acc << std::endl;
                }
            }
        }
    }

    VulkanBackend::free(qDev); VulkanBackend::free(kvDev);
    VulkanBackend::free(ptDev); VulkanBackend::free(csDev); VulkanBackend::free(outDev);

    if (!allCorrect) { TRACE_FAIL("Incorrect MLA FMHA sliding-window results"); }
    else { std::cout << "      Result verified: MLA FMHA sliding-window matches CPU reference" << std::endl; TRACE_PASS(); }
    return allCorrect;
}

// ==================== MLA FMHA (fp8/bf16/fp16 storage dtype) ====================
// Exercises pc.storageType + pc.kvScale: Q/KV/output are packed into uint buffers
// (two 16-bit lanes per uint for fp16/bf16; one fp8 byte per uint) and the shader
// dequantizes to fp32 compute. Host reference runs on the original fp32 data, so
// the GPU result is compared against full-precision scores (tolerance absorbs
// fp16/bf16 rounding; fp8 is looser).
bool test_mla_fmha_storage_types()
{
    TRACE_TEST("MLA FMHA Storage-Type (fp8/bf16/fp16) Kernel Dispatch");

    auto backend = VulkanBackend::getInstance();
    if (!backend->isActive() && !backend->initialize(0))
    {
        TRACE_FAIL("Backend not available");
        return false;
    }

    const uint32_t batchSize  = 2;
    const uint32_t numHeads  = 16;
    const uint32_t seqQLen   = 1;
    const uint32_t dLatent   = 128;
    const uint32_t dRope     = 64;
    const uint32_t D         = dLatent + dRope;
    const uint32_t pageSize  = 64;
    const uint32_t maxPages  = 2;
    const uint32_t numPages  = 2;
    float softmaxScale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<int32_t> hostPageTable = {0, -1, 1, -1};
    std::vector<int32_t> hostCacheSeqs = {50, 60};

    std::vector<float> refQ(batchSize * seqQLen * numHeads * D);
    std::vector<float> refKv(numPages * pageSize * D);
    std::minstd_rand rng(21);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : refQ) v = dist(rng);
    for (auto& v : refKv) v = dist(rng);

    const size_t pagBytes = static_cast<size_t>(batchSize) * maxPages * sizeof(int32_t);
    const size_t seqBytes = static_cast<size_t>(batchSize) * sizeof(int32_t);

    bool allCorrect = true;
    const float kTolFp16 = 1e-2f, kTolBf16 = 1e-2f, kTolFp8 = 0.5f;

    for (uint32_t storageType : {1u, 2u, 3u})  // fp16, fp8, bf16
    {
        size_t elemQ   = batchSize * seqQLen * numHeads * D;
        size_t elemKv  = numPages * pageSize * D;
        size_t elemOut = batchSize * seqQLen * numHeads * dLatent;
        // fp8: one uint per element (byte used); fp16/bf16: two lanes per uint.
        size_t qWords  = (storageType == 2u) ? elemQ  : (elemQ + 1u) / 2u;
        size_t kvWords = (storageType == 2u) ? elemKv : (elemKv + 1u) / 2u;
        size_t outWords= (storageType == 2u) ? elemOut : (elemOut + 1u) / 2u;

        std::vector<uint32_t> qHost(qWords, 0);
        std::vector<uint32_t> kvHost(kvWords, 0);

        for (size_t i = 0; i < elemQ; ++i)
        {
            uint32_t packed;
            if (storageType == 1u)      packed = fp32ToFp16Bits(refQ[i]);
            else if (storageType == 3u) packed = fp32ToBf16Bits(refQ[i]);
            else                        packed = fp32ToFp16Bits(refQ[i]) & 0xFFu;  // fp8 via fp16 bits, low byte (test scale)
            if (storageType != 2u)
            {
                uint32_t w = i >> 1u;
                if (i & 1u) qHost[w] = (qHost[w] & 0xFFFFu) | (packed << 16u);
                else        qHost[w] = (qHost[w] & 0xFFFF0000u) | (packed & 0xFFFFu);
            }
            else
            {
                qHost[i] = packed;
            }
        }
        for (size_t i = 0; i < elemKv; ++i)
        {
            uint32_t packed;
            if (storageType == 1u)      packed = fp32ToFp16Bits(refKv[i]);
            else if (storageType == 3u) packed = fp32ToBf16Bits(refKv[i]);
            else                        packed = fp32ToFp16Bits(refKv[i]) & 0xFFu;  // fp8: store dequant-scale-aware
            if (storageType != 2u)
            {
                uint32_t w = i >> 1u;
                if (i & 1u) kvHost[w] = (kvHost[w] & 0xFFFFu) | (packed << 16u);
                else        kvHost[w] = (kvHost[w] & 0xFFFF0000u) | (packed & 0xFFFFu);
            }
            else
            {
                kvHost[i] = packed;
            }
        }

        void* qDev   = VulkanBackend::malloc(qWords * sizeof(uint32_t));
        void* kvDev  = VulkanBackend::malloc(kvWords * sizeof(uint32_t));
        void* ptDev  = VulkanBackend::malloc(pagBytes);
        void* csDev  = VulkanBackend::malloc(seqBytes);
        void* outDev = VulkanBackend::malloc(outWords * sizeof(uint32_t));
        if (!qDev || !kvDev || !ptDev || !csDev || !outDev)
        {
            TRACE_FAIL("Device memory allocation failed");
            if (qDev) VulkanBackend::free(qDev);
            if (kvDev) VulkanBackend::free(kvDev);
            if (ptDev) VulkanBackend::free(ptDev);
            if (csDev) VulkanBackend::free(csDev);
            if (outDev) VulkanBackend::free(outDev);
            return false;
        }

        float kvScale = 1.0f;
        if (storageType == 2u)
        {
            float maxAbs = 0.0f;
            for (auto v : refKv) maxAbs = std::max(maxAbs, std::abs(v));
            kvScale = (maxAbs > 0.0f) ? maxAbs : 1.0f;  // symmetric dequant
        }

        if (!VulkanBackend::memcpyHostToDevice(qDev, qHost.data(), qWords * sizeof(uint32_t)) ||
            !VulkanBackend::memcpyHostToDevice(kvDev, kvHost.data(), kvWords * sizeof(uint32_t)) ||
            !VulkanBackend::memcpyHostToDevice(ptDev, hostPageTable.data(), pagBytes) ||
            !VulkanBackend::memcpyHostToDevice(csDev, hostCacheSeqs.data(), seqBytes))
        {
            TRACE_FAIL("HostToDevice memcpy failed");
            VulkanBackend::free(qDev); VulkanBackend::free(kvDev);
            VulkanBackend::free(ptDev); VulkanBackend::free(csDev); VulkanBackend::free(outDev);
            return false;
        }

        if (!VulkanBackend::launchMlaFmha(qDev, kvDev, ptDev, csDev, outDev,
                                           numHeads, seqQLen, batchSize, dLatent, dRope,
                                           pageSize, maxPages, softmaxScale,
                                           0u, storageType, kvScale))
        {
            TRACE_FAIL(std::string("MLA FMHA storage-type launch failed: ") + backend->getLastError());
            VulkanBackend::free(qDev); VulkanBackend::free(kvDev);
            VulkanBackend::free(ptDev); VulkanBackend::free(csDev); VulkanBackend::free(outDev);
            return false;
        }

        std::vector<uint32_t> outHost(outWords, 0);
        VulkanBackend::memcpyDeviceToHost(outHost.data(), outDev, outWords * sizeof(uint32_t));

        float tol = (storageType == 1u) ? kTolFp16 : (storageType == 3u ? kTolBf16 : kTolFp8);
        const char* tn = (storageType == 1u) ? "fp16" : (storageType == 3u ? "bf16" : "fp8");
        for (uint32_t b = 0; b < batchSize && allCorrect; ++b)
        {
            int kvLen = hostCacheSeqs[b];
            if (kvLen > static_cast<int>(pageSize)) kvLen = pageSize;
            int page0 = hostPageTable[b * maxPages];
            for (uint32_t h = 0; h < numHeads && allCorrect; ++h)
            {
                uint32_t qOff = (b * seqQLen * numHeads + h) * D;
                uint32_t oOff = (b * seqQLen * numHeads + h) * dLatent;
                std::vector<float> scores(kvLen, 0.0f);
                for (int p = 0; p < kvLen; ++p)
                {
                    uint32_t kvBase = (uint32_t(page0) * pageSize + uint32_t(p)) * D;
                    float dn = 0.0f, dr = 0.0f;
                    for (uint32_t d = 0; d < dLatent; ++d) dn += refQ[qOff + d] * refKv[kvBase + d];
                    for (uint32_t d = 0; d < dRope; ++d)  dr += refQ[qOff + dLatent + d] * refKv[kvBase + dLatent + d];
                    scores[p] = softmaxScale * (dn + dr);
                }
                float maxScore = -1.0e30f;
                for (int p = 0; p < kvLen; ++p) maxScore = std::max(maxScore, scores[p]);
                float sum = 0.0f;
                for (int p = 0; p < kvLen; ++p) sum += std::exp(scores[p] - maxScore);
                float invSum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
                for (uint32_t d = 0; d < dLatent; ++d)
                {
                    float acc = 0.0f;
                    for (int p = 0; p < kvLen; ++p)
                    {
                        uint32_t kvBase = (uint32_t(page0) * pageSize + uint32_t(p)) * D;
                        float w = std::exp(scores[p] - maxScore) * invSum;
                        acc += w * refKv[kvBase + d];
                    }
                    uint32_t idx = oOff + d;
                    uint32_t w = outHost[idx >> 1u];
                    uint32_t lane = (idx & 1u) == 0u ? (w & 0xFFFFu) : (w >> 16u);
                    float got;
                    if (storageType == 1u)      got = fp16BitsToFp32(lane);
                    else if (storageType == 3u) got = bf16BitsToFp32(lane);
                    else                         got = fp8BitsToFp32(lane & 0xFFu, kvScale);
                    if (std::abs(got - acc) > tol)
                    {
                        allCorrect = false;
                        std::cout << "      [" << tn << "] mismatch b=" << b << " h=" << h << " d=" << d
                                  << " got=" << got << " ref=" << acc << " tol=" << tol << std::endl;
                    }
                }
            }
        }

        VulkanBackend::free(qDev); VulkanBackend::free(kvDev);
        VulkanBackend::free(ptDev); VulkanBackend::free(csDev); VulkanBackend::free(outDev);
        if (allCorrect)
            std::cout << "      [" << tn << "] verified: MLA FMHA storage-type matches CPU reference" << std::endl;
    }

    if (!allCorrect) { TRACE_FAIL("Incorrect MLA FMHA storage-type results"); }
    else { TRACE_PASS(); }
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
    runTest(test_fp16_gemm, "FP16 GEMM Kernel");
    runTest(test_q8_0_gemm, "Q8_0 GEMM Kernel");
    runTest(test_softmax, "Softmax Kernel");
    runTest(test_attention, "Attention Kernel");
    runTest(test_topk, "Top-K Kernel");
    runTest(test_spec_decode_accept, "Spec-Decode Acceptance Kernel");
    runTest(test_tree_spec_decode, "Tree Spec-Decode Build + Greedy Verify");
    runTest(test_tree_spec_rejection, "Tree Spec-Decode Rejection Sampler");
    runTest(test_kv_cache_update_2d, "KV Cache Update (2D) Kernel");
    runTest(test_mla_fmha, "MLA FMHA Kernel");
    runTest(test_mla_fmha_prefill, "MLA FMHA Prefill Kernel");
    runTest(test_mla_fmha_sliding_window, "MLA FMHA Sliding-Window Kernel");
    runTest(test_mla_fmha_storage_types, "MLA FMHA Storage-Type Kernel");
    runTest(test_resource_leak_prevention, "Resource Leak Prevention");
    runTest(test_utilization_tracking, "GPU Utilization Tracking");

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
