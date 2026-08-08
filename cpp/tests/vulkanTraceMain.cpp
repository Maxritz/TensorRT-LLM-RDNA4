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
