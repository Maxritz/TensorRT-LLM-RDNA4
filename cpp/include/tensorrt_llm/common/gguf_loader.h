// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

#include <vulkan/vulkan.h>

namespace tllm::vk
{

// GGML quantization type IDs (from ggml.h)
static constexpr int32_t GGML_TYPE_F32     = 0;
static constexpr int32_t GGML_TYPE_F16     = 1;
static constexpr int32_t GGML_TYPE_BF16    = 30;
static constexpr int32_t GGML_TYPE_Q4_0    = 2;
static constexpr int32_t GGML_TYPE_Q4_1    = 3;
static constexpr int32_t GGML_TYPE_Q5_0    = 4;
static constexpr int32_t GGML_TYPE_Q5_1    = 5;
static constexpr int32_t GGML_TYPE_Q8_0    = 8;
static constexpr int32_t GGML_TYPE_Q4_K    = 12;
static constexpr int32_t GGML_TYPE_Q5_K    = 13;
static constexpr int32_t GGML_TYPE_Q6_K    = 14;
static constexpr int32_t GGML_TYPE_Q8_1    = 35;

// Block structures (from ggml-common.h)
// QK_K = 256, QK8_0 = 32
static constexpr int QK_K  = 256;
static constexpr int QK8_0 = 32;

// ggml_half = uint16 (16-bit float)
using ggml_half = uint16_t;

// fp16 -> fp32
inline float fp16_to_fp32(ggml_half h)
{
    uint32_t sign = (h >> 15u) & 0x1u;
    uint32_t exp  = (h >> 10u) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    if (exp == 0u && mant == 0u) return sign ? -0.0f : 0.0f;
    if (exp == 0u) {
        float v = static_cast<float>(mant) * (1.0f / 16777216.0f);
        return sign ? -v : v;
    }
    uint32_t f;
    if (exp == 31u) f = 0x7f800000u | (mant << 13u) | (sign << 31u);
    else f = ((sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u));
    float v;
    std::memcpy(&v, &f, sizeof(v));
    return v;
}

struct block_q8_0
{
    ggml_half d;         // scale (fp16)
    int8_t qs[QK8_0];    // 32 quantized values
};

struct block_q6_K
{
    uint8_t ql[QK_K / 2];     // 128: lower 4 bits of quantized values
    uint8_t qh[QK_K / 4];     // 64:  upper 2 bits of quantized values
    int8_t scales[QK_K / 16];  // 16: per-group-16 scales
    ggml_half d;              // 2:  super-block scale
};

struct LoadedTensor
{
    std::string name;
    std::vector<int64_t> shape;   // GGUF storage order (reversed from logical)
    std::vector<float> data;      // dequantized to fp32
    int32_t qtype = GGML_TYPE_F32;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

// Model metadata extracted from GGUF fields
struct GgufMeta
{
    std::string architecture = "unknown";
    int32_t n_layers = 0;
    int32_t hidden_dim = 0;
    int32_t n_heads = 0;
    int32_t n_kv_heads = 0;
    int32_t head_dim = 0;
    int32_t intermediate_dim = 0;
    int32_t vocab_size = 0;
    float norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    bool is_fused_qkv = false;
    int32_t bos_token_id = 151643;
    int32_t eos_token_id = 151643;
    uint64_t alignment = 32;  // tensor data alignment (default 32, from GGUF spec)
};

// GGUF model file loader: parses header, metadata, and dequantizes tensors.
// Dequantization logic is ported from ggml-quants.c (llama.cpp).
class GgufModelLoader
{
public:
    GgufModelLoader() = default;
    ~GgufModelLoader();

    // Load and parse a GGUF model file. Returns true on success.
    bool load(const std::string& gguf_path);

    // Set Vulkan runtime for streaming raw-bytes upload to device.
    void setVulkanRuntime(class VulkanRuntime* rt) { m_rt = rt; }

    // Accessors
    const GgufMeta& getMeta() const { return m_meta; }
    const std::vector<LoadedTensor>& getTensors() const { return m_tensors; }

    // Get tensor by name (returns nullptr if not found)
    const LoadedTensor* getTensor(const std::string& name) const;

    // Dequantize raw GGUF tensor data to float32 (static for reuse)
    static std::vector<float> dequantize(const uint8_t* raw_data, int64_t num_elements, int32_t qtype);

private:
    struct TensorInfo
    {
        std::string name;
        std::vector<uint64_t> shape;
        uint64_t offset = 0;
        uint64_t n_elements = 1;
        int32_t qtype = GGML_TYPE_F32;
    };

    GgufMeta m_meta;
    std::vector<TensorInfo> m_tensor_infos;
    std::vector<LoadedTensor> m_tensors;
    uint64_t m_tensor_data_offset = 0;
    class VulkanRuntime* m_rt = nullptr; // Vulkan runtime for device upload (may be nullptr)

    bool parseMetadata(FILE* f, uint64_t metadata_count);
    bool parseTensors(FILE* f, uint64_t tensor_count, uint32_t version, uint64_t tensor_data_offset);
    void extractMeta();
    bool loadTensorData(FILE* f);
};

} // namespace tllm::vk