// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorrt_llm/common/gguf_loader.h"
#include "tensorrt_llm/common/vulkanBackend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace tllm::vk
{

// ============================================================
// GGUF type constants (must match ggml-quant.h)
// ============================================================
static const int32_t GGUF_TYPE_UINT8  = 0;
static const int32_t GGUF_TYPE_INT8   = 1;
static const int32_t GGUF_TYPE_UINT16 = 2;
static const int32_t GGUF_TYPE_INT16  = 3;
static const int32_t GGUF_TYPE_UINT32 = 4;
static const int32_t GGUF_TYPE_INT32  = 5;
static const int32_t GGUF_TYPE_FLOAT32 = 6;
static const int32_t GGUF_TYPE_BOOL   = 7;
static const int32_t GGUF_TYPE_STRING = 8;
static const int32_t GGUF_TYPE_ARRAY  = 9;
static const int32_t GGUF_TYPE_UINT64 = 10;
static const int32_t GGUF_TYPE_INT64  = 11;
static const int32_t GGUF_TYPE_FLOAT64 = 12;
static const char* GGUF_MAGIC = "GGUF";

// ============================================================
// Dequantization implementations (ported from ggml-quants.c)
// ============================================================

static void dequantize_row_q8_0(const block_q8_0* x, float* y, int64_t k)
{
    const int nb = k / QK8_0;
    for (int i = 0; i < nb; i++)
    {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; ++j)
        {
            y[i * QK8_0 + j] = static_cast<float>(x[i].qs[j]) * d;
        }
    }
}

static void dequantize_row_q6_K(const block_q6_K* x, float* y, int64_t k)
{
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++)
    {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128)
        {
            for (int l = 0; l < 32; ++l)
            {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0]   = d * sc[is + 0] * q1;
                y[l + 32]  = d * sc[is + 2] * q2;
                y[l + 64]  = d * sc[is + 4] * q3;
                y[l + 96]  = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

std::vector<float> GgufModelLoader::dequantize(const uint8_t* raw_data, int64_t num_elements, int32_t qtype)
{
    std::vector<float> out(num_elements);

    // Debug: print struct sizes
    if (qtype == GGML_TYPE_Q6_K) {
        std::fprintf(stderr, "[DBG] sizeof(block_q6_K)=%zu num_elements=%lld num_blocks=%lld expected_raw=%zu\n",
                  sizeof(block_q6_K), (long long)num_elements, (long long)(num_elements / QK_K),
                  (size_t)(num_elements / QK_K * sizeof(block_q6_K)));
    }

    switch (qtype)
    {
    case GGML_TYPE_F32:
        std::memcpy(out.data(), raw_data, num_elements * sizeof(float));
        break;
    case GGML_TYPE_F16:
    {
        const ggml_half* src = reinterpret_cast<const ggml_half*>(raw_data);
        for (int64_t i = 0; i < num_elements; i++)
            out[i] = fp16_to_fp32(src[i]);
        break;
    }
    case GGML_TYPE_BF16:
    {
        const uint16_t* raw = reinterpret_cast<const uint16_t*>(raw_data);
        for (int64_t i = 0; i < num_elements; i++)
        {
            uint32_t bits = (raw[i] & 0xFFFFu) << 16u;
            float v;
            std::memcpy(&v, &bits, sizeof(v));
            out[i] = v;
        }
        break;
    }
    case GGML_TYPE_Q8_0:
        dequantize_row_q8_0(reinterpret_cast<const block_q8_0*>(raw_data), out.data(), num_elements);
        break;
    case GGML_TYPE_Q6_K:
        dequantize_row_q6_K(reinterpret_cast<const block_q6_K*>(raw_data), out.data(), num_elements);
        break;
    default:
        throw std::runtime_error("Unsupported GGUF quantization type: " + std::to_string(qtype));
    }

    return out;
}

// ============================================================
// GGUF parsing
// ============================================================

bool GgufModelLoader::parseMetadata(FILE* f, uint64_t metadata_count)
{
    for (uint64_t i = 0; i < metadata_count; i++)
    {
        // Read key length (uint64_t in GGUF spec)
        uint64_t key_len;
        if (std::fread(&key_len, sizeof(uint64_t), 1, f) != 1) return false;

        std::string key(key_len, '\0');
        if (key_len > 0 && std::fread(&key[0], 1, key_len, f) != key_len) return false;

        // Read value type
        uint32_t val_type;
        if (std::fread(&val_type, sizeof(uint32_t), 1, f) != 1) return false;

        if (val_type == GGUF_TYPE_STRING)
        {
            uint64_t str_len;
            if (std::fread(&str_len, sizeof(uint64_t), 1, f) != 1) return false;
            if (str_len > 0)
            {
                std::vector<char> buf(str_len);
                if (std::fread(buf.data(), 1, str_len, f) != str_len) return false;
            }
        }
        else if (val_type == GGUF_TYPE_ARRAY)
        {
            uint32_t array_type;
            if (std::fread(&array_type, sizeof(uint32_t), 1, f) != 1) return false;

            uint64_t array_len;
            if (std::fread(&array_len, sizeof(uint64_t), 1, f) != 1) return false;

            // Skip array elements (we extract what we need separately)
            for (uint64_t j = 0; j < array_len; j++)
            {
                if (array_type == GGUF_TYPE_STRING)
                {
                    uint64_t str_len;
                    if (std::fread(&str_len, sizeof(uint64_t), 1, f) != 1) return false;
                    if (str_len > 0)
                    {
                        std::vector<char> buf(str_len);
                        if (std::fread(buf.data(), 1, str_len, f) != str_len) return false;
                    }
                }
                else if (array_type == GGUF_TYPE_BOOL)
                {
                    uint8_t val;
                    if (std::fread(&val, 1, 1, f) != 1) return false;
                }
                else if (array_type == GGUF_TYPE_UINT32 || array_type == GGUF_TYPE_INT32)
                {
                    uint32_t val;
                    if (std::fread(&val, sizeof(uint32_t), 1, f) != 1) return false;
                }
                else if (array_type == GGUF_TYPE_UINT8 || array_type == GGUF_TYPE_INT8)
                {
                    uint8_t val;
                    if (std::fread(&val, 1, 1, f) != 1) return false;
                }
                else if (array_type == GGUF_TYPE_UINT64 || array_type == GGUF_TYPE_INT64)
                {
                    uint64_t val;
                    if (std::fread(&val, sizeof(uint64_t), 1, f) != 1) return false;
                }
                else if (array_type == GGUF_TYPE_FLOAT32)
                {
                    float val;
                    if (std::fread(&val, sizeof(float), 1, f) != 1) return false;
                }
                else if (array_type == GGUF_TYPE_FLOAT64)
                {
                    double val;
                    if (std::fread(&val, sizeof(double), 1, f) != 1) return false;
                }
                else if (array_type == GGUF_TYPE_UINT16 || array_type == GGUF_TYPE_INT16)
                {
                    uint16_t val;
                    if (std::fread(&val, sizeof(uint16_t), 1, f) != 1) return false;
                }
                else
                {
                    return false;
                }
            }
        }
        else if (val_type == GGUF_TYPE_BOOL)
        {
            uint8_t b;
            if (std::fread(&b, 1, 1, f) != 1) return false;
        }
        else if (val_type == GGUF_TYPE_UINT32 || val_type == GGUF_TYPE_INT32)
        {
            uint32_t val;
            if (std::fread(&val, sizeof(uint32_t), 1, f) != 1) return false;
        }
        else if (val_type == GGUF_TYPE_FLOAT32)
        {
            float val;
            if (std::fread(&val, sizeof(float), 1, f) != 1) return false;
        }
        else if (val_type == GGUF_TYPE_UINT8)
        {
            uint8_t val;
            if (std::fread(&val, 1, 1, f) != 1) return false;
        }
         else if (val_type == GGUF_TYPE_INT8)
        {
            int8_t val;
            if (std::fread(&val, 1, 1, f) != 1) return false;
        }
        else if (val_type == GGUF_TYPE_UINT16 || val_type == GGUF_TYPE_INT16)
        {
            uint16_t val;
            if (std::fread(&val, sizeof(uint16_t), 1, f) != 1) return false;
        }
        else if (val_type == GGUF_TYPE_UINT64 || val_type == GGUF_TYPE_INT64)
        {
            uint64_t val;
            if (std::fread(&val, sizeof(uint64_t), 1, f) != 1) return false;
        }
        else if (val_type == GGUF_TYPE_FLOAT64)
        {
            double val;
            if (std::fread(&val, sizeof(double), 1, f) != 1) return false;
        }
        else
        {
            // Unknown type
            return false;
        }
    }

    return true;
}

bool GgufModelLoader::parseTensors(FILE* f, uint64_t tensor_count, uint32_t version, uint64_t tensor_data_offset)
{
    for (uint64_t i = 0; i < tensor_count; i++)
    {
        TensorInfo info;

        // Read name length (uint64_t in GGUF spec)
        uint64_t name_len;
        if (std::fread(&name_len, sizeof(uint64_t), 1, f) != 1) return false;

        // Read name
        info.name.resize(name_len);
        if (name_len > 0 && std::fread(&info.name[0], 1, name_len, f) != name_len) return false;

        // Read shape (array of uint64)
        uint32_t n_dims;
        if (std::fread(&n_dims, sizeof(uint32_t), 1, f) != 1) return false;

        info.shape.resize(n_dims);
        info.n_elements = 1;
        for (uint32_t d = 0; d < n_dims; d++)
        {
            if (std::fread(&info.shape[d], sizeof(uint64_t), 1, f) != 1) return false;
            info.n_elements *= info.shape[d];
        }

        // Read quantization type
        uint32_t qtype;
        if (std::fread(&qtype, sizeof(uint32_t), 1, f) != 1) return false;
        info.qtype = static_cast<int32_t>(qtype);

        // Read tensor offset (uint64, relative to tensor data start)
        uint64_t offset;
        if (std::fread(&offset, sizeof(uint64_t), 1, f) != 1) return false;
        info.offset = offset;

        m_tensor_infos.push_back(info);
    }

    // Compute the tensor data offset = current file position, aligned to alignment
    long pos = std::ftell(f);
    if (m_meta.alignment > 0)
    {
        pos = (pos + m_meta.alignment - 1) & ~(m_meta.alignment - 1);
    }
    m_tensor_data_offset = static_cast<uint64_t>(pos);

    return true;
}

void GgufModelLoader::extractMeta()
{
    // This is called after parseMetadata. We need to re-parse metadata to extract
    // specific fields. For simplicity, we do a second pass here.
    // Actually, we'll parse metadata in load() directly.
}

bool GgufModelLoader::loadTensorData(FILE* f)
{
    // Seek to tensor data start
    if (std::fseek(f, static_cast<long>(m_tensor_data_offset), SEEK_SET) != 0)
        return false;

    m_tensors.clear();

    for (const auto& info : m_tensor_infos)
    {
        if (std::fseek(f, static_cast<long>(m_tensor_data_offset + info.offset), SEEK_SET) != 0)
            return false;

        // Compute byte count of raw data
        size_t type_size = 0;
        switch (info.qtype)
        {
        case GGML_TYPE_F32:  type_size = sizeof(float); break;
        case GGML_TYPE_F16:  type_size = sizeof(ggml_half); break;
        case GGML_TYPE_BF16: type_size = sizeof(uint16_t); break;
        case GGML_TYPE_Q8_0: type_size = sizeof(block_q8_0); break;
        case GGML_TYPE_Q6_K: type_size = sizeof(block_q6_K); break;
        default:
            throw std::runtime_error("Unsupported quant type: " + std::to_string(info.qtype));
        }

        int64_t num_blocks = 1;
        if (info.qtype == GGML_TYPE_Q8_0)
        {
            num_blocks = info.n_elements / QK8_0;
        }
        else if (info.qtype == GGML_TYPE_Q6_K)
        {
            num_blocks = info.n_elements / QK_K;
        }

        size_t raw_bytes = num_blocks * type_size;
        if (info.qtype == GGML_TYPE_F32 || info.qtype == GGML_TYPE_F16 || info.qtype == GGML_TYPE_BF16)
        {
            raw_bytes = info.n_elements * type_size;
        }

        LoadedTensor tensor;
        tensor.name = info.name;
        tensor.shape.assign(info.shape.begin(), info.shape.end());
        tensor.qtype = info.qtype;

        if (m_rt)
        {
            // Streamed upload of raw quantized bytes to device VkBuffer
            constexpr size_t CHUNK = 64u << 20;
            tensor.buffer = reinterpret_cast<VkBuffer>(VulkanBackend::malloc(raw_bytes));
            if (!tensor.buffer)
                return false;
            tensor.size = raw_bytes;

            if (std::fseek(f, static_cast<long>(m_tensor_data_offset + info.offset), SEEK_SET) != 0)
                return false;

            std::vector<uint8_t> chunk(CHUNK);
            size_t uploaded = 0;
            size_t remaining = raw_bytes;
            while (remaining > 0)
            {
                size_t n = remaining < CHUNK ? remaining : CHUNK;
                if (std::fread(chunk.data(), 1, n, f) != n)
                    return false;
                void* dst = reinterpret_cast<char*>(tensor.buffer) + uploaded;
                VulkanBackend::memcpyHostToDevice(dst, chunk.data(), n);
                uploaded += n;
                remaining -= n;
            }
        }
        else
        {
            // Host dequantize path (no Vulkan runtime)
            if (std::fseek(f, static_cast<long>(m_tensor_data_offset + info.offset), SEEK_SET) != 0)
                return false;
            std::vector<uint8_t> raw_data(raw_bytes);
            if (std::fread(raw_data.data(), 1, raw_bytes, f) != raw_bytes)
                return false;
            tensor.data = dequantize(raw_data.data(), info.n_elements, info.qtype);
        }

        m_tensors.push_back(std::move(tensor));
    }

    return true;
}

const LoadedTensor* GgufModelLoader::getTensor(const std::string& name) const
{
    for (const auto& t : m_tensors)
    {
        if (t.name == name)
            return &t;
    }
    return nullptr;
}

GgufModelLoader::~GgufModelLoader()
{
    if (m_rt && m_rt->isInitialized())
    {
        for (auto& t : m_tensors)
        {
            if (t.buffer)
                VulkanBackend::free(t.buffer);
        }
    }
}

bool GgufModelLoader::load(const std::string& gguf_path)
{
    if (!m_rt)
        m_rt = VulkanRuntime::getInstance().get();

    FILE* f = nullptr;
#ifdef _MSC_VER
    fopen_s(&f, gguf_path.c_str(), "rb");
#else
    f = std::fopen(gguf_path.c_str(), "rb");
#endif
    if (!f)
        return false;

    // Read header
    char magic[4];
    if (std::fread(magic, 1, 4, f) != 4)
    {
        std::fclose(f);
        return false;
    }

    if (std::memcmp(magic, GGUF_MAGIC, 4) != 0)
    {
        std::fclose(f);
        return false;
    }

    uint32_t version;
    if (std::fread(&version, sizeof(uint32_t), 1, f) != 1)
    {
        std::fclose(f);
        return false;
    }

    uint64_t tensor_count;
    if (std::fread(&tensor_count, sizeof(uint64_t), 1, f) != 1)
    {
        std::fclose(f);
        return false;
    }

    uint64_t metadata_count;
    if (std::fread(&metadata_count, sizeof(uint64_t), 1, f) != 1)
    {
        std::fclose(f);
        return false;
    }

    uint64_t tensor_data_offset = 0;

    // Parse metadata
    if (!parseMetadata(f, metadata_count))
    {
        std::fclose(f);
        return false;
    }

    // Parse tensor headers
    if (!parseTensors(f, tensor_count, version, tensor_data_offset))
    {
        std::fclose(f);
        return false;
    }

         // Load tensor data
    if (!loadTensorData(f))
    {
        std::fclose(f);
        return false;
    }

    std::fclose(f);

    // Extract metadata
    // We need to re-read metadata to extract specific fields. Let's parse again.
    // For efficiency, we should have stored metadata fields. Let's re-open and parse.
    FILE* f2 = nullptr;
#ifdef _MSC_VER
    fopen_s(&f2, gguf_path.c_str(), "rb");
#else
    f2 = std::fopen(gguf_path.c_str(), "rb");
#endif
    if (!f2)
        return false;

    // Skip header (magic + version + tensor_count + metadata_count = 24 bytes)
    std::fseek(f2, 24, SEEK_SET);

    // Parse metadata again, extracting what we need
    for (uint64_t i = 0; i < metadata_count; i++)
    {
        uint64_t key_len;
        std::fread(&key_len, sizeof(uint64_t), 1, f2);
        std::string key(key_len, '\0');
        std::fread(&key[0], 1, key_len, f2);

        uint32_t val_type;
        std::fread(&val_type, sizeof(uint32_t), 1, f2);

        if (val_type == GGUF_TYPE_STRING)
        {
            uint64_t str_len;
            std::fread(&str_len, sizeof(uint64_t), 1, f2);
            std::string val(str_len, '\0');
            if (str_len > 0) std::fread(&val[0], 1, str_len, f2);

            if (key == "general.architecture")
                m_meta.architecture = val;
            else if (key == "tokenizer.ggml.pre")
                m_meta.architecture = val; // tokenizer pre = model type
        }
        else if (val_type == GGUF_TYPE_ARRAY)
        {
            uint32_t array_type;
            std::fread(&array_type, sizeof(uint32_t), 1, f2);
            uint64_t array_len;
            std::fread(&array_len, sizeof(uint64_t), 1, f2);

            for (uint64_t j = 0; j < array_len; j++)
            {
                if (array_type == GGUF_TYPE_STRING)
                {
                    uint64_t str_len;
                    std::fread(&str_len, sizeof(uint64_t), 1, f2);
                    std::vector<char> buf(str_len);
                    if (str_len > 0) std::fread(buf.data(), 1, str_len, f2);
                }
                else if (array_type == GGUF_TYPE_BOOL)
                {
                    uint8_t val;
                    std::fread(&val, 1, 1, f2);
                }
                else if (array_type == GGUF_TYPE_UINT32 || array_type == GGUF_TYPE_INT32)
                {
                    uint32_t val;
                    std::fread(&val, sizeof(uint32_t), 1, f2);
                }
                else if (array_type == GGUF_TYPE_UINT8 || array_type == GGUF_TYPE_INT8)
                {
                    uint8_t val;
                    std::fread(&val, 1, 1, f2);
                }
                else if (array_type == GGUF_TYPE_UINT64 || array_type == GGUF_TYPE_INT64)
                {
                    uint64_t val;
                    std::fread(&val, sizeof(uint64_t), 1, f2);
                }
                else if (array_type == GGUF_TYPE_FLOAT32)
                {
                    float val;
                    std::fread(&val, sizeof(float), 1, f2);
                }
                else if (array_type == GGUF_TYPE_FLOAT64)
                {
                    double val;
                    std::fread(&val, sizeof(double), 1, f2);
                }
                else if (array_type == GGUF_TYPE_UINT16 || array_type == GGUF_TYPE_INT16)
                {
                    uint16_t val;
                    std::fread(&val, sizeof(uint16_t), 1, f2);
                }
                else
                {
                    // Unknown array element type
                    std::fclose(f2);
                    return false;
                }
            }
        }
        else if (val_type == GGUF_TYPE_BOOL)
        {
            uint8_t b;
            std::fread(&b, 1, 1, f2);
        }
        else if (val_type == GGUF_TYPE_UINT32 || val_type == GGUF_TYPE_INT32)
        {
            uint32_t val;
            std::fread(&val, sizeof(uint32_t), 1, f2);

            // Extract model config values
            std::string prefix = m_meta.architecture;
            if (key == prefix + ".block_count") m_meta.n_layers = static_cast<int32_t>(val);
            else if (key == prefix + ".embedding_length") m_meta.hidden_dim = static_cast<int32_t>(val);
            else if (key == prefix + ".attention.head_count") m_meta.n_heads = static_cast<int32_t>(val);
            else if (key == prefix + ".attention.head_count_kv") m_meta.n_kv_heads = static_cast<int32_t>(val);
            else if (key == prefix + ".feed_forward_length") m_meta.intermediate_dim = static_cast<int32_t>(val);
            else if (key == prefix + ".vocab_size") m_meta.vocab_size = static_cast<int32_t>(val);
            else if (key == "general.alignment") m_meta.alignment = val;
        }
        else if (val_type == GGUF_TYPE_FLOAT32)
        {
            float val;
            std::fread(&val, sizeof(float), 1, f2);

            std::string prefix = m_meta.architecture;
            if (key == (prefix + ".attention.layer_norm_rms_epsilon")) m_meta.norm_eps = val;
            else if (key == (prefix + ".rope.freq_base")) m_meta.rope_theta = val;
        }
        else
        {
            // Skip unknown types
            if (val_type == GGUF_TYPE_UINT8 || val_type == GGUF_TYPE_INT8 || val_type == GGUF_TYPE_BOOL)
            {
                uint8_t v;
                std::fread(&v, 1, 1, f2);
            }
            else if (val_type == GGUF_TYPE_UINT16 || val_type == GGUF_TYPE_INT16)
            {
                uint16_t v;
                std::fread(&v, sizeof(uint16_t), 1, f2);
            }
            else if (val_type == GGUF_TYPE_UINT32 || val_type == GGUF_TYPE_INT32)
            {
                uint32_t val;
                std::fread(&val, sizeof(uint32_t), 1, f2);
            }
            else if (val_type == GGUF_TYPE_UINT64 || val_type == GGUF_TYPE_INT64)
            {
                uint64_t val;
                std::fread(&val, sizeof(uint64_t), 1, f2);
            }
            else if (val_type == GGUF_TYPE_FLOAT64)
            {
                double val;
                std::fread(&val, sizeof(double), 1, f2);
            }
            else
            {
                std::fclose(f2);
                return false;
            }
        }
    }

    std::fclose(f2);

    // Finalize derived fields
    if (m_meta.n_kv_heads == 0) m_meta.n_kv_heads = m_meta.n_heads;
    if (m_meta.head_dim == 0 && m_meta.n_heads > 0)
        m_meta.head_dim = m_meta.hidden_dim / m_meta.n_heads;

    // Check for fused QKV
    for (const auto& t : m_tensors)
    {
        if (t.name.find(".attn_qkv.weight") != std::string::npos)
        {
            m_meta.is_fused_qkv = true;
            break;
        }
    }

    return true;
}

} // namespace tllm::vk