// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorrt_llm/common/safetensors_loader.h"
#include "tensorrt_llm/common/vulkanBackend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace tllm::vk
{

float SafetensorsModelLoader::fp16ToFp32(uint16_t h)
{
    uint32_t sign = (h >> 15u) & 0x1u;
    uint32_t exp  = (h >> 10u) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0u) f = 0u;
    else if (exp == 31u) f = 0x7f800000u | (mant << 13u) | (sign << 31u);
    else f = ((sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u));
    float v;
    std::memcpy(&v, &f, sizeof(v));
    return v;
}

float SafetensorsModelLoader::bf16ToFp32(uint16_t b)
{
    uint32_t bits = (static_cast<uint32_t>(b) & 0xFFFFu) << 16u;
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// ============================================================
// Minimal JSON parser (no external deps)
// ============================================================

// Find a key value in a JSON object string. Returns the start position of the value,
// or std::string::npos if not found.
static size_t findJsonValue(const std::string& json, const std::string& key)
{
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return std::string::npos;
    // Skip past the key and whitespace to find the colon
    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r'))
        pos++;
    if (pos >= json.size() || json[pos] != ':') return std::string::npos;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r'))
        pos++;
    return pos;
}

void SafetensorsModelLoader::parseJson(const std::string& json_str)
{
    m_json = json_str;
    m_data_offset = static_cast<int64_t>(m_json.size()) + 8; // 8 bytes for length

    // Parse each tensor in the JSON
    size_t search_pos = 0;
    while (true)
    {
        // Find next string key followed by : {
        size_t key_start = m_json.find('{', search_pos);
        if (key_start == std::string::npos) break;

        // Find the tensor name (the first string key in this object)
        size_t tensor_name_pos = m_json.find('\"', search_pos);
        if (tensor_name_pos == std::string::npos || tensor_name_pos > key_start)
        {
            search_pos = key_start + 1;
            continue;
        }

        // Extract tensor name
        size_t name_end = m_json.find('\"', tensor_name_pos + 1);
        if (name_end == std::string::npos) break;
        std::string tensor_name = m_json.substr(tensor_name_pos + 1, name_end - tensor_name_pos - 1);

        // Find the object for this tensor
        size_t obj_start = m_json.find('{', name_end);
        if (obj_start == std::string::npos) break;
        size_t obj_end = m_json.find('}', obj_start + 1);
        if (obj_end == std::string::npos) break;

        std::string obj_str = m_json.substr(obj_start, obj_end - obj_start + 1);

        StTensorInfo info;
        info.name = tensor_name;
        info.shape = jsonGetShape(obj_str);
        std::tie(info.offset_begin, info.offset_end) = jsonGetDataOffsets(obj_str);
        info.dtype = jsonGetDtype(obj_str);

        m_tensor_infos.push_back(std::move(info));
        search_pos = obj_end + 1;
    }
}

std::vector<int64_t> SafetensorsModelLoader::jsonGetShape(const std::string& json_obj)
{
    std::vector<int64_t> shape;
    size_t pos = json_obj.find("\"shape\"");
    if (pos == std::string::npos) return shape;

    pos = json_obj.find('[', pos);
    if (pos == std::string::npos) return shape;

    pos++; // skip [
    while (pos < json_obj.size())
    {
        // Skip whitespace
        while (pos < json_obj.size() && (json_obj[pos] == ' ' || json_obj[pos] == ',' || json_obj[pos] == '\t' || json_obj[pos] == '\n' || json_obj[pos] == '\r'))
            pos++;
        if (pos >= json_obj.size() || json_obj[pos] == ']') break;

        // Parse number
        size_t num_start = pos;
        while (pos < json_obj.size() && json_obj[pos] >= '0' && json_obj[pos] <= '9')
            pos++;
        if (num_start < json_obj.size())
        {
            shape.push_back(std::stoll(json_obj.substr(num_start, pos - num_start)));
        }
    }
    return shape;
}

std::pair<int64_t, int64_t> SafetensorsModelLoader::jsonGetDataOffsets(const std::string& json_obj)
{
    size_t pos = json_obj.find("\"data_offsets\"");
    if (pos == std::string::npos) return {0, 0};

    pos = json_obj.find('[', pos);
    if (pos == std::string::npos) return {0, 0};

    pos++; // skip [
    // Parse first number
    while (pos < json_obj.size() && (json_obj[pos] == ' ' || json_obj[pos] == ','))
        pos++;
    size_t first_start = pos;
    while (pos < json_obj.size() && json_obj[pos] >= '0' && json_obj[pos] <= '9')
        pos++;
    int64_t first = std::stoll(json_obj.substr(first_start, pos - first_start));

    // Skip comma
    while (pos < json_obj.size() && (json_obj[pos] == ' ' || json_obj[pos] == ','))
        pos++;
    size_t second_start = pos;
    while (pos < json_obj.size() && json_obj[pos] >= '0' && json_obj[pos] <= '9')
        pos++;
    int64_t second = std::stoll(json_obj.substr(second_start, pos - second_start));

    return {first, second};
}

std::string SafetensorsModelLoader::jsonGetDtype(const std::string& json_obj)
{
    size_t pos = json_obj.find("\"dtype\"");
    if (pos == std::string::npos) return "F32";

    pos = json_obj.find('\"', pos + 7);
    if (pos == std::string::npos) return "F32";
    pos++;
    size_t end = json_obj.find('\"', pos);
    if (end == std::string::npos) return "F32";
    return json_obj.substr(pos, end - pos);
}

std::string SafetensorsModelLoader::jsonGetString(const std::string& json, const std::string& key)
{
    size_t pos = findJsonValue(json, key);
    if (pos == std::string::npos) return "";
    if (pos < json.size() && json[pos] == '\"')
    {
        pos++;
        size_t end = json.find('\"', pos);
        if (end != std::string::npos)
            return json.substr(pos, end - pos);
    }
    return "";
}

int64_t SafetensorsModelLoader::jsonGetInt(const std::string& json, const std::string& key, int64_t default_val)
{
    size_t pos = findJsonValue(json, key);
    if (pos == std::string::npos) return default_val;
    std::string num_str;
    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9')
        num_str += json[pos++];
    if (num_str.empty())
    {
        // Could be negative or have other format
        while (pos < json.size() && (json[pos] == '-' || json[pos] == '+' || (json[pos] >= '0' && json[pos] <= '9')))
            num_str += json[pos++];
    }
    if (num_str.empty()) return default_val;
    return std::stoll(num_str);
}

float SafetensorsModelLoader::jsonGetFloat(const std::string& json, const std::string& key, float default_val)
{
    size_t pos = findJsonValue(json, key);
    if (pos == std::string::npos) return default_val;
    std::string num_str;
    while (pos < json.size() && (json[pos] == '-' || json[pos] == '+' || json[pos] == '.' ||
          (json[pos] >= '0' && json[pos] <= '9') || json[pos] == 'e' || json[pos] == 'E'))
        num_str += json[pos++];
    if (num_str.empty()) return default_val;
    return std::stof(num_str);
}

std::vector<int64_t> SafetensorsModelLoader::jsonGetIntArray(const std::string& json, const std::string& key)
{
    std::vector<int64_t> result;
    size_t pos = findJsonValue(json, key);
    if (pos == std::string::npos) return result;
    if (pos < json.size() && json[pos] == '[')
    {
        pos++;
        while (pos < json.size() && json[pos] != ']')
        {
            if (json[pos] >= '0' && json[pos] <= '9')
            {
                std::string num_str;
                while (pos < json.size() && (json[pos] == '-' || json[pos] == '+' || (json[pos] >= '0' && json[pos] <= '9')))
                    num_str += json[pos++];
                result.push_back(std::stoll(num_str));
            }
            else pos++;
        }
    }
    return result;
}

// ============================================================
// File loading
// ============================================================

bool SafetensorsModelLoader::load(const std::string& safetensors_path)
{
    // Read the 8-byte length prefix
    FILE* f = nullptr;
#ifdef _MSC_VER
    fopen_s(&f, safetensors_path.c_str(), "rb");
#else
    f = std::fopen(safetensors_path.c_str(), "rb");
#endif
    if (!f) return false;

    uint64_t json_len;
    if (std::fread(&json_len, sizeof(uint64_t), 1, f) != 1)
    {
        std::fclose(f);
        return false;
    }

    m_json.resize(json_len);
    if (std::fread(&m_json[0], 1, json_len, f) != json_len)
    {
        std::fclose(f);
        return false;
    }

    std::fclose(f);

    // Compute data offset
    m_data_offset = static_cast<int64_t>(json_len) + 8;

    // Parse JSON for tensor info and metadata
    parseJson(m_json);

    // Extract metadata from config.json if available (same directory)
    m_config_path = safetensors_path;
    size_t last_slash = m_config_path.find_last_of("\\/");
    if (last_slash != std::string::npos)
        m_config_path = m_config_path.substr(0, last_slash + 1) + "config.json";

    // Try to load config.json for model dimensions
    FILE* cf = nullptr;
#ifdef _MSC_VER
    fopen_s(&cf, m_config_path.c_str(), "rb");
#else
    cf = std::fopen(m_config_path.c_str(), "rb");
#endif
    if (cf)
    {
        std::fseek(cf, 0, SEEK_END);
        long cfg_size = std::ftell(cf);
        std::fseek(cf, 0, SEEK_SET);
        std::string cfg_json(cfg_size, '\0');
        std::fread(&cfg_json[0], 1, cfg_size, cf);
        std::fclose(cf);

        m_meta.architecture = jsonGetString(cfg_json, "model_type");
        m_meta.n_layers = static_cast<int32_t>(jsonGetInt(cfg_json, "num_hidden_layers", 0));
        m_meta.hidden_dim = static_cast<int32_t>(jsonGetInt(cfg_json, "hidden_size", 0));
        m_meta.n_heads = static_cast<int32_t>(jsonGetInt(cfg_json, "num_attention_heads", 0));
        m_meta.n_kv_heads = static_cast<int32_t>(jsonGetInt(cfg_json, "num_key_value_heads", 0));
        m_meta.intermediate_dim = static_cast<int32_t>(jsonGetInt(cfg_json, "intermediate_size", 0));
        m_meta.vocab_size = static_cast<int32_t>(jsonGetInt(cfg_json, "vocab_size", 0));
        m_meta.norm_eps = jsonGetFloat(cfg_json, "rms_norm_eps", 1e-6f);
        m_meta.rope_theta = jsonGetFloat(cfg_json, "rope_theta", 10000.0f);
    }

    // Derive head_dim
    if (m_meta.n_heads > 0)
        m_meta.head_dim = m_meta.hidden_dim / m_meta.n_heads;
    if (m_meta.n_kv_heads == 0) m_meta.n_kv_heads = m_meta.n_heads;

    // Check for BOS/EOS token IDs in tokenizer_config.json
    std::string tok_config_path = safetensors_path;
    size_t last_slash2 = tok_config_path.find_last_of("\\/");
    if (last_slash2 != std::string::npos)
        tok_config_path = tok_config_path.substr(0, last_slash2 + 1) + "tokenizer_config.json";

    FILE* tf = nullptr;
#ifdef _MSC_VER
    fopen_s(&tf, tok_config_path.c_str(), "rb");
#else
    tf = std::fopen(tok_config_path.c_str(), "rb");
#endif
    if (tf)
    {
        std::fseek(tf, 0, SEEK_END);
        long tok_size = std::ftell(tf);
        std::fseek(tf, 0, SEEK_SET);
        std::string tok_json(tok_size, '\0');
        std::fread(&tok_json[0], 1, tok_size, tf);
        std::fclose(tf);

        int64_t bos = jsonGetInt(tok_json, "bos_token_id", -1);
        int64_t eos = jsonGetInt(tok_json, "eos_token_id", -1);
        if (bos >= 0) m_meta.bos_token_id = static_cast<int32_t>(bos);
        if (eos >= 0) m_meta.eos_token_id = static_cast<int32_t>(eos);
    }

    // Load tensor data
    if (!loadTensorData(safetensors_path))
    {
        std::fclose(f);
        return false;
    }

    return true;
}

bool SafetensorsModelLoader::loadTensorData(const std::string& filepath)
{
    FILE* f = nullptr;
#ifdef _MSC_VER
    fopen_s(&f, filepath.c_str(), "rb");
#else
    f = std::fopen(filepath.c_str(), "rb");
#endif
    if (!f) return false;

    m_tensors.clear();

    // Stream chunk size — mirrors VAiT VKMODEL_STREAM_CHUNK (64 MiB)
    constexpr size_t ST_CHUNK = 64u << 20;

    for (const auto& info : m_tensor_infos)
    {
        int64_t data_size = info.offset_end - info.offset_begin;
        if (data_size <= 0)
        {
            std::fclose(f);
            return false;
        }

        int64_t n_elements = 1;
        for (auto d : info.shape) n_elements *= d;

        LoadedTensor tensor;
        tensor.name = info.name;
        tensor.shape.assign(info.shape.begin(), info.shape.end());
        tensor.qtype = GGML_TYPE_F32;

        // ---- VAiA: streamed upload raw bytes to VkBuffer ----
        if (m_rt)
        {
            // Allocate device buffer for raw quantized bytes
            VkDeviceSize bytes = static_cast<VkDeviceSize>(data_size);
            tensor.buffer = reinterpret_cast<VkBuffer>(
                VulkanBackend::malloc(bytes));
            if (!tensor.buffer)
            {
                std::fclose(f);
                return false;
            }
            tensor.size = bytes;

            // Streamed upload: seek + fseek/read/chunked memcpy
            if (std::fseek(f, static_cast<long>(info.offset_begin), SEEK_SET) != 0)
            {
                std::fclose(f);
                return false;
            }

            std::vector<uint8_t> chunk(ST_CHUNK);
            VkDeviceSize uploaded = 0;
            int64_t remaining = data_size;
            while (remaining > 0)
            {
                size_t n = remaining < (int64_t)ST_CHUNK ? (size_t)remaining
                                                          : ST_CHUNK;
                if (std::fread(chunk.data(), 1, n, f) != n)
                {
                    std::fclose(f);
                    return false;
                }
                // Compute device-side pointer at current offset
                void* dst = reinterpret_cast<char*>(tensor.buffer) + uploaded;
                VulkanBackend::memcpyHostToDevice(dst, chunk.data(), n);
                uploaded += (VkDeviceSize)n;
                remaining -= (int64_t)n;
            }
        }
        else
        {
            // ---- Legacy host-dequantize path (no Vulkan runtime) ----
            if (std::fseek(f, static_cast<long>(info.offset_begin), SEEK_SET) != 0)
            {
                std::fclose(f);
                return false;
            }

            std::vector<uint8_t> raw_data(static_cast<size_t>(data_size));
            if (std::fread(raw_data.data(), 1, data_size, f) != static_cast<size_t>(data_size))
            {
                std::fclose(f);
                return false;
            }

            // Convert from stored dtype to fp32
            if (info.dtype == "F32")
            {
                const float* src = reinterpret_cast<const float*>(raw_data.data());
                tensor.data.assign(src, src + n_elements);
            }
            else if (info.dtype == "F16")
            {
                const uint16_t* src = reinterpret_cast<const uint16_t*>(raw_data.data());
                tensor.data.resize(n_elements);
                for (int64_t i = 0; i < n_elements; i++)
                    tensor.data[i] = fp16ToFp32(src[i]);
            }
            else if (info.dtype == "BF16")
            {
                const uint16_t* src = reinterpret_cast<const uint16_t*>(raw_data.data());
                tensor.data.resize(n_elements);
                for (int64_t i = 0; i < n_elements; i++)
                    tensor.data[i] = bf16ToFp32(src[i]);
            }
            else
            {
                std::fclose(f);
                throw std::runtime_error("Unsupported safetensors dtype: " + info.dtype);
            }
        }

        m_tensors.push_back(std::move(tensor));
    }

    std::fclose(f);
    return true;
}

const LoadedTensor* SafetensorsModelLoader::getTensor(const std::string& name) const
{
    for (const auto& t : m_tensors)
    {
        if (t.name == name)
            return &t;
    }
    return nullptr;
}

} // namespace tllm::vk