// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tensorrt_llm/common/gguf_loader.h"
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

namespace tllm::vk
{

// Safetensors file header: 8-byte little-endian JSON length, then JSON, then raw data.
// This is a minimal, dependency-free parser for safetensors (no nlohmann/json).
// Supports F32, F16, BF16, I32, I64, BOOL, U8 types.
class SafetensorsModelLoader
{
public:
    SafetensorsModelLoader() = default;
    ~SafetensorsModelLoader() = default;

    // Load and parse a safetensors model file. Returns true on success.
    bool load(const std::string& safetensors_path);

    // Set Vulkan runtime for streaming raw-bytes upload to device.
    void setVulkanRuntime(class VulkanRuntime* rt) { m_rt = rt; }

    // Accessors (same interface as GgufModelLoader)
    const GgufMeta& getMeta() const { return m_meta; }
    const std::vector<LoadedTensor>& getTensors() const { return m_tensors; }
    const LoadedTensor* getTensor(const std::string& name) const;

private:
    struct StTensorInfo
    {
        std::string name;
        std::vector<int64_t> shape;
        int64_t offset_begin = 0;
        int64_t offset_end = 0;
        std::string dtype;
    };

    GgufMeta m_meta;
    std::vector<StTensorInfo> m_tensor_infos;
    std::vector<LoadedTensor> m_tensors;
    int64_t m_data_offset = 0;
    std::string m_json;
    std::string m_config_path;
    class VulkanRuntime* m_rt = nullptr; // Vulkan runtime for device upload (may be nullptr)

    // Parse minimal JSON (no external deps) to extract tensor info and metadata
    void parseJson(const std::string& json_str);
    std::string jsonGetString(const std::string& json, const std::string& key);
    int64_t jsonGetInt(const std::string& json, const std::string& key, int64_t default_val);
    float jsonGetFloat(const std::string& json, const std::string& key, float default_val);
    std::vector<int64_t> jsonGetIntArray(const std::string& json, const std::string& key);
    std::vector<int64_t> jsonGetShape(const std::string& json_obj);
    std::pair<int64_t, int64_t> jsonGetDataOffsets(const std::string& json_obj);
    std::string jsonGetDtype(const std::string& json_obj);

    bool loadTensorData(const std::string& filepath);
    float fp16ToFp32(uint16_t h);
    float bf16ToFp32(uint16_t b);
};

} // namespace tllm::vk