// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorrt_llm/common/vulkan_inference.h"

#include <filesystem>
#include <iostream>
#include <string>

static bool endsWith(const std::string& s, const std::string& suffix)
{
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: vulkan_inference <model_path> <prompt> [max_tokens] [temperature]\n";
        std::cerr << "  model_path can be a .gguf file or a directory with model.safetensors + config.json\n";
        std::cerr << "Example:\n";
        std::cerr << "  vulkan_inference E:/OLLAMA-Models/GGUF/acrux-500m-o1-journey-q6_k.gguf \"Hello world\"\n";
        std::cerr << "  vulkan_inference E:/Models/HF/Qwen2.5-0.5B-Instruct \"Explain quantum computing\"\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt = argv[2];
    int max_tokens = (argc > 3) ? std::atoi(argv[3]) : 128;
    float temperature = (argc > 4) ? static_cast<float>(std::atof(argv[4])) : 0.8f;

    // Determine model type
    bool is_gguf = false;
    std::string resolved_path = model_path;

    if (endsWith(model_path, ".gguf"))
    {
        is_gguf = true;
    }
    else if (std::filesystem::is_directory(model_path))
    {
        // Check for safetensors
        for (const auto& entry : std::filesystem::directory_iterator(model_path))
        {
            if (entry.path().extension() == ".safetensors")
            {
                resolved_path = entry.path().string();
                break;
            }
        }
        if (resolved_path == model_path)
        {
            resolved_path = model_path + "/model.safetensors";
        }
    }
    else if (endsWith(model_path, ".safetensors"))
    {
        // Single safetensors file
    }

    std::cout << "[INFO] Loading model: " << resolved_path << " (gguf=" << is_gguf << ")\n";

    tllm::vk::Qwen2VulkanInference engine;
    if (!engine.loadModel(resolved_path, is_gguf))
    {
        std::cerr << "[ERROR] Failed to load model\n";
        return 1;
    }

    std::cout << "[INFO] Generating...\n";
    auto tokens = engine.generate(prompt, max_tokens, temperature, 50, 0.9f);

    if (tokens.empty())
    {
        std::cerr << "[ERROR] Generation failed\n";
        return 1;
    }

    std::cout << "\n[INFO] Generated " << tokens.size() << " tokens\n";
    return 0;
}
