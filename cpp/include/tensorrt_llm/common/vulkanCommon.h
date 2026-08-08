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

#ifndef VULKAN_COMMON_H
#define VULKAN_COMMON_H

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <mutex>
#include <cmath>
#include <atomic>
#include <set>

#define TRTLLM_NAMESPACE_BEGIN namespace tensorrt_llm {
#define TRTLLM_NAMESPACE_END }

// Logging macros that work without CUDA
#define TLLM_LOG_INFO(...) do { fprintf(stderr, "[INFO] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#define TLLM_LOG_ERROR(...) do { fprintf(stderr, "[ERROR] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#define TLLM_LOG_WARNING(...) do { fprintf(stderr, "[WARN] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#define TLLM_LOG_TRACE(...) do { fprintf(stderr, "[TRACE] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#define TLLM_LOG_DEBUG(...) do { fprintf(stderr, "[DEBUG] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)

// Check macros
#define TLLM_CHECK(x) do { if(!(x)) { TLLM_LOG_ERROR("Check failed: %s at %s:%d", #x, __FILE__, __LINE__); } } while(0)
#define TLLM_CHECK_WITH_INFO(x, info) do { if(!(x)) { TLLM_LOG_ERROR("Check failed: %s (%s) at %s:%d", #x, info, __FILE__, __LINE__); } } while(0)

#endif // VULKAN_COMMON_H
