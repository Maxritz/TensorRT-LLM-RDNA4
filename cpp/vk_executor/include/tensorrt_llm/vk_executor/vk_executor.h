/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/vulkanCommon.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// Vulkan C API (matching existing TRT-LLM Vulkan runtime)
#include <vulkan/vulkan.h>

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor
{

// --- Enums mirroring PyExecutor's request lifecycle ---

enum class RequestStatus : uint8_t
{
    kPending = 0,
    kRunning,
    kScheduled,
    kPreparing,
    kExecuting,
    kFinished,
    kCancelled,
    kError
};

enum class TokenType : uint8_t
{
    kInputTokens = 0,
    kOutputTokens
};

// --- Forward declarations ---
struct SamplingConfig;
struct Request;
struct Batch;
struct ResourceMap;
class Scheduler;
class TorchSampler;
class KvCacheManager;
class ModelRunner;

// --- Sampling configuration (mirrors PyExecutor.sampler.SamplingConfig) ---

struct TopKTopPConfig
{
    int32_t top_k{1};
    float   top_p{1.0f};
};

struct BeamSearchConfig
{
    int32_t num_beams{1};
    float   length_penalty{1.0f};
    bool    diverse{false};
    float   diversity_rate{0.0f};
};

enum class PenaltyType : uint8_t
{
    kNone = 0,
    kBrevity,
    kFamiliarity,
    kRepetition
};

struct PenaltyConfig
{
    float frequency_penalty{0.0f};
    float presence_penalty{0.0f};
    float repetition_penalty{1.0f};
    PenaltyType type{PenaltyType::kNone};
};

struct GuidedDecodeConfig
{
    std::string grammar;
    std::optional<std::string> json_schema;
    std::optional<std::string> regex;
};

struct SamplingConfig
{
    TopKTopPConfig             topk_topp;
    BeamSearchConfig           beam;
    PenaltyConfig              penalty;
    GuidedDecodeConfig         guided;
    int32_t                    num_return_sequences{1};
    bool                       output_logits{false};
    std::optional<std::vector<int32_t>> forced_decoder_input_ids;
};

// --- Request structure ---

struct Request
{
    int64_t  request_id;
    int32_t  max_new_tokens;
    int32_t  min_new_tokens{0};
    std::vector<int32_t> input_token_ids;
    SamplingConfig       sampling_config;
    std::optional<std::string> model_name;
    RequestStatus        status{RequestStatus::kPending};
    std::vector<int32_t> output_token_ids;
    std::vector<int32_t> sequence_lengths;
    float    cumulative_logprobs{0.0f};
    bool     streaming{false};
    std::function<void(int32_t token_id)> token_callback;
    std::chrono::steady_clock::time_point enqueue_time;
};

// --- Resource tracking ---

struct GpuResource
{
    uint64_t total_memory_bytes;
    uint64_t free_memory_bytes;
    uint32_t compute_cap_major{0};
    uint32_t compute_cap_minor{0};
    std::string device_name;
};

struct ResourceMap
{
    GpuResource gpu;
    uint64_t vulkan_buffer_memory{0};
    uint64_t vulkan_image_memory{0};
    uint32_t active_requests{0};
    uint32_t queued_requests{0};
};

// --- Forward-pass outputs ---

struct ForwardOutput
{
    std::vector<float>  logits;          // [batch, vocab_size]
    std::vector<uint8_t> kv_cache_update; // packed delta updates
    int32_t             batch_size;
    int32_t             seq_len;
};

// --- Executor callbacks ---

using RequestCallback = std::function<void(const Request&)>;
using BatchCallback   = std::function<void(const Batch&)>;
using ErrorCallback   = std::function<void(int64_t request_id, const std::string& error_msg)>;
using LogCallback     = std::function<void(const std::string& module, int level, const std::string& msg)>;

// --- Batch structure ---

struct Batch
{
    std::vector<Request*> requests;
    int32_t               batch_id;
    int32_t               batch_size{0};
    int32_t               scheduled_length;
    int32_t               new_tokens;
    int32_t               max_input_length;
    int32_t               max_output_length;
    bool                  has_encoder{false};
    bool                  is_speculative{false};
    std::vector<uint32_t> sequence_lengths;
    std::vector<uint32_t> past_key_values;
    ForwardOutput         forward_output;
    RequestStatus         status{RequestStatus::kPending};
};

// --- VkExecutor class ---
// Replaces PyExecutor (py_executor.py) with Vulkan-backed C++17.
// Lifecycle mirrors: PyExecutor.__init__ → _prepare → _execute_iteration
//                     → _cleanup

class VkExecutor
{
public:
    // Factory method (mirrors PyExecutorCreator.create_executor)
    static std::unique_ptr<VkExecutor> Create(
        std::string_view config_path,
        std::string_view engine_path,
        std::vector<std::string> plugin_paths = {},
        std::vector<std::string> adapter_paths = {});

    // --- Lifecycle (mirrors py_executor.py: __init__ / _prepare / _cleanup) ---

    VkExecutor(const VkExecutor&) = delete;
    VkExecutor& operator=(const VkExecutor&) = delete;
    ~VkExecutor();

    // --- Request management (mirrors PyExecutor.enqueue_request) ---

    bool enqueue_request(Request& request);
    void cancel_request(int64_t request_id);
    void cancel_all_requests();

    // --- Execution control (mirrors PyExecutor.run / _execute_iteration) ---

    bool   is_ready() const;
    bool   start();
    void   stop();
    void   pause();
    void   resume();
    int32_t execute_iteration();       // mirrors PyExecutor._execute_iteration
    void   flush();

    // --- Batch operations (mirrors PyExecutor._build_batch / _prepare_batch_inputs) ---

    uint32_t get_batch_size() const;
    uint32_t get_num_running_requests() const;
    void     prepare_batch(std::vector<Request*> requests);
    Batch*   current_batch() const;

    // --- Resource management ---

    ResourceMap get_resource_map() const;
    GpuResource get_gpu_resource() const;
    void        gc();                  // garbage-collect finished requests

    // --- Config / introspection ---

    std::string get_config_str() const;
    std::vector<std::string> get_supported_dtypes() const;
    std::vector<std::string> get_loaded_adapters() const;

    // --- Callbacks ---

    void set_request_callback(RequestCallback cb);
    void set_batch_callback(BatchCallback cb);
    void set_error_callback(ErrorCallback cb);
    void set_log_callback(LogCallback cb);

    // --- Debugging ---

    void set_log_level(std::string_view level);
    std::string get_debug_dump() const;

private:
    VkExecutor();

    friend class ModelRunner;

    struct Impl;
    std::unique_ptr<Impl> pimpl_;

    // --- Sub-component accessors (for unit tests) ---
public:
    const Scheduler&       scheduler() const;
    const TorchSampler&        sampler() const;
    const KvCacheManager&  kv_cache_manager() const;
};

} // namespace vulkan_executor
TRTLLM_NAMESPACE_END