/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/vk_executor/vk_executor.h"

#include <cstdint>
#include <optional>
#include <vector>

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor
{

inline constexpr int32_t DEFAULT_BEAM_IDX = 0;
inline constexpr int32_t DEFAULT_STEP_IDX = 0;
inline constexpr int32_t max_slots_const = 1024;

struct UtilsSamplingParams
{
    std::optional<float>  temperature{};
    std::optional<float>  top_p{};
    std::optional<int32_t> top_k{};
    std::optional<bool>   use_beam_search{};
    std::optional<float>  min_p{};
    std::optional<int32_t> beam_width_in{};
    std::optional<int32_t> beam_width_out{};
    std::optional<float>  top_p_decay{};
    std::optional<float>  top_p_min{};
    std::optional<int32_t> top_p_reset_ids{};
};

struct RequestInfo
{
    int32_t py_seq_slot{};
    int32_t py_beam_width{1};
    bool    is_context_init_state{false};
    int32_t orig_prompt_len{0};
    int32_t py_max_new_tokens{0};
    std::optional<int32_t> py_end_id{};
    std::vector<int32_t> py_stop_words_list_tokens{};
    std::vector<int32_t> py_stop_words_list_prefix_sum{};
    std::vector<int32_t> py_draft_tokens{};
    bool    py_return_log_probs{false};
    std::optional<int32_t> py_num_logprobs{};
    std::optional<int32_t> py_logprobs_mode{};
    bool    py_logprobs_simple_format{false};
    std::optional<std::vector<int32_t>> sampling_config_random_seed{};
    std::optional<std::vector<float>>   sampling_config_temperature{};
    std::optional<std::vector<float>>   sampling_config_top_p{};
    std::optional<std::vector<int32_t>> sampling_config_top_k{};
    std::optional<std::vector<float>>   sampling_config_min_p{};
    std::optional<std::vector<int32_t>> sampling_config_top_p_decay{};
    std::optional<std::vector<float>>   sampling_config_top_p_min{};
    std::optional<std::vector<int32_t>> sampling_config_top_p_reset_ids{};
    std::optional<std::vector<int32_t>> sampling_config_beam_width{};
    std::optional<std::vector<int32_t>> sampling_config_beam_width_array{};
};

int32_t _get_max_beam_width(const RequestInfo& req);
int32_t _get_beam_width_in(const RequestInfo& req);
int32_t _get_beam_width_out(const RequestInfo& req);
std::optional<int32_t> request_random_seed(const RequestInfo& req);
UtilsSamplingParams request_get_sampling_params(const RequestInfo& req);
bool request_sampling_params_cachable(const UtilsSamplingParams& params);
std::optional<int32_t> unwrap_singleton_i32(const std::optional<std::vector<int32_t>>& p);
std::optional<float> unwrap_singleton_f32(const std::optional<std::vector<float>>& p);
std::optional<bool> unwrap_singleton_bool(const std::optional<std::vector<int32_t>>& p);

static inline float unwrap_or(std::optional<float> p, float fallback)
{
    return p.has_value() ? *p : fallback;
}

static inline int32_t unwrap_or_i32(std::optional<int32_t> p, int32_t fallback)
{
    return p.has_value() ? *p : fallback;
}

} // namespace vulkan_executor
TRTLLM_NAMESPACE_END
