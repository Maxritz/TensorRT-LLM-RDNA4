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

#include "sampler_common.h"

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor
{

std::optional<int32_t> unwrap_singleton_i32(const std::optional<std::vector<int32_t>>& p)
{
    if (!p || p->empty())
        return std::nullopt;
    return (*p)[0];
}

std::optional<float> unwrap_singleton_f32(const std::optional<std::vector<float>>& p)
{
    if (!p || p->empty())
        return std::nullopt;
    return (*p)[0];
}

std::optional<bool> unwrap_singleton_bool(const std::optional<std::vector<int32_t>>& p)
{
    if (!p || p->empty())
        return std::nullopt;
    return (*p)[0] != 0;
}

std::optional<int32_t> request_random_seed(const RequestInfo& req)
{
    auto seed = unwrap_singleton_i32(req.sampling_config_random_seed);
    return seed;
}

int32_t _get_beam_width_in(const RequestInfo& req)
{
    if (req.is_context_init_state)
        return 1;
    auto bw = unwrap_singleton_i32(req.sampling_config_beam_width);
    if (bw.has_value() && *bw > 0)
        return *bw;
    return req.py_beam_width > 0 ? req.py_beam_width : 1;
}

int32_t _get_beam_width_out(const RequestInfo& req)
{
    return _get_beam_width_in(req);
}

int32_t _get_max_beam_width(const RequestInfo& req)
{
    int32_t max_beam_width = 1;
    auto bw = unwrap_singleton_i32(req.sampling_config_beam_width);
    if (bw.has_value() && *bw > max_beam_width)
        max_beam_width = *bw;
    if (req.sampling_config_beam_width_array && !req.sampling_config_beam_width_array->empty())
    {
        for (int32_t v : *req.sampling_config_beam_width_array)
        {
            if (v > max_beam_width)
                max_beam_width = v;
        }
    }
    return max_beam_width;
}

UtilsSamplingParams request_get_sampling_params(const RequestInfo& req)
{
    UtilsSamplingParams params;
    params.temperature = unwrap_singleton_f32(req.sampling_config_temperature);
    params.top_p   = unwrap_singleton_f32(req.sampling_config_top_p);
    params.top_k   = unwrap_singleton_i32(req.sampling_config_top_k);
    params.min_p   = unwrap_singleton_f32(req.sampling_config_min_p);
    params.top_p_decay = unwrap_singleton_i32(req.sampling_config_top_p_decay);
    params.top_p_min = unwrap_singleton_f32(req.sampling_config_top_p_min);
    params.top_p_reset_ids = unwrap_singleton_i32(req.sampling_config_top_p_reset_ids);
    params.beam_width_in  = _get_beam_width_in(req);
    params.beam_width_out = _get_beam_width_out(req);
    params.use_beam_search = _get_max_beam_width(req) > 1;
    return params;
}

bool request_sampling_params_cachable(const UtilsSamplingParams& params)
{
    return !params.use_beam_search.value_or(false);
}

} // namespace vulkan_executor
TRTLLM_NAMESPACE_END
