#pragma once

#include <cstdint>
#include <string>

#include "sampler_common.h"

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor
{
namespace sampler
{

enum class SamplingStrategy
{
    GREEDY = 0,
    TOP_K = 1,
    TOP_P = 2,
    BEAM_SEARCH = 3,
    NONE = 4
};

struct Strategy
{
    SamplingStrategy type;
    int32_t top_k = 1;
    float top_p = 1.0f;
    float temperature = 1.0f;
    float min_p = 0.0f;
    int32_t beam_width = 1;
    bool output_log_probs = false;
    bool return_context = false;
    bool return_generation_length = false;
};

class StrategyResolver
{
public:
    static SamplingStrategy resolve_strategy(const UtilsSamplingParams& params, int32_t batch_size);

    static bool is_greedy(const Strategy& strategy);
    static bool is_beam_search(const Strategy& strategy);
    static bool is_top_k(const Strategy& strategy);
    static bool is_top_p(const Strategy& strategy);

    static std::string strategy_to_string(SamplingStrategy strategy);

    static Strategy create_strategy(const UtilsSamplingParams& params, int32_t batch_size);
};

}  // namespace sampler
}  // namespace vulkan_executor
TRTLLM_NAMESPACE_END
