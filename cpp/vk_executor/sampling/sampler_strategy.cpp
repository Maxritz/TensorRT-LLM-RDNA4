#include "sampler_strategy.h"

#include <algorithm>

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor::sampler
{

SamplingStrategy StrategyResolver::resolve_strategy(const UtilsSamplingParams& params, int32_t /*batch_size*/) {
    bool use_beam_search = unwrap_or(*params.use_beam_search, false);
    int32_t beam_width = unwrap_or_i32(*params.beam_width_in, 1);
    if (use_beam_search || beam_width > 1) {
        return SamplingStrategy::BEAM_SEARCH;
    }
    int32_t top_k = unwrap_or_i32(*params.top_k, 1);
    if (top_k > 1) {
        return SamplingStrategy::TOP_K;
    }
    float top_p = unwrap_or(*params.top_p, 1.0f);
    if (top_p < 1.0f) {
        return SamplingStrategy::TOP_P;
    }
    return SamplingStrategy::GREEDY;
}

bool StrategyResolver::is_greedy(const Strategy& strategy) {
    return strategy.type == SamplingStrategy::GREEDY;
}

bool StrategyResolver::is_beam_search(const Strategy& strategy) {
    return strategy.type == SamplingStrategy::BEAM_SEARCH;
}

bool StrategyResolver::is_top_k(const Strategy& strategy) {
    return strategy.type == SamplingStrategy::TOP_K;
}

bool StrategyResolver::is_top_p(const Strategy& strategy) {
    return strategy.type == SamplingStrategy::TOP_P;
}

std::string StrategyResolver::strategy_to_string(SamplingStrategy strategy) {
    switch (strategy) {
        case SamplingStrategy::GREEDY:
            return "GREEDY";
        case SamplingStrategy::TOP_K:
            return "TOP_K";
        case SamplingStrategy::TOP_P:
            return "TOP_P";
        case SamplingStrategy::BEAM_SEARCH:
            return "BEAM_SEARCH";
        case SamplingStrategy::NONE:
            return "NONE";
        default:
            return "UNKNOWN";
    }
}

Strategy StrategyResolver::create_strategy(const UtilsSamplingParams& params, int32_t batch_size) {
    Strategy strategy;
    strategy.type = resolve_strategy(params, batch_size);
    strategy.top_k = unwrap_or_i32(*params.top_k, 1);
    strategy.top_p = unwrap_or(*params.top_p, 1.0f);
    strategy.temperature = unwrap_or(*params.temperature, 1.0f);
    strategy.min_p = unwrap_or(*params.min_p, 0.0f);
    strategy.beam_width = unwrap_or_i32(*params.beam_width_in, 1);
    strategy.output_log_probs = false;
    strategy.return_context = false;
    strategy.return_generation_length = false;
    return strategy;
}

}  // namespace vulkan_executor::sampler
TRTLLM_NAMESPACE_END
