#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "tensorrt_llm/common/vulkanBackend.h"
#include "penalties.h"
#include "top_p_decay.h"

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor
{

namespace sampler
{
class PenaltyHandler;
class FinishReasonsHandler;
class TopPDecayHandler;
class TokenBanHandler;
}

struct SlotInfo
{
    std::vector<int32_t> input_token_ids;
    std::vector<int32_t> output_token_ids;
    int32_t step_idx = 0;
    int32_t generation_length = 0;
    int32_t beam_width = 1;
};

class TorchSampler
{
public:
    TorchSampler(common::VulkanBackend& backend, int32_t max_slots, int32_t vocab_size);
    ~TorchSampler();

    void setup_request(int32_t slot_id, const std::vector<int32_t>& input_token_ids,
                       const sampler::PenaltyConfig& penalty_config,
                       const sampler::TopPDecayConfig& top_p_decay_config,
                       const std::vector<int32_t>& banned_tokens,
                       int32_t max_new_tokens, int32_t end_id);

    void reset_slot(int32_t slot_id);

    int32_t sample_token(int32_t slot_id, float* logits_ptr, int32_t num_beams = 1);

    int32_t sample_token_cpu(std::vector<float>& logits, float top_p, float temperature,
                             int32_t vocab_size, std::mt19937& rng);

    int32_t sample_token_gpu(int32_t slot_id, std::vector<float>& logits,
                             const sampler::PenaltyConfig& penalty_config,
                             float top_p, float temperature, std::mt19937& rng);

    int32_t beam_search_step(int32_t slot_id, float* logits_ptr, int32_t beam_width,
                             int32_t num_return_sequences, int32_t max_new_tokens);

    bool is_finished(int32_t slot_id) const;

    std::string get_finish_reason_string(int32_t slot_id) const;

    const SlotInfo& get_slot_info(int32_t slot_id) const;

    void set_end_id(int32_t slot_id, int32_t end_id);
    int32_t get_end_id(int32_t slot_id) const;

    void set_max_new_tokens(int32_t slot_id, int32_t max_new_tokens);
    int32_t get_max_new_tokens(int32_t slot_id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace vulkan_executor
TRTLLM_NAMESPACE_END
