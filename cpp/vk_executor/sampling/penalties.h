#pragma once

#include <cstdint>
#include <vector>

#include "tensorrt_llm/common/vulkanBackend.h"

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor
{
namespace sampler
{

struct PenaltyConfig
{
    bool repetition_penalty_enabled = false;
    float repetition_penalty = 1.0f;

    bool presence_penalty_enabled = false;
    float presence_penalty = 0.0f;

    bool frequency_penalty_enabled = false;
    float frequency_penalty = 0.0f;

    int32_t max_input_length = 0;
    float temperature = 1.0f;
};

struct PenaltyState
{
    std::vector<int32_t> output_token_ids;
    std::vector<int32_t> presence_counter;
    int32_t current_generated_length = 0;
};

class PenaltyStore
{
public:
    explicit PenaltyStore(int32_t max_slots);

    void init_slot(int32_t slot_id, const std::vector<int32_t>& input_token_ids, int32_t max_input_length);
    void add_token(int32_t slot_id, int32_t token_id);
    void reset(int32_t slot_id);

    const PenaltyState& get_state(int32_t slot_id) const;
    int32_t get_token_count(int32_t slot_id, int32_t token_id) const;
    int32_t get_max_slots() const { return m_max_slots; }

private:
    int32_t m_max_slots;
    std::vector<PenaltyState> m_states;
    std::vector<bool> m_initialized;
};

class PenaltyHandler
{
public:
    PenaltyHandler(common::VulkanBackend& backend, int32_t max_slots);

    void apply_penalties(int32_t slot_id, const PenaltyConfig& config,
                         std::vector<float>& logits,
                         const std::vector<int32_t>& output_ids,
                         int32_t vocab_size);

    void apply_penalties_batch(int32_t batch_size, int32_t slot_offset,
                               const PenaltyConfig& config,
                               float* logits_ptr, int32_t vocab_size,
                               const std::vector<std::vector<int32_t>>& all_output_ids);

    PenaltyStore& get_store() { return m_store; }

private:
    common::VulkanBackend& m_backend;
    PenaltyStore m_store;
};

}  // namespace sampler
}  // namespace vulkan_executor
TRTLLM_NAMESPACE_END
