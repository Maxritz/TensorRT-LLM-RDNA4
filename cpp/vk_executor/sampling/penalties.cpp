#include "penalties.h"

#include <algorithm>
#include <cmath>

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor::sampler
{

PenaltyStore::PenaltyStore(int32_t max_slots)
    : m_max_slots(max_slots)
    , m_states(max_slots)
    , m_initialized(max_slots, false) {}

void PenaltyStore::init_slot(int32_t slot_id, const std::vector<int32_t>& input_token_ids,
                              int32_t max_input_length)
{
    if (slot_id < 0 || slot_id >= m_max_slots) {
        return;
    }
    auto& state = m_states[slot_id];
    state.output_token_ids.clear();
    state.presence_counter.assign(max_input_length > 0 ? max_input_length : 1024, 0);
    state.current_generated_length = 0;

    for (int32_t token_id : input_token_ids) {
        if (token_id >= 0 && token_id < static_cast<int32_t>(state.presence_counter.size())) {
            state.presence_counter[token_id]++;
        }
    }
    state.output_token_ids = input_token_ids;
    m_initialized[slot_id] = true;
}

void PenaltyStore::add_token(int32_t slot_id, int32_t token_id)
{
    if (slot_id < 0 || slot_id >= m_max_slots || !m_initialized[slot_id]) {
        return;
    }
    auto& state = m_states[slot_id];
    state.output_token_ids.push_back(token_id);
    state.current_generated_length++;
    if (token_id >= 0 && token_id < static_cast<int32_t>(state.presence_counter.size())) {
        state.presence_counter[token_id]++;
    }
}

void PenaltyStore::reset(int32_t slot_id)
{
    if (slot_id < 0 || slot_id >= m_max_slots) {
        return;
    }
    m_states[slot_id] = PenaltyState{};
    m_initialized[slot_id] = false;
}

const PenaltyState& PenaltyStore::get_state(int32_t slot_id) const
{
    if (slot_id < 0 || slot_id >= m_max_slots) {
        static PenaltyState empty;
        return empty;
    }
    return m_states[slot_id];
}

int32_t PenaltyStore::get_token_count(int32_t slot_id, int32_t token_id) const
{
    if (slot_id < 0 || slot_id >= m_max_slots || !m_initialized[slot_id]) {
        return 0;
    }
    const auto& state = m_states[slot_id];
    if (token_id >= 0 && token_id < static_cast<int32_t>(state.presence_counter.size())) {
        return state.presence_counter[token_id];
    }
    return 0;
}

PenaltyHandler::PenaltyHandler(common::VulkanBackend& backend, int32_t max_slots)
    : m_backend(backend)
    , m_store(max_slots) {}

void PenaltyHandler::apply_penalties(int32_t slot_id, const PenaltyConfig& config,
                                      std::vector<float>& logits,
                                      const std::vector<int32_t>& output_ids,
                                      int32_t vocab_size)
{
    if (config.repetition_penalty_enabled || config.presence_penalty_enabled ||
        config.frequency_penalty_enabled) {
        for (int32_t token_id : output_ids) {
            if (token_id < 0 || token_id >= vocab_size) {
                continue;
            }
            int32_t count = m_store.get_token_count(slot_id, token_id);

            if (config.repetition_penalty_enabled) {
                logits[token_id] = logits[token_id] * config.repetition_penalty;
            }
            if (config.presence_penalty_enabled) {
                logits[token_id] -= config.presence_penalty;
            }
            if (config.frequency_penalty_enabled) {
                logits[token_id] -= config.frequency_penalty * count;
            }
        }
    }
}

void PenaltyHandler::apply_penalties_batch(int32_t batch_size, int32_t slot_offset,
                                            const PenaltyConfig& config,
                                            float* logits_ptr, int32_t vocab_size,
                                            const std::vector<std::vector<int32_t>>& all_output_ids)
{
    for (int32_t i = 0; i < batch_size; ++i) {
        int32_t slot_id = slot_offset + i;
        if (slot_id >= m_store.get_max_slots()) continue;

        const auto& output_ids = (i < static_cast<int32_t>(all_output_ids.size()))
            ? all_output_ids[i] : std::vector<int32_t>{};

        std::vector<float> logits(logits_ptr + i * vocab_size,
                                  logits_ptr + (i + 1) * vocab_size);

        apply_penalties(slot_id, config, logits, output_ids, vocab_size);

        std::copy(logits.begin(), logits.end(),
                  logits_ptr + i * vocab_size);
    }
}

}  // namespace vulkan_executor::sampler
TRTLLM_NAMESPACE_END
