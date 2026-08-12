#include "top_p_decay.h"

#include <algorithm>

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor::sampler
{

TopPDecayHandler::TopPDecayHandler(int32_t max_slots)
    : m_max_slots(max_slots) {
    m_applied_lens.assign(max_slots, 0);
}

float TopPDecayHandler::compute_top_p(int32_t slot_id, const TopPDecayConfig& config) {
    if (!config.top_p_decay_enabled) {
        return config.top_p;
    }

    if (config.top_p_decay_steps <= 0) {
        return config.top_p;
    }

    int32_t applied_len = (slot_id >= 0 && slot_id < m_max_slots)
        ? m_applied_lens[slot_id] : 0;

    float decay_factor = std::min(1.0f, static_cast<float>(applied_len) / config.top_p_decay_steps);
    float top_p = config.top_p * (1.0f - config.top_p_decay * decay_factor);
    top_p = std::clamp(top_p, 0.01f, 1.0f);

    return top_p;
}

void TopPDecayHandler::update_applied_len(int32_t slot_id, int32_t new_len) {
    if (slot_id >= 0 && slot_id < m_max_slots) {
        m_applied_lens[slot_id] = new_len;
    }
}

void TopPDecayHandler::reset(int32_t slot_id) {
    if (slot_id >= 0 && slot_id < m_max_slots) {
        m_applied_lens[slot_id] = 0;
    }
}

int32_t TopPDecayHandler::get_applied_len(int32_t slot_id) const {
    if (slot_id < 0 || slot_id >= m_max_slots) {
        return 0;
    }
    return m_applied_lens[slot_id];
}

}  // namespace vulkan_executor::sampler
TRTLLM_NAMESPACE_END
