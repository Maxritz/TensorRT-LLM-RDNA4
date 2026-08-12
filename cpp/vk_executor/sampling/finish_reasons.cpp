#include "finish_reasons.h"

#include <algorithm>

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor::sampler
{

FinishReasonsHandler::FinishReasonsHandler(int32_t max_slots)
    : m_max_slots(max_slots) {
    m_finish_reasons.resize(max_slots, FinishReason::FINISH_REASON_GENERATING);
}

void FinishReasonsHandler::reset(int32_t slot_id) {
    if (slot_id >= 0 && slot_id < m_max_slots) {
        m_finish_reasons[slot_id] = FinishReason::FINISH_REASON_GENERATING;
    }
}

bool FinishReasonsHandler::is_finished(int32_t slot_id) const {
    if (slot_id < 0 || slot_id >= m_max_slots) {
        return false;
    }
    return m_finish_reasons[slot_id] != FinishReason::FINISH_REASON_GENERATING;
}

FinishReason FinishReasonsHandler::get_finish_reason(int32_t slot_id) const {
    if (slot_id < 0 || slot_id >= m_max_slots) {
        return FinishReason::FINISH_REASON_GENERATING;
    }
    return m_finish_reasons[slot_id];
}

void FinishReasonsHandler::set_finish_reason(int32_t slot_id, FinishReason reason) {
    if (slot_id >= 0 && slot_id < m_max_slots) {
        m_finish_reasons[slot_id] = reason;
    }
}

bool FinishReasonsHandler::check_end_of_sequence(int32_t slot_id, int32_t last_token_id,
                                                  int32_t end_id, int32_t generated_length,
                                                  int32_t max_new_tokens) {
    if (last_token_id == end_id) {
        set_finish_reason(slot_id, FinishReason::FINISH_REASON_END_OF_SEQUENCE);
        return true;
    }
    if (generated_length >= max_new_tokens) {
        set_finish_reason(slot_id, FinishReason::FINISH_REASON_LENGTH_CAPPED);
        return true;
    }
    return false;
}

bool FinishReasonsHandler::check_max_length(int32_t slot_id, int32_t batch_input_len,
                                             int32_t generated_length, int32_t max_new_tokens,
                                             int32_t max_total_length) {
    if (generated_length >= max_new_tokens ||
        (batch_input_len + generated_length) >= max_total_length) {
        set_finish_reason(slot_id, FinishReason::FINISH_REASON_MAX_LENGTH);
        return true;
    }
    return false;
}

bool FinishReasonsHandler::check_length_capped(int32_t slot_id, int32_t batch_input_len,
                                                int32_t generated_length, int32_t max_new_tokens) {
    if (generated_length >= max_new_tokens) {
        set_finish_reason(slot_id, FinishReason::FINISH_REASON_LENGTH_CAPPED);
        return true;
    }
    return false;
}

bool FinishReasonsHandler::check_stop_words(int32_t slot_id, int32_t batch_input_len,
                                             const std::vector<int32_t>& stop_words_list,
                                             int32_t generated_length, int32_t max_new_tokens,
                                             int32_t end_id, const std::vector<int32_t>& output_ids) {
    if (stop_words_list.empty()) {
        return false;
    }

    for (int32_t stop_id : stop_words_list) {
        if (!output_ids.empty() && output_ids.back() == stop_id) {
            set_finish_reason(slot_id, FinishReason::FINISH_REASON_STOP_WORDS);
            return true;
        }
    }
    return false;
}

std::string finish_reason_to_string(FinishReason reason) {
    switch (reason) {
        case FinishReason::FINISH_REASON_GENERATING:
            return "GENERATING";
        case FinishReason::FINISH_REASON_END_OF_SEQUENCE:
            return "END_OF_SEQUENCE";
        case FinishReason::FINISH_REASON_MAX_LENGTH:
            return "MAX_LENGTH";
        case FinishReason::FINISH_REASON_STOP_WORDS:
            return "STOP_WORDS";
        case FinishReason::FINISH_REASON_LENGTH_CAPPED:
            return "LENGTH_CAPPED";
        default:
            return "UNKNOWN";
    }
}

}  // namespace vulkan_executor::sampler
TRTLLM_NAMESPACE_END
