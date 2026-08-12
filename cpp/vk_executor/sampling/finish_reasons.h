#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tensorrt_llm/vk_executor/vk_executor.h"

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor
{
namespace sampler
{

enum class FinishReason : int32_t
{
    FINISH_REASON_GENERATING = 0,
    FINISH_REASON_END_OF_SEQUENCE = 1,
    FINISH_REASON_MAX_LENGTH = 2,
    FINISH_REASON_STOP_WORDS = 3,
    FINISH_REASON_LENGTH_CAPPED = 4
};

class FinishReasonsHandler
{
public:
    explicit FinishReasonsHandler(int32_t max_slots);

    void reset(int32_t slot_id);
    bool is_finished(int32_t slot_id) const;
    FinishReason get_finish_reason(int32_t slot_id) const;
    void set_finish_reason(int32_t slot_id, FinishReason reason);

    bool check_stop_words(int32_t slot_id, int32_t batch_input_len,
                          const std::vector<int32_t>& stop_words_list,
                          int32_t generated_length, int32_t max_new_tokens,
                          int32_t end_id, const std::vector<int32_t>& output_ids);

    bool check_end_of_sequence(int32_t slot_id, int32_t last_token_id,
                               int32_t end_id, int32_t generated_length,
                               int32_t max_new_tokens);

    bool check_max_length(int32_t slot_id, int32_t batch_input_len,
                          int32_t generated_length, int32_t max_new_tokens,
                          int32_t max_total_length);

    bool check_length_capped(int32_t slot_id, int32_t batch_input_len,
                             int32_t generated_length, int32_t max_new_tokens);

private:
    int32_t m_max_slots;
    std::vector<FinishReason> m_finish_reasons;
};

std::string finish_reason_to_string(FinishReason reason);

}  // namespace sampler
}  // namespace vulkan_executor
TRTLLM_NAMESPACE_END
