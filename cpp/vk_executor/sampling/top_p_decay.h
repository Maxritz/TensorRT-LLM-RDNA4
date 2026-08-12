#pragma once

#include <cstdint>
#include <vector>

#include "tensorrt_llm/common/vulkanCommon.h"

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor
{
namespace sampler
{

inline constexpr int32_t DEFAULT_BEAM_IDX = 0;
inline constexpr int32_t DEFAULT_STEP_IDX = 0;

struct TopPDecayConfig
{
    bool top_p_decay_enabled = false;
    float top_p = 1.0f;
    float top_p_decay = 0.0f;
    int32_t top_p_decay_steps = 0;
    int32_t applied_len = 0;
};

class TopPDecayHandler
{
public:
    explicit TopPDecayHandler(int32_t max_slots = 1024);

    float compute_top_p(int32_t slot_id, const TopPDecayConfig& config);
    void update_applied_len(int32_t slot_id, int32_t new_len);
    void reset(int32_t slot_id);

    int32_t get_applied_len(int32_t slot_id) const;

private:
    int32_t m_max_slots;
    std::vector<int32_t> m_applied_lens;
};

}  // namespace sampler
}  // namespace vulkan_executor
TRTLLM_NAMESPACE_END
