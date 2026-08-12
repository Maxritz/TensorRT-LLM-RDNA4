#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "tensorrt_llm/common/vulkanCommon.h"

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor
{
namespace sampler
{

class TokenBanHandler
{
public:
    explicit TokenBanHandler(int32_t max_slots = 1024);

    void ban_tokens_for_request(int32_t slot_id, const std::vector<int32_t>& banned_tokens);
    void unban_tokens(int32_t slot_id, const std::vector<int32_t>& tokens);
    void unban_all(int32_t slot_id);
    void reset(int32_t slot_id);

    bool is_token_banned(int32_t slot_id, int32_t token_id) const;

    void apply_bans(std::vector<float>& logits, int32_t slot_id, int32_t vocab_size) const;

private:
    int32_t m_max_slots;
    std::vector<std::unordered_map<int32_t, bool>> m_banned_tokens;
};

}  // namespace sampler
}  // namespace vulkan_executor
TRTLLM_NAMESPACE_END
