#include "token_ban.h"

#include <algorithm>
#include <limits>

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor::sampler
{

TokenBanHandler::TokenBanHandler(int32_t max_slots)
    : m_max_slots(max_slots) {
    m_banned_tokens.resize(max_slots);
}

void TokenBanHandler::ban_tokens_for_request(int32_t slot_id, const std::vector<int32_t>& banned_tokens) {
    if (slot_id < 0 || slot_id >= m_max_slots) {
        return;
    }
    for (int32_t token_id : banned_tokens) {
        m_banned_tokens[slot_id][token_id] = true;
    }
}

void TokenBanHandler::unban_tokens(int32_t slot_id, const std::vector<int32_t>& tokens) {
    if (slot_id < 0 || slot_id >= m_max_slots) {
        return;
    }
    for (int32_t token_id : tokens) {
        m_banned_tokens[slot_id].erase(token_id);
    }
}

void TokenBanHandler::unban_all(int32_t slot_id) {
    if (slot_id < 0 || slot_id >= m_max_slots) {
        return;
    }
    m_banned_tokens[slot_id].clear();
}

void TokenBanHandler::reset(int32_t slot_id) {
    unban_all(slot_id);
}

bool TokenBanHandler::is_token_banned(int32_t slot_id, int32_t token_id) const {
    if (slot_id < 0 || slot_id >= m_max_slots) {
        return false;
    }
    return m_banned_tokens[slot_id].find(token_id) != m_banned_tokens[slot_id].end();
}

void TokenBanHandler::apply_bans(std::vector<float>& logits, int32_t slot_id, int32_t vocab_size) const {
    if (slot_id < 0 || slot_id >= m_max_slots) {
        return;
    }
    const auto& banned = m_banned_tokens[slot_id];
    for (const auto& [token_id, _] : banned) {
        if (token_id >= 0 && token_id < vocab_size) {
            logits[token_id] = -std::numeric_limits<float>::infinity();
        }
    }
}

}  // namespace vulkan_executor::sampler
TRTLLM_NAMESPACE_END
