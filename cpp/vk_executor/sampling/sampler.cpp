#include "sampler.h"

#include "finish_reasons.h"
#include "penalties.h"
#include "sampler_common.h"
#include "sampler_strategy.h"
#include "top_p_decay.h"
#include "token_ban.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#ifdef USE_VULKAN
#include "tensorrt_llm/common/vulkanRuntime.h"
#endif

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor
{

struct TorchSampler::Impl
{
    common::VulkanBackend& backend;
    int32_t max_slots;
    int32_t vocab_size;

    std::unique_ptr<sampler::PenaltyHandler> penalty_handler;
    std::unique_ptr<sampler::FinishReasonsHandler> finish_handler;
    std::unique_ptr<sampler::TopPDecayHandler> top_p_decay_handler;
    std::unique_ptr<sampler::TokenBanHandler> token_ban_handler;

    std::vector<SlotInfo> slot_infos;
    std::vector<int32_t> end_ids;
    std::vector<int32_t> max_new_tokens_vec;
    std::vector<int32_t> max_input_lengths;
    std::vector<int32_t> max_total_lengths;

    std::vector<uint64_t> random_seeds;
    std::vector<std::mt19937> rng_engines;

    std::vector<sampler::PenaltyConfig> penalty_configs;
    std::vector<sampler::TopPDecayConfig> top_p_decay_configs;
    std::vector<bool> slot_initialized;

    Impl(common::VulkanBackend& backend_, int32_t max_slots_, int32_t vocab_size_)
        : backend(backend_)
        , max_slots(max_slots_)
        , vocab_size(vocab_size_)
        , slot_infos(max_slots_)
        , end_ids(max_slots, -1)
        , max_new_tokens_vec(max_slots, 0)
        , max_input_lengths(max_slots, 0)
        , max_total_lengths(max_slots, 2048)
        , random_seeds(max_slots, 0)
        , penalty_configs(max_slots)
        , top_p_decay_configs(max_slots)
        , slot_initialized(max_slots, false)
    {
        penalty_handler = std::make_unique<sampler::PenaltyHandler>(backend, max_slots);
        finish_handler = std::make_unique<sampler::FinishReasonsHandler>(max_slots);
        top_p_decay_handler = std::make_unique<sampler::TopPDecayHandler>(max_slots);
        token_ban_handler = std::make_unique<sampler::TokenBanHandler>(max_slots);

        std::random_device rd;
        for (int32_t i = 0; i < max_slots; ++i) {
            random_seeds[i] = (static_cast<uint64_t>(rd()) << 32) | static_cast<uint64_t>(rd());
            rng_engines.emplace_back(static_cast<uint32_t>(random_seeds[i]));
        }
    }

    std::mt19937& get_rng(int32_t slot_id)
    {
        return rng_engines[slot_id];
    }
};

TorchSampler::TorchSampler(common::VulkanBackend& backend, int32_t max_slots, int32_t vocab_size)
    : m_impl(std::make_unique<Impl>(backend, max_slots, vocab_size)) {}

TorchSampler::~TorchSampler() = default;

void TorchSampler::setup_request(int32_t slot_id, const std::vector<int32_t>& input_token_ids,
                                  const sampler::PenaltyConfig& penalty_config,
                                  const sampler::TopPDecayConfig& top_p_decay_config,
                                  const std::vector<int32_t>& banned_tokens,
                                  int32_t max_new_tokens_val, int32_t end_id)
{
    if (slot_id < 0 || slot_id >= m_impl->max_slots) {
        return;
    }

    auto& info = m_impl->slot_infos[slot_id];
    info.input_token_ids = input_token_ids;
    info.output_token_ids.clear();
    info.step_idx = 0;
    info.generation_length = 0;
    info.beam_width = 1;

    m_impl->penalty_configs[slot_id] = penalty_config;
    m_impl->top_p_decay_configs[slot_id] = top_p_decay_config;
    m_impl->end_ids[slot_id] = end_id;
    m_impl->max_new_tokens_vec[slot_id] = max_new_tokens_val;
    m_impl->max_input_lengths[slot_id] = static_cast<int32_t>(input_token_ids.size());

    if (penalty_config.max_input_length > 0) {
        m_impl->max_input_lengths[slot_id] = penalty_config.max_input_length;
    }

    m_impl->penalty_handler->get_store().init_slot(slot_id, input_token_ids,
                                                     m_impl->max_input_lengths[slot_id]);
    m_impl->finish_handler->reset(slot_id);
    m_impl->top_p_decay_handler->reset(slot_id);
    m_impl->token_ban_handler->reset(slot_id);

    m_impl->token_ban_handler->ban_tokens_for_request(slot_id, banned_tokens);
    m_impl->slot_initialized[slot_id] = true;
}

void TorchSampler::reset_slot(int32_t slot_id)
{
    if (slot_id < 0 || slot_id >= m_impl->max_slots) {
        return;
    }
    m_impl->slot_infos[slot_id] = SlotInfo{};
    m_impl->end_ids[slot_id] = -1;
    m_impl->max_new_tokens_vec[slot_id] = 0;
    m_impl->max_input_lengths[slot_id] = 0;
    m_impl->max_total_lengths[slot_id] = 2048;
    m_impl->penalty_configs[slot_id] = sampler::PenaltyConfig{};
    m_impl->top_p_decay_configs[slot_id] = sampler::TopPDecayConfig{};
    m_impl->finish_handler->reset(slot_id);
    m_impl->top_p_decay_handler->reset(slot_id);
    m_impl->token_ban_handler->reset(slot_id);
    m_impl->slot_initialized[slot_id] = false;
}

int32_t TorchSampler::sample_token(int32_t slot_id, float* logits_ptr, int32_t num_beams)
{
    if (slot_id < 0 || slot_id >= m_impl->max_slots || !m_impl->slot_initialized[slot_id]) {
        return 0;
    }

    auto& info = m_impl->slot_infos[slot_id];
    const auto& penalty_config = m_impl->penalty_configs[slot_id];
    const auto& top_p_decay_config = m_impl->top_p_decay_configs[slot_id];

    std::vector<float> logits(logits_ptr, logits_ptr + m_impl->vocab_size);

    m_impl->token_ban_handler->apply_bans(logits, slot_id, m_impl->vocab_size);

    float temperature = penalty_config.temperature;

    if (!info.output_token_ids.empty()) {
        m_impl->penalty_handler->apply_penalties(slot_id, penalty_config, logits,
                                                  info.output_token_ids,
                                                  m_impl->vocab_size);
    }

    float top_p = m_impl->top_p_decay_handler->compute_top_p(slot_id, top_p_decay_config);

    auto& rng = m_impl->get_rng(slot_id);

    int32_t sampled_token = 0;

    if (m_impl->backend.isActive()) {
        sampled_token = sample_token_gpu(slot_id, logits, penalty_config, top_p, temperature, rng);
    } else {
        sampled_token = sample_token_cpu(logits, top_p, temperature, m_impl->vocab_size, rng);
    }

    info.output_token_ids.push_back(sampled_token);
    info.generation_length++;
    info.step_idx++;

    m_impl->penalty_handler->get_store().add_token(slot_id, sampled_token);

    if (penalty_config.max_input_length > 0) {
        m_impl->max_input_lengths[slot_id] = std::max(m_impl->max_input_lengths[slot_id],
                                                       static_cast<int32_t>(info.output_token_ids.size()));
    }

    bool finished = false;
    int32_t input_len = static_cast<int32_t>(info.input_token_ids.size());
    finished = m_impl->finish_handler->check_end_of_sequence(
        slot_id, sampled_token, m_impl->end_ids[slot_id],
        info.generation_length, m_impl->max_new_tokens_vec[slot_id]);

    if (!finished) {
        finished = m_impl->finish_handler->check_max_length(
            slot_id, input_len, info.generation_length,
            m_impl->max_new_tokens_vec[slot_id], m_impl->max_total_lengths[slot_id]);
    }

    return sampled_token;
}

int32_t TorchSampler::sample_token_cpu(std::vector<float>& logits, float top_p,
                                         float temperature, int32_t vocab_size, std::mt19937& rng)
{
    if (temperature > 0 && temperature != 1.0f) {
        for (auto& l : logits) {
            l /= temperature;
        }
    }

    int32_t sampled_token = 0;

    if (top_p < 1.0f) {
        std::vector<std::pair<float, int32_t>> token_probs;
        for (int32_t i = 0; i < vocab_size; ++i) {
            token_probs.emplace_back(logits[i], i);
        }

        std::sort(token_probs.begin(), token_probs.end(),
                  [](const auto& a, const auto& b) {
                      return a.first > b.first;
                  });

        float max_logit = token_probs.front().first;
        float sum_exp = 0.0f;
        std::vector<std::pair<float, int32_t>> filtered_probs;

        for (const auto& [logit_val, token_id] : token_probs) {
            float prob = std::exp(logit_val - max_logit);
            sum_exp += prob;
            filtered_probs.emplace_back(prob, token_id);
            if (sum_exp >= top_p) {
                break;
            }
        }

        float total_prob = 0.0f;
        for (const auto& [prob, _] : filtered_probs) {
            total_prob += prob;
        }

        if (total_prob <= 0.0f) {
            sampled_token = static_cast<int32_t>(
                std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        } else {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float r = dist(rng) * total_prob;

            float cumulative = 0.0f;
            for (const auto& [prob, token_id] : filtered_probs) {
                cumulative += prob;
                if (r <= cumulative) {
                    sampled_token = token_id;
                    break;
                }
            }
        }
    } else {
        auto max_it = std::max_element(logits.begin(), logits.end());
        sampled_token = static_cast<int32_t>(std::distance(logits.begin(), max_it));
    }

    return sampled_token;
}

int32_t TorchSampler::sample_token_gpu(int32_t slot_id, std::vector<float>& logits,
                                         const sampler::PenaltyConfig& penalty_config,
                                         float top_p, float temperature, std::mt19937& rng)
{
    if (temperature > 0 && temperature != 1.0f) {
        for (auto& l : logits) {
            l /= temperature;
        }
    }

    const int32_t vocab_size = m_impl->vocab_size;
    const size_t logits_bytes = vocab_size * sizeof(float);

    void* gpu_logits = common::VulkanBackend::malloc(logits_bytes);
    if (!gpu_logits) {
        return sample_token_cpu(logits, top_p, temperature, vocab_size, rng);
    }

    common::VulkanBackend::memcpyHostToDevice(gpu_logits, logits.data(), logits_bytes);

    int32_t top_k = 1;
    if (top_p < 1.0f) {
        top_k = vocab_size;
    }

    void* gpu_topk_indices = common::VulkanBackend::malloc(top_k * sizeof(uint32_t));
    void* gpu_topk_values = common::VulkanBackend::malloc(top_k * sizeof(float));

    if (!gpu_topk_indices || !gpu_topk_values) {
        common::VulkanBackend::free(gpu_logits);
        common::VulkanBackend::free(gpu_topk_indices);
        common::VulkanBackend::free(gpu_topk_values);
        return sample_token_cpu(logits, top_p, temperature, vocab_size, rng);
    }

    if (top_p < 1.0f) {
        void* gpu_probs = common::VulkanBackend::malloc(logits_bytes);
        if (gpu_probs) {
            common::VulkanBackend::launchSoftmax(gpu_logits, gpu_probs, 1, 1, vocab_size);

            std::vector<float> probs(vocab_size);
            common::VulkanBackend::memcpyDeviceToHost(probs.data(), gpu_probs, logits_bytes);
            common::VulkanBackend::free(gpu_probs);

            std::vector<std::pair<float, int32_t>> token_probs;
            for (int32_t i = 0; i < vocab_size; ++i) {
                token_probs.emplace_back(probs[i], i);
            }

            std::sort(token_probs.begin(), token_probs.end(),
                      [](const auto& a, const auto& b) {
                          return a.first > b.first;
                      });

            float cumsum = 0.0f;
            std::vector<std::pair<float, int32_t>> filtered;
            for (const auto& [prob, token_id] : token_probs) {
                cumsum += prob;
                filtered.emplace_back(prob, token_id);
                if (cumsum >= top_p) {
                    break;
                }
            }

            float total = 0.0f;
            for (const auto& [prob, _] : filtered) {
                total += prob;
            }

            if (total > 0.0f) {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                float r = dist(rng) * total;
                float cumulative = 0.0f;
                for (const auto& [prob, token_id] : filtered) {
                    cumulative += prob;
                    if (r <= cumulative) {
                        common::VulkanBackend::free(gpu_logits);
                        common::VulkanBackend::free(gpu_topk_indices);
                        common::VulkanBackend::free(gpu_topk_values);
                        return token_id;
                    }
                }
            }
        }
        common::VulkanBackend::free(gpu_logits);
        common::VulkanBackend::free(gpu_topk_indices);
        common::VulkanBackend::free(gpu_topk_values);
        return sample_token_cpu(logits, top_p, temperature, vocab_size, rng);
    }

    common::VulkanBackend::launchTopKGeneral(
        gpu_logits, gpu_topk_indices, gpu_topk_values, 1, vocab_size, top_k);

    std::vector<uint32_t> topk_indices(top_k);
    common::VulkanBackend::memcpyDeviceToHost(topk_indices.data(), gpu_topk_indices, top_k * sizeof(uint32_t));

    common::VulkanBackend::free(gpu_logits);
    common::VulkanBackend::free(gpu_topk_indices);
    common::VulkanBackend::free(gpu_topk_values);

    int32_t sampled_token = static_cast<int32_t>(topk_indices[0]);
    return sampled_token;
}

int32_t TorchSampler::beam_search_step(int32_t slot_id, float* logits_ptr, int32_t beam_width,
                                        int32_t /*num_return_sequences*/, int32_t /*max_new_tokens*/)
{
    auto& info = m_impl->slot_infos[slot_id];
    info.beam_width = beam_width;

    int32_t best_token = 0;
    float best_logit = -std::numeric_limits<float>::lowest();
    for (int32_t i = 0; i < m_impl->vocab_size; ++i) {
        if (logits_ptr[i] > best_logit) {
            best_logit = logits_ptr[i];
            best_token = i;
        }
    }

    info.output_token_ids.push_back(best_token);
    info.generation_length++;
    info.step_idx++;

    m_impl->penalty_handler->get_store().add_token(slot_id, best_token);

    return best_token;
}

bool TorchSampler::is_finished(int32_t slot_id) const
{
    if (slot_id < 0 || slot_id >= m_impl->max_slots) {
        return false;
    }
    return m_impl->finish_handler->is_finished(slot_id);
}

std::string TorchSampler::get_finish_reason_string(int32_t slot_id) const
{
    if (slot_id < 0 || slot_id >= m_impl->max_slots) {
        return "UNKNOWN";
    }
    return finish_reason_to_string(m_impl->finish_handler->get_finish_reason(slot_id));
}

const SlotInfo& TorchSampler::get_slot_info(int32_t slot_id) const
{
    static SlotInfo empty;
    if (slot_id < 0 || slot_id >= m_impl->max_slots) {
        return empty;
    }
    return m_impl->slot_infos[slot_id];
}

void TorchSampler::set_end_id(int32_t slot_id, int32_t end_id)
{
    if (slot_id >= 0 && slot_id < m_impl->max_slots) {
        m_impl->end_ids[slot_id] = end_id;
    }
}

int32_t TorchSampler::get_end_id(int32_t slot_id) const
{
    if (slot_id < 0 || slot_id >= m_impl->max_slots) {
        return -1;
    }
    return m_impl->end_ids[slot_id];
}

void TorchSampler::set_max_new_tokens(int32_t slot_id, int32_t max_new_tokens_val)
{
    if (slot_id >= 0 && slot_id < m_impl->max_slots) {
        m_impl->max_new_tokens_vec[slot_id] = max_new_tokens_val;
    }
}

int32_t TorchSampler::get_max_new_tokens(int32_t slot_id) const
{
    if (slot_id < 0 || slot_id >= m_impl->max_slots) {
        return 0;
    }
    return m_impl->max_new_tokens_vec[slot_id];
}

}  // namespace vulkan_executor
TRTLLM_NAMESPACE_END
