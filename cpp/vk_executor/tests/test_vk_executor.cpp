/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/vk_executor/vk_executor.h"
#include "sampler.h"

#include <cassert>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace tensorrt_llm::vulkan_executor;

static int test_counter = 0;
static int tests_passed = 0;

#define RUN_TEST(fn) \
    do { \
        test_counter++; \
        printf("[ RUN      ] %s\n", #fn); \
        fn(); \
        tests_passed++; \
        printf("[       OK ] %s\n", #fn); \
    } while (0)

// Test 1: Request lifecycle states
static void test_request_lifecycle()
{
    Request req{};
    req.request_id = 1;
    req.max_new_tokens = 10;
    req.input_token_ids = {1, 2, 3, 4, 5};

    assert(req.status == RequestStatus::kPending);
    req.status = RequestStatus::kRunning;
    assert(req.status == RequestStatus::kRunning);
    req.status = RequestStatus::kFinished;
    assert(req.status == RequestStatus::kFinished);
}

// Test 2: Sampling config defaults
static void test_sampling_defaults()
{
    SamplingConfig cfg{};
    assert(cfg.topk_topp.top_k == 1);
    assert(cfg.topk_topp.top_p == 1.0f);
    assert(cfg.beam.num_beams == 1);
    assert(cfg.penalty.repetition_penalty == 1.0f);
    assert(cfg.num_return_sequences == 1);
}

// Test 3: Batch structure
static void test_batch_structure()
{
    Batch batch{};
    assert(batch.batch_size == 0);
    assert(batch.status == RequestStatus::kPending);
    assert(!batch.has_encoder);
    assert(!batch.is_speculative);
}

// Test 4: ResourceMap defaults
static void test_resource_map_defaults()
{
    ResourceMap rm{};
    assert(rm.vulkan_buffer_memory == 0);
    assert(rm.vulkan_image_memory == 0);
    assert(rm.active_requests == 0);
    assert(rm.queued_requests == 0);
}

// Test 5: GpuResource defaults
static void test_gpu_resource_defaults()
{
    GpuResource gr{};
    assert(gr.total_memory_bytes == 0);
    assert(gr.free_memory_bytes == 0);
    assert(gr.compute_cap_major == 0);
    assert(gr.compute_cap_minor == 0);
    assert(gr.device_name.empty());
}

// Test 6: ForwardOutput structure
static void test_forward_output()
{
    ForwardOutput out{};
    assert(out.batch_size == 0);
    assert(out.seq_len == 0);
    assert(out.logits.empty());
}

// Test 7: PenaltyType enum values
static void test_penalty_type_values()
{
    assert(static_cast<uint8_t>(PenaltyType::kNone) == 0);
    assert(static_cast<uint8_t>(PenaltyType::kBrevity) == 1);
    assert(static_cast<uint8_t>(PenaltyType::kFamiliarity) == 2);
    assert(static_cast<uint8_t>(PenaltyType::kRepetition) == 3);
}

// Test 8: RequestStatus enum values
static void test_request_status_values()
{
    assert(static_cast<uint8_t>(RequestStatus::kPending) == 0);
    assert(static_cast<uint8_t>(RequestStatus::kRunning) == 1);
    assert(static_cast<uint8_t>(RequestStatus::kScheduled) == 2);
    assert(static_cast<uint8_t>(RequestStatus::kPreparing) == 3);
    assert(static_cast<uint8_t>(RequestStatus::kExecuting) == 4);
    assert(static_cast<uint8_t>(RequestStatus::kFinished) == 5);
    assert(static_cast<uint8_t>(RequestStatus::kCancelled) == 6);
    assert(static_cast<uint8_t>(RequestStatus::kError) == 7);
}

// Test 9: Create factory returns nullptr on bad path (no actual model)
static void test_create_factory_bad_path()
{
    auto executor = VkExecutor::Create(
        "/nonexistent/config",
        "/nonexistent/engine"
    );
    // Should return nullptr since engine doesn't exist
    assert(executor == nullptr);
}

// Test 10: Callbacks can be set without crashing
static void test_callback_setting()
{
    // This would need a valid executor, but we can at least verify the
    // callback types compile and are assignable
    RequestCallback req_cb = [](const Request&) {};
    BatchCallback   batch_cb = [](const Batch&) {};
    ErrorCallback   err_cb = [](int64_t, const std::string&) {};
    LogCallback     log_cb = [](const std::string&, int, const std::string&) {};

    (void)req_cb;
    (void)batch_cb;
    (void)err_cb;
    (void)log_cb;
}

    // Test 11: TorchSampler basic operations
static void test_torch_sampler_basic()
{
    // Initialize Vulkan backend
    auto backend = tensorrt_llm::common::VulkanBackend::getInstance();
    assert(backend != nullptr);
    assert(backend->initialize(0));

    constexpr int32_t max_slots = 32;
    constexpr int32_t vocab_size = 10;

    auto sampler = std::make_unique<TorchSampler>(*backend, max_slots, vocab_size);

    // Setup a request
    std::vector<int32_t> input_ids = {1, 2, 3};
    sampler::PenaltyConfig pen_cfg;
    pen_cfg.repetition_penalty_enabled = true;
    pen_cfg.repetition_penalty = 1.2f;
    pen_cfg.temperature = 0.8f;

    sampler::TopPDecayConfig tpd_config;
    tpd_config.top_p = 0.9f;

    std::vector<int32_t> banned_tokens;
    sampler->setup_request(0, input_ids, pen_cfg, tpd_config, banned_tokens, 10, 1);

    // Verify slot info
    const auto& info = sampler->get_slot_info(0);
    assert(info.input_token_ids == input_ids);
    assert(info.output_token_ids.empty());
    assert(info.generation_length == 0);

    // Create test logits (vocab_size = 10)
    std::vector<float> logits(vocab_size);
    for (int32_t i = 0; i < vocab_size; ++i) {
        logits[i] = static_cast<float>(i);
    }

    // Sample a token (greedy, no output history yet)
    int32_t token = sampler->sample_token(0, logits.data(), 1);
    assert(token >= 0 && token < vocab_size);

    // After sampling, generation_length should be incremented
    assert(sampler->get_slot_info(0).generation_length == 1);

    // Test finish reason
    assert(!sampler->is_finished(0));

    // Test reset
    sampler->reset_slot(0);
    assert(sampler->get_slot_info(0).output_token_ids.empty());
    assert(sampler->get_slot_info(0).generation_length == 0);
    assert(!sampler->is_finished(0));
}

int main()
{
    printf("=== VkExecutor Unit Tests ===\n\n");

    RUN_TEST(test_request_lifecycle);
    RUN_TEST(test_sampling_defaults);
    RUN_TEST(test_batch_structure);
    RUN_TEST(test_resource_map_defaults);
    RUN_TEST(test_gpu_resource_defaults);
    RUN_TEST(test_forward_output);
    RUN_TEST(test_penalty_type_values);
    RUN_TEST(test_request_status_values);
    RUN_TEST(test_create_factory_bad_path);
    RUN_TEST(test_callback_setting);
    RUN_TEST(test_torch_sampler_basic);

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, test_counter);
    return (tests_passed == test_counter) ? 0 : 1;
}
