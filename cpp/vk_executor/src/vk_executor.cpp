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

#include "tensorrt_llm/common/vulkanCommon.h"
#include "tensorrt_llm/common/vulkanRuntime.h"
#include "tensorrt_llm/common/vulkanContext.h"
#include "tensorrt_llm/common/vulkanBackend.h"
#include "tensorrt_llm/kernels/vulkanKernelRegistry.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <queue>
#include <thread>
#include <condition_variable>
#include <unordered_map>

TRTLLM_NAMESPACE_BEGIN
namespace vulkan_executor
{

// ============================================================================
// Internal sub-components (will be expanded into separate files)
// ============================================================================

class Scheduler
{
public:
    bool schedule(std::vector<Request*>& requests, uint32_t max_batch_size)
    {
        if (requests.size() <= max_batch_size) return true;
        requests.resize(max_batch_size);
        return true;
    }
    uint32_t get_batch_size() const noexcept { return m_batch_size; }
    void     set_batch_size(uint32_t sz) noexcept { m_batch_size = sz; }
private:
    uint32_t m_batch_size{32};
};

class KvCacheManager
{
public:
    uint64_t allocated_bytes() const noexcept { return m_allocated; }
    uint64_t total_bytes() const noexcept { return m_total; }
    void* acquire_buffer(uint64_t size)
    {
        // TODO: integrate with VulkanMemoryAllocator
        m_allocated += size;
        return nullptr;
    }
    void release_buffer(void* ptr) { (void)ptr; }
private:
    uint64_t m_allocated{0};
    uint64_t m_total{0};
};

class ModelRunner
{
public:
    bool load_model(std::string_view engine_path,
                    std::shared_ptr<common::VulkanContext> ctx)
    {
        // Load SPIR-V shaders and build compute pipelines via VulkanKernelRegistry.
        // Mirrors PyExecutor._load_engine which loads CUDA kernels/weights.
        m_ctx = ctx;
        m_kernel_dispatcher = std::make_shared<kernels::VulkanKernelDispatcher>(ctx);
        if (!m_kernel_dispatcher) {
            TLLM_LOG_ERROR("ModelRunner: failed to create VulkanKernelDispatcher");
            return false;
        }
        m_engine_path = std::string(engine_path);
        TLLM_LOG_INFO("ModelRunner: loaded engine from %s", m_engine_path.c_str());
        return true;
    }
    bool run_forward(Batch& batch, ForwardOutput& out,
                     const TorchSampler& sampler)
    {
        // Dispatch Vulkan compute pipeline for model forward (GEMM + activations).
        // Mirrors PyExecutor._execute_model → torch forward + CUDA gemm kernels.
        out.batch_size = static_cast<int32_t>(batch.requests.size());
        out.seq_len    = batch.max_input_length;

        // Allocate GPU-side logits buffer via VulkanMemoryAllocator.
        // In a full implementation, this would be a VkBuffer allocated from
        // the VulkanBackend allocator and bound to the GEMM output descriptor.
        uint32_t vocab_size = 32000; // Llama-2 vocab default; TODO: read from engine
        out.logits.resize(static_cast<size_t>(out.batch_size) * vocab_size);

        // Dispatch a dummy GEMM to exercise the kernel dispatch path.
        // A real implementation would iterate over model layers, dispatching
        // each linear as dispatchFp16Gemm or dispatchQ4_0Gemm depending on
        // the quant format stored in the engine.
        void* fake_weight = nullptr;
        void* fake_activation = nullptr;
        void* fake_output = out.logits.data();
        uint32_t M = static_cast<uint32_t>(out.batch_size);
        uint32_t N = vocab_size;
        uint32_t K = 4096; // hidden size

        // Dispatch RMSNorm + GEMM as a representative forward op chain.
        // The actual model weights would be uploaded as VkBuffers and bound
        // via descriptor sets in a real deployment.
        auto result = m_kernel_dispatcher->dispatchRmsNorm(
            fake_activation, fake_weight, nullptr, fake_output,
            1e-6f, K, M);
        if (result != common::VulkanResult::SUCCESS &&
            result != common::VulkanResult::FEATURE_NOT_PRESENT) {
            TLLM_LOG_WARNING("ModelRunner: RMSNorm dispatch failed: %d", static_cast<int>(result));
        }

        result = m_kernel_dispatcher->dispatchFp16Gemm(
            fake_activation, fake_weight, fake_output,
            M, N, K);
        if (result != common::VulkanResult::SUCCESS &&
            result != common::VulkanResult::FEATURE_NOT_PRESENT) {
            TLLM_LOG_WARNING("ModelRunner: GEMM dispatch failed: %d", static_cast<int>(result));
            return false;
        }

        // Fill the logits with deterministic test values so sampling works.
        // In a real run, dispatchSoftmax + dispatchTopKGeneral on GPU produce these.
        for (size_t i = 0; i < out.logits.size(); ++i) {
            out.logits[i] = static_cast<float>(i % vocab_size) / static_cast<float>(vocab_size);
        }

        return true;
    }
    void cleanup() {}

    const kernels::VulkanKernelDispatcher& get_kernel_dispatcher() const
    { return *m_kernel_dispatcher; }
    kernels::VulkanKernelDispatcher& get_kernel_dispatcher_nonconst()
    { return *m_kernel_dispatcher; }

private:
    std::string m_engine_path;
    std::shared_ptr<common::VulkanContext> m_ctx;
    std::shared_ptr<kernels::VulkanKernelDispatcher> m_kernel_dispatcher;
};

// ============================================================================
// Impl struct (Pimpl idiom)
// ============================================================================

struct VkExecutor::Impl
{
    // --- Vulkan handles (mirroring PyExecutor's _prepare) ---
    std::shared_ptr<common::VulkanBackend>        backend;
    std::shared_ptr<common::VulkanRuntime>        vulkan_runtime;
    std::shared_ptr<common::VulkanContext>        vulkan_context;
    VkDevice                                    device{VK_NULL_HANDLE};
    VkQueue                                     compute_queue{VK_NULL_HANDLE};
    uint32_t                                    queue_family_index{0};
    VkCommandPool                                command_pool{VK_NULL_HANDLE};

    // --- Sub-components ---
    std::unique_ptr<Scheduler>                  scheduler;
    std::unique_ptr<TorchSampler>               sampler;
    std::unique_ptr<KvCacheManager>             kv_cache_manager;
    std::unique_ptr<ModelRunner>                model_runner;

    // --- Execution state ---
    bool                                        running{false};
    bool                                        paused{false};
    std::atomic_bool                            stop_flag{false};
    uint32_t                                    max_batch_size{32};

    // --- Request queues ---
    std::queue<Request*>                        pending_queue;
    std::vector<Request*>                       active_requests;
    std::mutex                                  queue_mutex;
    std::condition_variable                     cv;

    // --- Batch tracking ---
    std::unique_ptr<Batch>                      current_batch_obj;

    // --- Callbacks ---
    RequestCallback                             req_cb;
    BatchCallback                               batch_cb;
    ErrorCallback                               err_cb;
    LogCallback                                 log_cb;
    std::string                                 log_level{"info"};

    // --- Config ---
    std::string                                 config_str;
    std::vector<std::string>                    supported_dtypes{"fp16", "bf16", "int8", "fp8"};
    std::vector<std::string>                    loaded_adapters;

    // --- Thread ---
    std::thread                                 executor_thread;

    Impl() = default;
    ~Impl()
    {
        if (executor_thread.joinable()) {
            stop_flag.store(true);
            cv.notify_all();
            executor_thread.join();
        }
        if (command_pool != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, command_pool, nullptr);
        }
    }
};

// ============================================================================
// VkExecutor implementation
// ============================================================================

VkExecutor::VkExecutor()
    : pimpl_(std::make_unique<Impl>())
{
}

VkExecutor::~VkExecutor() = default;

// --- Factory method (mirrors PyExecutorCreator.create_executor) ---

std::unique_ptr<VkExecutor> VkExecutor::Create(
    std::string_view config_path,
    std::string_view engine_path,
    std::vector<std::string> plugin_paths,
    std::vector<std::string> adapter_paths)
{
    auto executor = std::unique_ptr<VkExecutor>(new VkExecutor());
    auto& impl = *executor->pimpl_;

    // 1. Initialize Vulkan runtime (mirrors PyExecutor._prepare)
    auto backend = common::VulkanBackend::getInstance();
    if (backend == nullptr || !backend->initialize(0)) {
        if (impl.err_cb) {
            impl.err_cb(-1, "Failed to initialize Vulkan backend");
        }
        TLLM_LOG_ERROR("VkExecutor::Create: Failed to initialize Vulkan backend");
        return nullptr;
    }
    impl.backend = backend;

    auto runtime = backend->getRuntime();
    impl.vulkan_runtime = runtime;
    auto ctx = runtime->getContext();
    impl.vulkan_context       = ctx;
    impl.device               = ctx->getDevice();
    impl.compute_queue        = ctx->getComputeQueue();
    impl.queue_family_index   = ctx->getComputeQueueFamilyIndex();

    // Create command pool for compute operations
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = impl.queue_family_index;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(impl.device, &pool_info, nullptr, &impl.command_pool) != VK_SUCCESS) {
        TLLM_LOG_ERROR("VkExecutor::Create: failed to create command pool");
        return nullptr;
    }

    // 2. Load model engine (mirrors PyExecutor._load_engine)
    impl.model_runner = std::make_unique<ModelRunner>();
    if (!impl.model_runner->load_model(engine_path, ctx)) {
        if (impl.err_cb) {
            impl.err_cb(-1, "Failed to load model engine");
        }
        TLLM_LOG_ERROR("VkExecutor::Create: Failed to load engine from %.*s",
                       static_cast<int>(engine_path.size()), engine_path.data());
        return nullptr;
    }

    // 3. Initialize sub-components
    impl.scheduler        = std::make_unique<Scheduler>();
    impl.sampler          = std::make_unique<TorchSampler>(*impl.backend, 1024, 32000);
    impl.kv_cache_manager = std::make_unique<KvCacheManager>();

    // 4. Read config
    impl.config_str = std::string(config_path);

    // 5. Load plugins and adapters (TODO: wire into shader registry)
    impl.loaded_adapters = std::move(adapter_paths);

    TLLM_LOG_INFO("VkExecutor created successfully (config=%.*s, engine=%.*s)",
                  static_cast<int>(config_path.size()), config_path.data(),
                  static_cast<int>(engine_path.size()), engine_path.data());

    return executor;
}

// --- Request management (mirrors PyExecutor.enqueue_request) ---

bool VkExecutor::enqueue_request(Request& request)
{
    std::lock_guard<std::mutex> lock(pimpl_->queue_mutex);
    request.status = RequestStatus::kPending;
    request.enqueue_time = std::chrono::steady_clock::now();
    pimpl_->pending_queue.push(&request);
    pimpl_->cv.notify_all();
    return true;
}

void VkExecutor::cancel_request(int64_t request_id)
{
    std::lock_guard<std::mutex> lock(pimpl_->queue_mutex);
    // Walk pending queue
    std::queue<Request*> temp;
    while (!pimpl_->pending_queue.empty()) {
        auto* req = pimpl_->pending_queue.front();
        pimpl_->pending_queue.pop();
        if (req->request_id == request_id) {
            req->status = RequestStatus::kCancelled;
            continue;
        }
        temp.push(req);
    }
    std::swap(temp, pimpl_->pending_queue);

    // Walk active
    pimpl_->active_requests.erase(
        std::remove_if(pimpl_->active_requests.begin(), pimpl_->active_requests.end(),
            [request_id](Request* r) {
                if (r->request_id == request_id) {
                    r->status = RequestStatus::kCancelled;
                    return true;
                }
                return false;
            }),
        pimpl_->active_requests.end());
}

void VkExecutor::cancel_all_requests()
{
    std::lock_guard<std::mutex> lock(pimpl_->queue_mutex);
    while (!pimpl_->pending_queue.empty()) {
        pimpl_->pending_queue.front()->status = RequestStatus::kCancelled;
        pimpl_->pending_queue.pop();
    }
    for (auto* req : pimpl_->active_requests) {
        req->status = RequestStatus::kCancelled;
    }
}

// --- Execution control (mirrors PyExecutor.run / _execute_iteration) ---

bool VkExecutor::is_ready() const
{
    return pimpl_->vulkan_runtime && pimpl_->model_runner && !pimpl_->stop_flag.load();
}

bool VkExecutor::start()
{
    if (!is_ready()) return false;
    pimpl_->running = true;
    pimpl_->stop_flag.store(false);
    pimpl_->executor_thread = std::thread([this]() {
        while (!pimpl_->stop_flag.load()) {
            if (!pimpl_->paused) {
                execute_iteration();
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });
    return true;
}

void VkExecutor::stop()
{
    pimpl_->stop_flag.store(true);
    pimpl_->cv.notify_all();
    if (pimpl_->executor_thread.joinable()) {
        pimpl_->executor_thread.join();
    }
    pimpl_->running = false;
}

void VkExecutor::pause()
{
    pimpl_->paused = true;
}

void VkExecutor::resume()
{
    pimpl_->paused = false;
    pimpl_->cv.notify_all();
}

int32_t VkExecutor::execute_iteration()
{
    // 1. Pull requests from pending queue (mirrors PyExecutor._dequeue_requests)
    std::vector<Request*> batch_requests;
    {
        std::lock_guard<std::mutex> lock(pimpl_->queue_mutex);
        while (!pimpl_->pending_queue.empty() &&
               batch_requests.size() < pimpl_->scheduler->get_batch_size()) {
            auto* req = pimpl_->pending_queue.front();
            pimpl_->pending_queue.pop();
            req->status = RequestStatus::kRunning;
            batch_requests.push_back(req);
            pimpl_->active_requests.push_back(req);
        }
    }

    if (batch_requests.empty()) {
        return 0;
    }

    // 2. Build batch (mirrors PyExecutor._build_batch)
    prepare_batch(batch_requests);

    // 3. Run forward pass (mirrors PyExecutor._execute_model)
    if (pimpl_->current_batch_obj) {
        ForwardOutput fwd_out;
        if (pimpl_->model_runner->run_forward(*pimpl_->current_batch_obj, fwd_out,
                                               *pimpl_->sampler)) {
            // 4. Sampling (mirrors PyExecutor._decode)
            for (auto* req : batch_requests) {
                const auto& cfg = req->sampling_config;
                sampler::PenaltyConfig pen_cfg;
                pen_cfg.repetition_penalty_enabled = true;
                pen_cfg.repetition_penalty = cfg.penalty.repetition_penalty;
                pen_cfg.presence_penalty_enabled = true;
                pen_cfg.presence_penalty = cfg.penalty.presence_penalty;
                pen_cfg.frequency_penalty_enabled = true;
                pen_cfg.frequency_penalty = cfg.penalty.frequency_penalty;
                pen_cfg.max_input_length = static_cast<int32_t>(req->input_token_ids.size());
                pen_cfg.temperature = 0.7f;

                sampler::TopPDecayConfig tpd_config;
                tpd_config.top_p = cfg.topk_topp.top_p;
                tpd_config.top_p_decay_enabled = false;

                std::vector<int32_t> banned_tokens;
                int32_t end_id = 1; // Default EOS token; TODO: read from model config
                pimpl_->sampler->setup_request(
                    static_cast<int32_t>(req->request_id % 1024),
                    req->input_token_ids, pen_cfg, tpd_config,
                    banned_tokens, req->max_new_tokens, end_id);

                int32_t token_id = pimpl_->sampler->sample_token(
                    static_cast<int32_t>(req->request_id % 1024),
                    fwd_out.logits.data(), 1);

                if (req->token_callback) {
                    req->token_callback(token_id);
                }
                req->output_token_ids.push_back(token_id);
                req->status = RequestStatus::kFinished;
            }
        }
    }

    // 5. Callbacks + GC
    if (pimpl_->batch_cb && pimpl_->current_batch_obj) {
        pimpl_->batch_cb(*pimpl_->current_batch_obj);
    }

    gc();
    return static_cast<int32_t>(batch_requests.size());
}

void VkExecutor::flush()
{
    std::lock_guard<std::mutex> lock(pimpl_->queue_mutex);
    while (!pimpl_->pending_queue.empty()) {
        pimpl_->pending_queue.front()->status = RequestStatus::kCancelled;
        pimpl_->pending_queue.pop();
    }
}

// --- Batch operations ---

uint32_t VkExecutor::get_batch_size() const
{
    return pimpl_->scheduler ? pimpl_->scheduler->get_batch_size() : 0;
}

uint32_t VkExecutor::get_num_running_requests() const
{
    std::lock_guard<std::mutex> lock(pimpl_->queue_mutex);
    return static_cast<uint32_t>(pimpl_->active_requests.size());
}

void VkExecutor::prepare_batch(std::vector<Request*> requests)
{
    if (requests.empty()) return;

    pimpl_->scheduler->schedule(requests, pimpl_->max_batch_size);

    auto batch = std::make_unique<Batch>();
    batch->batch_id           = 0;
    batch->requests           = std::move(requests);
    batch->batch_size         = static_cast<int32_t>(batch->requests.size());
    batch->scheduled_length   = batch->batch_size;
    batch->max_input_length   = 0;
    batch->max_output_length  = 0;
    batch->status             = RequestStatus::kScheduled;

    // Compute batch metadata
    for (auto* req : batch->requests) {
        batch->max_input_length  = std::max(batch->max_input_length, static_cast<int32_t>(req->input_token_ids.size()));
        batch->max_output_length = std::max(batch->max_output_length, req->max_new_tokens);
    }

    pimpl_->current_batch_obj = std::move(batch);
}

Batch* VkExecutor::current_batch() const
{
    return pimpl_->current_batch_obj.get();
}

// --- Resource management ---

ResourceMap VkExecutor::get_resource_map() const
{
    ResourceMap rm;
    if (pimpl_->kv_cache_manager) {
        rm.vulkan_buffer_memory = pimpl_->kv_cache_manager->allocated_bytes();
        rm.vulkan_image_memory  = 0;
    }
    {
        std::lock_guard<std::mutex> lock(pimpl_->queue_mutex);
        rm.active_requests  = static_cast<uint32_t>(pimpl_->active_requests.size());
        rm.queued_requests  = static_cast<uint32_t>(pimpl_->pending_queue.size());
    }
    return rm;
}

GpuResource VkExecutor::get_gpu_resource() const
{
    GpuResource gr{};
    if (pimpl_->vulkan_runtime) {
        auto ctx = pimpl_->vulkan_runtime->getContext();
        if (ctx) {
            const auto& info = ctx->getDeviceInfo();

            // Query VRAM via Vulkan (mirrors cuMemGetInfo pattern)
            VkPhysicalDeviceMemoryProperties mem_props{};
            vkGetPhysicalDeviceMemoryProperties(
                ctx->getPhysicalDevice(), &mem_props);

            VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_props{};
            budget_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
            // Try to query heap budget for total memory
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &budget_props;
            vkGetPhysicalDeviceProperties2(ctx->getPhysicalDevice(), &props2);

            for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
                if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                    gr.total_memory_bytes += mem_props.memoryHeaps[i].size;
                    gr.free_memory_bytes  += budget_props.heapUsage[i];
                    break;
                }
            }

            gr.device_name       = info.deviceName;
            gr.compute_cap_major = info.apiVersion >> 12;
            gr.compute_cap_minor = (info.apiVersion >> 8) & 0x7;
        }
    }
    return gr;
}

void VkExecutor::gc()
{
    std::lock_guard<std::mutex> lock(pimpl_->queue_mutex);
    pimpl_->active_requests.erase(
        std::remove_if(pimpl_->active_requests.begin(), pimpl_->active_requests.end(),
            [](Request* r) {
                return r->status == RequestStatus::kFinished ||
                       r->status == RequestStatus::kError ||
                       r->status == RequestStatus::kCancelled;
            }),
        pimpl_->active_requests.end());
}

// --- Config / introspection ---

std::string VkExecutor::get_config_str() const
{
    return pimpl_->config_str;
}

std::vector<std::string> VkExecutor::get_supported_dtypes() const
{
    return pimpl_->supported_dtypes;
}

std::vector<std::string> VkExecutor::get_loaded_adapters() const
{
    return pimpl_->loaded_adapters;
}

// --- Callbacks ---

void VkExecutor::set_request_callback(RequestCallback cb) { pimpl_->req_cb = std::move(cb); }
void VkExecutor::set_batch_callback(BatchCallback cb)     { pimpl_->batch_cb = std::move(cb); }
void VkExecutor::set_error_callback(ErrorCallback cb)     { pimpl_->err_cb = std::move(cb); }
void VkExecutor::set_log_callback(LogCallback cb)         { pimpl_->log_cb = std::move(cb); }

void VkExecutor::set_log_level(std::string_view level)
{
    pimpl_->log_level = std::string(level);
}

std::string VkExecutor::get_debug_dump() const
{
    std::string dump;
    dump += "=== VkExecutor Debug Dump ===\n";
    dump += "Running: " + std::string(pimpl_->running ? "true" : "false") + "\n";
    dump += "Paused: " + std::string(pimpl_->paused ? "true" : "false") + "\n";
    dump += "Batch size: " + std::to_string(get_batch_size()) + "\n";
    dump += "Active requests: " + std::to_string(get_num_running_requests()) + "\n";
    if (pimpl_->kv_cache_manager) {
        dump += "KV cache bytes: " + std::to_string(pimpl_->kv_cache_manager->allocated_bytes()) + "\n";
    }
    return dump;
}

// --- Sub-component accessors ---

const Scheduler& VkExecutor::scheduler() const            { return *pimpl_->scheduler; }
const TorchSampler& VkExecutor::sampler() const         { return *pimpl_->sampler; }
const KvCacheManager& VkExecutor::kv_cache_manager() const { return *pimpl_->kv_cache_manager; }

} // namespace vulkan_executor
TRTLLM_NAMESPACE_END