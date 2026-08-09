import os
import platform

_on_windows_stub = (platform.system() == "Windows"
                    and os.environ.get("TLLM_VULKAN_BACKEND", "0") == "1")

if _on_windows_stub:
    # Windows Vulkan path: use lazy __getattr__ to avoid eager imports
    # that trigger circular import chains through executor -> llmapi -> llm -> executor
    pass
else:
    from .._torch.async_llm import AsyncLLM
    from ..conversation_params import ConversationParams
    from ..disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
    from ..executor import CompletionOutput, LoRARequest, RequestError
    from ..sampling_params import GuidedDecodingParams, SamplingParams
    from ..scheduling_params import SchedulingParams
    from .llm import LLM, RequestOutput
    # yapf: disable
    from .llm_args import (AttentionDpConfig, AutoDecodingConfig, BatchingType,
                           BlockReuseConfig, CacheTransceiverConfig, CalibConfig,
                           CapacitySchedulerPolicy, ContextChunkingPolicy,
                           CudaGraphConfig, DecodeCudaGraphConfig,
                           DeepSeekSparseAttentionConfig,
                           DeepSeekV4SparseAttentionConfig, DFlashDecodingConfig,
                           DraftTargetDecodingConfig, DSparkDecodingConfig,
                           DynamicBatchConfig, Eagle3DecodingConfig,
                           EagleDecodingConfig, EncodeCudaGraphConfig,
                           ExtendedRuntimePerfKnobConfig, KvCacheConfig, LlmArgs,
                           LookaheadDecodingConfig, MambaStateConfig,
                           MedusaDecodingConfig, MiniMaxM3SparseAttentionConfig,
                           MoeConfig, MTPDecodingConfig, MultimodalConfig,
                           NGramDecodingConfig, PARDDecodingConfig,
                           PrometheusMetricsConfig, ReorderRequestPolicyConfig,
                           RocketSparseAttentionConfig, SADecodingConfig,
                           SAEnhancerConfig, SaveHiddenStatesDecodingConfig,
                           SchedulerConfig, SkipSoftmaxAttentionConfig,
                           TorchCompileConfig, TorchLlmArgs,
                           TriAttentionKvCacheCompressionConfig,
                           UserProvidedDecodingConfig)
    from .llm_utils import KvCacheRetentionConfig, QuantAlgo, QuantConfig
    from .mm_encoder import MultimodalEncoder
    from .mpi_session import MpiCommSession
    from .thinking_budget import (ThinkingBudgetLogitsProcessor,
                                   add_thinking_budget_logits_processor)
    # yapf: enable

# Lazy import mappings for Vulkan/Windows path
_LAZY_NAMES = {
    "AsyncLLM": lambda: __import__(
        "tensorrt_llm._torch.async_llm", fromlist=["AsyncLLM"]).AsyncLLM,
    "LLM": lambda: __import__(
        "tensorrt_llm.llmapi.llm", fromlist=["LLM"]).LLM,
    "RequestOutput": lambda: __import__(
        "tensorrt_llm.llmapi.llm", fromlist=["RequestOutput"]).RequestOutput,
    "MultimodalEncoder": lambda: __import__(
        "tensorrt_llm.llmapi.mm_encoder", fromlist=["MultimodalEncoder"]).MultimodalEncoder,
    "MpiCommSession": lambda: __import__(
        "tensorrt_llm.llmapi.mpi_session", fromlist=["MpiCommSession"]).MpiCommSession,
    "ThinkingBudgetLogitsProcessor": lambda: __import__(
        "tensorrt_llm.llmapi.thinking_budget", fromlist=["ThinkingBudgetLogitsProcessor"]).ThinkingBudgetLogitsProcessor,
    "add_thinking_budget_logits_processor": lambda: __import__(
        "tensorrt_llm.llmapi.thinking_budget", fromlist=["add_thinking_budget_logits_processor"]).add_thinking_budget_logits_processor,
    "ConversationParams": lambda: __import__(
        "tensorrt_llm.conversation_params", fromlist=["ConversationParams"]).ConversationParams,
    "DisaggregatedParams": lambda: __import__(
        "tensorrt_llm.disaggregated_params", fromlist=["DisaggregatedParams"]).DisaggregatedParams,
    "DisaggScheduleStyle": lambda: __import__(
        "tensorrt_llm.disaggregated_params", fromlist=["DisaggScheduleStyle"]).DisaggScheduleStyle,
    "CompletionOutput": lambda: __import__(
        "tensorrt_llm.executor", fromlist=["CompletionOutput"]).CompletionOutput,
    "LoRARequest": lambda: __import__(
        "tensorrt_llm.executor", fromlist=["LoRARequest"]).LoRARequest,
    "RequestError": lambda: __import__(
        "tensorrt_llm.executor", fromlist=["RequestError"]).RequestError,
    "GuidedDecodingParams": lambda: __import__(
        "tensorrt_llm.sampling_params", fromlist=["GuidedDecodingParams"]).GuidedDecodingParams,
    "SamplingParams": lambda: __import__(
        "tensorrt_llm.sampling_params", fromlist=["SamplingParams"]).SamplingParams,
    "SchedulingParams": lambda: __import__(
        "tensorrt_llm.scheduling_params", fromlist=["SchedulingParams"]).SchedulingParams,
    "KvCacheRetentionConfig": lambda: __import__(
        "tensorrt_llm.llmapi.llm_utils", fromlist=["KvCacheRetentionConfig"]).KvCacheRetentionConfig,
    "QuantAlgo": lambda: __import__(
        "tensorrt_llm.llmapi.llm_utils", fromlist=["QuantAlgo"]).QuantAlgo,
    "QuantConfig": lambda: __import__(
        "tensorrt_llm.llmapi.llm_utils", fromlist=["QuantConfig"]).QuantConfig,
}

# yapf: disable
_pybind_names = [
    "AttentionDpConfig", "AutoDecodingConfig", "BatchingType",
    "BlockReuseConfig", "CacheTransceiverConfig", "CalibConfig",
    "CapacitySchedulerPolicy", "ContextChunkingPolicy",
    "CudaGraphConfig", "DecodeCudaGraphConfig",
    "DeepSeekSparseAttentionConfig",
    "DeepSeekV4SparseAttentionConfig", "DFlashDecodingConfig",
    "DraftTargetDecodingConfig", "DSparkDecodingConfig",
    "DynamicBatchConfig", "Eagle3DecodingConfig",
    "EagleDecodingConfig", "EncodeCudaGraphConfig",
    "ExtendedRuntimePerfKnobConfig", "KvCacheConfig", "LlmArgs",
    "LookaheadDecodingConfig", "MambaStateConfig",
    "MedusaDecodingConfig", "MiniMaxM3SparseAttentionConfig",
    "MoeConfig", "MTPDecodingConfig", "MultimodalConfig",
    "NGramDecodingConfig", "PARDDecodingConfig",
    "PrometheusMetricsConfig", "ReorderRequestPolicyConfig",
    "RocketSparseAttentionConfig", "SADecodingConfig",
    "SAEnhancerConfig", "SaveHiddenStatesDecodingConfig",
    "SchedulerConfig", "SkipSoftmaxAttentionConfig",
    "TorchCompileConfig", "TorchLlmArgs",
    "TriAttentionKvCacheCompressionConfig",
    "UserProvidedDecodingConfig",
]
# yapf: enable

for _name in _pybind_names:
    _LAZY_NAMES[_name] = lambda n=_name: __import__(
        "tensorrt_llm.llmapi.llm_args", fromlist=[n]).__getattribute__(n)


def __getattr__(name):
    if _on_windows_stub and name in _LAZY_NAMES:
        return _LAZY_NAMES[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'LLM',
    'AsyncLLM',
    'MultimodalEncoder',
    'CompletionOutput',
    'RequestOutput',
    'GuidedDecodingParams',
    'SamplingParams',
    'DisaggregatedParams',
    'ConversationParams',
    'DisaggScheduleStyle',
    'BlockReuseConfig',
    'KvCacheConfig',
    'MambaStateConfig',
    'KvCacheRetentionConfig',
    'CudaGraphConfig',
    'DecodeCudaGraphConfig',
    'EncodeCudaGraphConfig',
    'MoeConfig',
    'LookaheadDecodingConfig',
    'MedusaDecodingConfig',
    'EagleDecodingConfig',
    'Eagle3DecodingConfig',
    'MTPDecodingConfig',
    'SchedulerConfig',
    'CapacitySchedulerPolicy',
    'QuantConfig',
    'QuantAlgo',
    'CalibConfig',
    'RequestError',
    'MpiCommSession',
    'ExtendedRuntimePerfKnobConfig',
    'BatchingType',
    'ContextChunkingPolicy',
    'DynamicBatchConfig',
    'CacheTransceiverConfig',
    'NGramDecodingConfig',
    'PARDDecodingConfig',
    'DFlashDecodingConfig',
    'DSparkDecodingConfig',
    'SADecodingConfig',
    'SAEnhancerConfig',
    'UserProvidedDecodingConfig',
    'TorchCompileConfig',
    'DraftTargetDecodingConfig',
    'LlmArgs',
    'TorchLlmArgs',
    'AutoDecodingConfig',
    'AttentionDpConfig',
    'LoRARequest',
    'SaveHiddenStatesDecodingConfig',
    'RocketSparseAttentionConfig',
    'ReorderRequestPolicyConfig',
    'DeepSeekSparseAttentionConfig',
    'DeepSeekV4SparseAttentionConfig',
    'MiniMaxM3SparseAttentionConfig',
    'SchedulingParams',
    'SkipSoftmaxAttentionConfig',
    'TriAttentionKvCacheCompressionConfig',
    'PrometheusMetricsConfig',
    'ThinkingBudgetLogitsProcessor',
    'add_thinking_budget_logits_processor',
    'MultimodalConfig',
]
