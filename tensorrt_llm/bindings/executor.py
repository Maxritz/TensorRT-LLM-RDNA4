# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stub for tensorrt_llm.bindings.executor — the C++ executor API.

All classes and enums are no-op stubs so that Python imports succeed on
Windows without the C++ bindings build.
"""

import ctypes
import inspect as _inspect
import time
from enum import IntEnum

_empty_sig = _inspect.Signature()


class FinishReason(IntEnum):
    FINISH_REASON_UNKNOWN = 0
    FINISH_REASON_END_ID = 1
    FINISH_REASON_LENGTH = 2
    FINISH_REASON_STOP = 3
    FINISH_REASON_FILTER = 4


class BatchingType(IntEnum):
    STATIC = 0
    INFLIGHT = 1


class CapacitySchedulerPolicy(IntEnum):
    GUARANTEED_NO_EVICT = 0
    MAX_UTILIZATION = 1
    STATIC_BATCH = 2


class ContextChunkingPolicy(IntEnum):
    FIRST_COME_FIRST_SERVED = 0
    EQUAL_PROGRESS = 1
    FORCE_CHUNK = 2


class CacheTransceiverBackendType(IntEnum):
    DISABLE = 0
    UCX = 1
    NIXL = 2
    MPI = 3
    UCX_TCP = 4


class DecodingConfig:
    pass


class DecodingMode(IntEnum):
    UNKNOWN = 0
    CLIENT = 1
    CONTEXT = 2
    GENERATION = 3
    CONTEXT_AND_GENERATION = 4


class SpecDecMode(IntEnum):
    DISABLE = 0
    DRAFT_IN_TARGET = 1
    DRAFT_SEPARATE = 2


class RequestType(IntEnum):
    REQUEST_TYPE_CONTEXT_AND_GENERATION = 0
    REQUEST_TYPE_CONTEXT_ONLY = 1
    REQUEST_TYPE_GENERATION_ONLY = 2


class AdditionalModelOutput:
    def __init__(self, name=None, gather_context=False, **kwargs):
        self.name = name
        self.gather_context = gather_context

    def __getattr__(self, name):
        # Support arbitrary field access for pybind compatibility
        raise AttributeError(name)


class SamplingConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def beam_width(self):
        return 1

    @property
    def frequency_penalty(self):
        return 0.0

    @property
    def presence_penalty(self):
        return 0.0

    @property
    def min_p(self):
        return 0.0

    @property
    def repetition_penalty(self):
        return 1.0

    @property
    def temperature(self):
        return 1.0

    @property
    def top_p(self):
        return 1.0

    @property
    def top_k(self):
        return 0

    @property
    def random_seed(self):
        return 0

    @property
    def exclude_input_from_output(self):
        return True


class OutputConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def logprobs(self):
        return 0

    @property
    def prompt_logprobs(self):
        return 0


class PromptTemplate:
    pass


class PromptTuningTask:
    pass


class LoRAModule:
    pass


class PromptTask:
    pass


class Request:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Result:
    pass


class GenerationOutput:
    pass


class SequenceLogprobs:
    pass


class SequenceOutput:
    pass


class TokenGenerator:
    pass


class OutputSequence:
    pass


class OutputLogprobs:
    pass


class OutputCumLogprobs:
    pass


class OutputTokenLogprobs:
    pass


class OutputLogitInfo:
    pass


class ReturnValue:
    pass


class ReturnValueBatchContext:
    pass


class ReturnValueBatch:
    pass


class LogitLookup:
    pass


class SpeculativeDecodingMode:
    pass


class SpeculativeDecoding:
    pass


class Executor:
    class Config:
        pass


class LogitsProcessor:
    pass


class LookaheadDecodingConfig:

    def __init__(self, max_window_size=None, max_ngram_size=None,
                 max_verification_set_size=None):
        if max_window_size is not None:
            self.max_window_size = max_window_size
        if max_ngram_size is not None:
            self.max_ngram_size = max_ngram_size
        if max_verification_set_size is not None:
            self.max_verification_set_size = max_verification_set_size

    @staticmethod
    def get_default_lookahead_decoding_window():
        return 4

    @staticmethod
    def get_default_lookahead_decoding_ngram():
        return 3

    @staticmethod
    def get_default_lookahead_decoding_verification_set():
        return 4

    @staticmethod
    def calculate_speculative_resource_tuple(windowSize, ngramSize, verificationSetSize):
        if ngramSize == 1:
            maxDraftTokens = 0 + (windowSize - 1 + verificationSetSize) * (ngramSize - 1)
        else:
            maxDraftTokens = (ngramSize - 2) + (windowSize - 1 + verificationSetSize) * (ngramSize - 1)
        maxPathLen = ngramSize
        maxDecodingTokens = maxDraftTokens + 1
        maxDraftPathLen = ngramSize - 1
        return (maxDecodingTokens, maxPathLen, maxDraftTokens, maxDraftPathLen)


class GuidedDecodingParams:
    class GuideType(IntEnum):
        JSON = 0
        JSON_SCHEMA = 1
        REGEX = 2
        EBNF_GRAMMAR = 3
        STRUCTURAL_TAG = 4


class KVCacheConfig:
    pass


class PagedKVCacheConfig:
    pass


class LoraTask:
    pass


class ContextPhaseParams:
    pass


class Metric:
    pass


class StringId:
    pass


class CacheType:
    CONTEXT = 0
    GENERATION = 1


class CacheTransceiverConfig:
    pass


class DynamicBatchConfig:
    pass


class ExecutorConfig:
    pass


class ExtendedRuntimePerfKnobConfig:
    pass


class PeftCacheConfig:
    pass


class SchedulerConfig:
    pass


class _Stub:
    """Generic stub class for any unbound name."""

    __signature__ = _empty_sig

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return _Stub()

    def __or__(self, other):
        return other

    def __ror__(self, other):
        return other

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False

    def __int__(self):
        return 0

    def __index__(self):
        return 0

    def __contains__(self, key):
        return False

    def __getitem__(self, key):
        return _Stub()

    def keys(self):
        return []

    def values(self):
        return []

    def items(self):
        return []


def __getattr__(name):
    return _Stub()


__all__ = [
    "Executor",
    "LogitsProcessor",
    "Request",
    "Result",
    "GenerationOutput",
    "SequenceLogprobs",
    "SequenceOutput",
    "TokenGenerator",
    "OutputSequence",
    "OutputLogprobs",
    "OutputCumLogprobs",
    "OutputTokenLogprobs",
    "OutputLogitInfo",
    "ReturnValue",
    "ReturnValueBatchContext",
    "ReturnValueBatch",
    "LogitLookup",
    "SpeculativeDecodingMode",
    "SpeculativeDecoding",
    "SpecDecMode",
    "FinishReason",
    "BatchingType",
    "CapacitySchedulerPolicy",
    "ContextChunkingPolicy",
    "CacheTransceiverBackendType",
    "CacheTransceiverConfig",
    "DynamicBatchConfig",
    "ExecutorConfig",
    "ExtendedRuntimePerfKnobConfig",
    "KvCacheConfig",
    "PeftCacheConfig",
    "SchedulerConfig",
    "DecodingConfig",
    "DecodingMode",
    "SpecDecodingStats",
    "RequestType",
    "AdditionalModelOutput",
    "SamplingConfig",
    "OutputConfig",
    "PromptTemplate",
    "PromptTuningTask",
    "LoRAModule",
    "PromptTask",
    "LookaheadDecodingConfig",
    "GuidedDecodingParams",
    "PagedKVCacheConfig",
    "LoraTask",
    "ContextPhaseParams",
    "Metric",
    "StringId",
    "CacheType",
]
