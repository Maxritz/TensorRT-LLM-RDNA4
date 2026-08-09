# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
Stub ``tensorrt_llm.bindings`` module for Windows development.

The real ``bindings`` is a nanobind C++ extension (``bindings.*.so`` / ``.pyd``)
that wraps the TensorRT-LLM C++ runtime.  On Windows the C++ build is not yet
available, so this pure-Python stub provides the API surface required for
importing the _torch PyTorch backend.

Production use should build the real C++ bindings via
``python scripts/build_wheel.py``.
"""

import ctypes
import os
import sys
import time
from enum import IntEnum
from typing import Any


class DataType(IntEnum):
    FLOAT = 0
    HALF = 1
    BF16 = 2
    INT8 = 3
    INT32 = 4
    INT64 = 5
    FP8 = 6
    UINT8 = 7
    INT4 = 8
    UINT4 = 9
    BOOL = 10
    NVFP4 = 11


class LayerType(IntEnum):
    DEFAULT = 0
    ATTENTION = 1
    MLP = 2
    EMBEDDING = 3
    LAYERNORM = 4
    ROWMUL = 5
    COLMUL = 6
    RMSNORM = 7
    GELU = 8
    SILU = 9
    RELU = 10
    GATED = 11
    LAYERNORM_LINEAR = 12


class BuildInfo:
    ENABLE_MULTI_DEVICE = False
    ENABLE_MULTI_DEVICE_SMALL = False
    CUDA_GRAPH_ENABLED = False
    TENSORRT_HIPP_SE_SUPPORTED = False
    TENSORRT_HIPP_SUPPORTED = False


class WorldConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_json_file(cls, path):
        import json
        with open(path) as f:
            return cls(**json.load(f))

    @staticmethod
    def is_tensorrt():
        return False

    @staticmethod
    def is_pytorch():
        return True


class LlmRequest:
    pass


class LlmRequestState(IntEnum):
    UNKNOWN = 0
    ENCODER_INIT = 1
    DISAGG_CONTEXT_WAIT_SCHEDULER = 7
    DISAGG_GENERATION_INIT = 8
    DISAGG_GENERATION_TRANS_IN_PROGRESS = 9
    CONTEXT_INIT = 10
    DISAGG_CONTEXT_INIT_AND_TRANS = 11
    DISAGG_GENERATION_TRANS_COMPLETE = 12
    GENERATION_IN_PROGRESS = 13
    GENERATION_TO_COMPLETE = 14
    GENERATION_COMPLETE = 20
    DISAGG_CONTEXT_TRANS_IN_PROGRESS = 21
    DISAGG_CONTEXT_COMPLETE = 22
    DISAGG_GENERATION_WAIT_TOKENS = 23
    DISAGG_TRANS_ERROR = -1


class SamplingConfig:
    pass


class SamplingConfigVector:
    def __init__(self, *args, **kwargs):
        self._items = []

    def push_back(self, item):
        self._items.append(item)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def __len__(self):
        return len(self._items)


class LoraModule:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class ModelConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class CudaStream:
    pass


def make_sampling_config(*args, **kwargs):
    """Stub: create a SamplingConfig from keyword arguments."""
    return kwargs


class MpiComm:
    _initialized = False

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def local_init(cls):
        if cls._initialized:
            return
        cls._initialized = True

    @classmethod
    def world_size(cls):
        return 1

    @classmethod
    def rank(cls):
        return 0

    def barrier(self):
        pass

    def bcast(self, data, root=0):
        return data


class BufferManager:
    pass


class CudaEvent:
    def __init__(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass

    def synchronize(self, *args, **kwargs):
        pass

    def query(self, *args, **kwargs):
        return True

    def elapsed_time(self, other, *args, **kwargs):
        return 0.0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class GptDecoderBatched:
    pass


# Internal module (used by distributed ops)
class _Internal:
    @staticmethod
    def all_reduce_sum(tensor, group=None):
        return tensor

    @staticmethod
    def all_reduce_min(tensor, group=None):
        return tensor

    @staticmethod
    def all_reduce_max(tensor, group=None):
        return tensor

    @staticmethod
    def all_gather(tensor, group=None, dim=0):
        return [tensor]

    @staticmethod
    def broadcast(tensor, src, group=None):
        return tensor

    @staticmethod
    def barrier(group=None):
        pass

    def __getattr__(self, name):
        import importlib
        try:
            return importlib.import_module(
                f"tensorrt_llm.bindings.internal.{name}")
        except ImportError:
            return _Stub()


internal = _Internal()


# executor submodule stub
class _ExecutorSubmodule:
    """Stub for the bindings.executor module on Windows.

    Mirrors the classes defined in executor.py but provides __getattr__ for any
    attribute not explicitly defined, so that code accessing executor.X for
    types we haven't stubbed yet gets a no-op _Stub() instead of AttributeError.
    """

    class Executor:
        pass

    class LogitsProcessor:
        pass

    class Request:
        pass

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

    class SpecDecMode:
        DISABLE = 0
        DRAFT_IN_TARGET = 1
        DRAFT_SEPARATE = 2

    class FinishReason:
        FINISH_REASON_UNKNOWN = 0

    class RequestType:
        REQUEST_TYPE_CONTEXT_AND_GENERATION = 0
        REQUEST_TYPE_CONTEXT_ONLY = 1
        REQUEST_TYPE_GENERATION_ONLY = 2

    class DecodingConfig:
        pass

    class DecodingMode:
        pass

    class SpecDecodingStats:
        pass

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

    class CacheTransceiverConfig:
        pass

    class DynamicBatchConfig:
        pass

    class ExecutorConfig:
        pass

    class ExtendedRuntimePerfKnobConfig:
        pass

    class KvCacheConfig:
        pass

    class PeftCacheConfig:
        pass

    class SchedulerConfig:
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
                maxDraftTokens = (windowSize - 1 + verificationSetSize) * (ngramSize - 1)
            else:
                maxDraftTokens = (ngramSize - 2) + (windowSize - 1 + verificationSetSize) * (ngramSize - 1)
            maxPathLen = ngramSize
            maxDecodingTokens = maxDraftTokens + 1
            maxDraftPathLen = ngramSize - 1
            return (maxDecodingTokens, maxPathLen, maxDraftTokens, maxDraftPathLen)

    def __getattr__(self, name):
        return _Stub()


executor = _ExecutorSubmodule()


def ipc_nvls_supported(device_id: int = 0) -> bool:
    return False


def steady_clock_now() -> int:
    return time.monotonic_ns()


def trtllm_supports_cublaslt_heuristics() -> bool:
    return False


class _Stub:
    """Generic stub for any unbound name in the bindings module."""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return _Stub()

    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self

    def __iter__(self):
        return iter([])

    def __int__(self):
        return 0

    def __mro_entries__(self, bases):
        return (object,)


def __getattr__(name):
    return _Stub()
