# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stub for tensorrt_llm.bindings.internal.batch_manager.

This is a C++ nanobind extension in the real package. On Windows without
the C++ build, these stubs allow Python imports to succeed.
"""

import os
from enum import IntEnum


class CacheType(IntEnum):
    CONTEXT = 0
    GENERATION = 1


class LinearCacheType(IntEnum):
    CONTEXT = 0
    GENERATION = 1


class LlmRequestType(IntEnum):
    LLMREQUEST_TYPE_CONTEXT_ONLY = 0
    LLMREQUEST_TYPE_GENERATION_ONLY = 1
    LLMREQUEST_TYPE_CONTEXT_AND_GENERATION = 2


class LlmRequest:
    pass


class Request:
    pass


class DecoderInputBuffers:
    pass


class KvCacheIterationStats:
    pass


class KvCacheStats:
    pass


class BlockKey:
    pass


class BlockKeyHasher:
    pass


class kv_cache_manager_v2:
    pass


class KvCacheManagerV2:
    pass


def add_new_tokens_to_requests(*args, **kwargs):
    pass


def make_decoding_batch_input(*args, **kwargs):
    pass


__all__ = [
    "CacheType",
    "LinearCacheType",
    "LlmRequestType",
    "LlmRequest",
    "Request",
    "DecoderInputBuffers",
    "KvCacheIterationStats",
    "KvCacheStats",
    "BlockKey",
    "BlockKeyHasher",
    "kv_cache_manager_v2",
    "KvCacheManagerV2",
    "add_new_tokens_to_requests",
    "make_decoding_batch_input",
]


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return _Stub()


def __getattr__(name):
    return _Stub()
