# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stub for tensorrt_llm.bindings.internal.runtime."""

import os


class DecoderState:
    def __init__(self, *args, **kwargs):
        pass


class TaskLayerModuleConfig:
    def __init__(self, *args, **kwargs):
        pass


class McastGPUBuffer:
    pass


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


class GptDecoderBatched:
    pass


def delay_kernel(*args, **kwargs):
    pass


def get_max_seq_len(*args, **kwargs):
    return 0


def get_max_batch_size(*args, **kwargs):
    return 0


def cuda_stream_synchronize(*args, **kwargs):
    pass


def init(*args, **kwargs):
    pass


__all__ = [
    "DecoderState",
    "TaskLayerModuleConfig",
    "McastGPUBuffer",
    "BufferManager",
    "CudaEvent",
    "GptDecoderBatched",
    "delay_kernel",
    "get_max_seq_len",
    "get_max_batch_size",
    "cuda_stream_synchronize",
    "init",
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
