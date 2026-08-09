# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stub for kv_cache_manager_v2_utils — C++ extension on other platforms."""

class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return _Stub()


class DiskAddress(_Stub):
    pass


class DiskToDiskTask(_Stub):
    pass


class DiskToHostTask(_Stub):
    pass


class HostToDiskTask(_Stub):
    pass


class MemToMemTask(_Stub):
    pass


class IndexMapper(_Stub):
    pass


def copy_device_to_device(*args, **kwargs):
    pass


def copy_device_to_host(*args, **kwargs):
    pass


def copy_disk_to_disk(*args, **kwargs):
    pass


def copy_disk_to_host(*args, **kwargs):
    pass


def copy_host_to_device(*args, **kwargs):
    pass


def copy_host_to_disk(*args, **kwargs):
    pass


def copy_host_to_host(*args, **kwargs):
    pass


def copy_batch_block_offsets_to_device(*args, **kwargs):
    pass


def __getattr__(name):
    return _Stub()


__all__ = [
    "DiskAddress",
    "DiskToDiskTask",
    "DiskToHostTask",
    "HostToDiskTask",
    "MemToMemTask",
    "IndexMapper",
    "copy_device_to_device",
    "copy_device_to_host",
    "copy_disk_to_disk",
    "copy_disk_to_host",
    "copy_host_to_device",
    "copy_host_to_disk",
    "copy_host_to_host",
    "copy_batch_block_offsets_to_device",
]
