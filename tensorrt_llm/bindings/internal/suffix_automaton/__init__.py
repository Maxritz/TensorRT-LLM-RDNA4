# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stub for tensorrt_llm.bindings.internal.suffix_automaton."""

import torch


class SuffixAutomatonManager:
    def __init__(self, *args, **kwargs):
        pass

    def insert(self, *args, **kwargs):
        pass

    def query(self, *args, **kwargs):
        return None


def suffix_automaton_create(*args, **kwargs):
    return SuffixAutomatonManager()


__all__ = ["SuffixAutomatonManager", "suffix_automaton_create"]
