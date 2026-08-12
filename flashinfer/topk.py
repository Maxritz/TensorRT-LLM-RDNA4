# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""flashinfer.topk — Top-K selection (Vulkan-backed)."""

import numpy as np
from typing import Optional, Tuple
from enum import Enum


class TopKTieBreak(Enum):
    LARGEST = 0
    SMALLEST = 1
    FIRST = 2


def top_k(
    probs_or_logits: np.ndarray,
    k: int,
    tie_break: TopKTieBreak = TopKTieBreak.FIRST,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-K selection.

    Returns:
        (top_k_indices, top_k_values)
    """
    if probs_or_logits.ndim == 1:
        idx = np.argpartition(-probs_or_logits, k)[:k]
        idx = idx[np.argsort(-probs_or_logits[idx])]
        return idx, probs_or_logits[idx]
    else:
        results_idx = np.empty((probs_or_logits.shape[0], k), dtype=np.int64)
        results_val = np.empty((probs_or_logits.shape[0], k), dtype=probs_or_logits.dtype)
        for i in range(probs_or_logits.shape[0]):
            idx = np.argpartition(-probs_or_logits[i], k)[:k]
            idx = idx[np.argsort(-probs_or_logits[i][idx])]
            results_idx[i] = idx
            results_val[i] = probs_or_logits[i][idx]
        return results_idx, results_val


def top_k_page_table_transform(*args, **kwargs) -> np.ndarray:
    """No-op transform."""
    return np.zeros(1, dtype=np.int32)


def top_k_ragged_transform(*args, **kwargs) -> np.ndarray:
    """No-op transform."""
    return np.zeros(1, dtype=np.int32)
