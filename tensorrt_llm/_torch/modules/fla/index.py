# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/index.py
# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np


def prepare_lens(cu_seqlens: np.ndarray) -> np.ndarray:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(cu_seqlens: np.ndarray,
                      chunk_size: int) -> np.ndarray:
    lens = prepare_lens(cu_seqlens)
    nt = (lens + chunk_size - 1) // chunk_size
    indices = np.concatenate([np.arange(n) for n in nt.tolist()])
    return np.stack([np.cumsum(indices == 0) - 1, indices], axis=1).astype(cu_seqlens.dtype)


def prepare_chunk_offsets(cu_seqlens: np.ndarray,
                      chunk_size: int) -> np.ndarray:
    lens = prepare_lens(cu_seqlens)
    nt = (lens + chunk_size - 1) // chunk_size
    return np.concatenate([np.array([0], dtype=cu_seqlens.dtype), nt]).cumsum(-1)