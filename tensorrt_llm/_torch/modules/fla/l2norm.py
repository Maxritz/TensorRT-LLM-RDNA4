# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/l2norm.py
# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np


def l2norm_fwd(x: np.ndarray,
               eps: float = 1e-6,
               output_dtype: Optional[np.dtype] = None) -> np.ndarray:
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if output_dtype is None:
        y = np.empty_like(x)
    else:
        y = np.empty(x.shape, dtype=output_dtype)
    x_f = x.astype(np.float32)
    var = np.sum(x_f * x_f, axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    y = (x_f * rstd)
    if output_dtype is not None:
        y = y.astype(output_dtype)
    else:
        y = y.astype(x.dtype)
    return y.reshape(x_shape_og)


def l2norm(x: np.ndarray,
           eps: float = 1e-6,
           output_dtype: Optional[np.dtype] = None) -> np.ndarray:
    return l2norm_fwd(x, eps, output_dtype)


l2_norm = l2norm


def l2norm_torch(x, eps=1e-6, output_dtype=None):
    """Torch-compatible wrapper for l2norm — delegates to numpy."""
    np_x = np.asarray(x)
    if hasattr(x, 'cpu'):
        np_x = x.cpu().numpy()
    out = l2norm_fwd(np_x, eps, output_dtype)
    if hasattr(x, 'device'):
        import torch as _t
        return _t.from_numpy(out).to(x.device)
    return out