# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/layernorm_gated.py
# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np


def np_silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))


def rms_norm_ref(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    upcast=True,
):
    np_x = np.asarray(x, dtype=np.float32) if upcast else np.asarray(x)
    np_weight = np.asarray(weight, dtype=np.float32) if upcast else np.asarray(weight)
    np_bias = np.asarray(bias, dtype=np.float32) if bias is not None and upcast else (np.asarray(bias) if bias is not None else None)
    np_z = np.asarray(z, dtype=np.float32) if z is not None and upcast else (np.asarray(z) if z is not None else None)

    if np_z is not None and not norm_before_gate:
        np_x = np_x * np_silu(np_z)

    if group_size is None:
        var = np.mean(np_x.square(), axis=-1, keepdims=True)
        rstd = 1.0 / np.sqrt(var + eps)
        out = np_x * rstd * np_weight
        if np_bias is not None:
            out = out + np_bias
    else:
        x_group = np_x.reshape(*np_x.shape[:-1], -1, group_size)
        gdim = np_x.shape[-1] // group_size
        x_group = np_x[..., :gdim * group_size].reshape(*np_x.shape[:-1], gdim, group_size)
        var = np.mean(x_group.square(), axis=-1, keepdims=True)
        rstd = 1.0 / np.sqrt(var + eps)
        out = (x_group * rstd)
        out = out.reshape(np_x.shape) if out.shape[0] > 1 else np_x.reshape(-1, group_size * gdim) if np_x.ndim == 1 else np_x
        out = out * np_weight
        if np_bias is not None:
            out = out + np_bias

    if np_z is not None and norm_before_gate:
        out = out * np_silu(np_z)

    original_dtype = np.asarray(x).dtype
    return out.astype(original_dtype)


def layernorm_fn(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    if is_rms_norm:
        return rms_norm_ref(x, weight, bias, z, eps, group_size, norm_before_gate)

    np_x = np.asarray(x, dtype=np.float32)
    np_weight = np.asarray(weight, dtype=np.float32)
    np_bias = np.asarray(bias, dtype=np.float32) if bias is not None else None
    np_z = np.asarray(z, dtype=np.float32) if z is not None else None

    if np_z is not None and not norm_before_gate:
        np_x = np_x * np_silu(np_z)

    x_shape_og = np_x.shape
    np_x = np_x.reshape(-1, np_x.shape[-1])
    if np_z is not None:
        np_z = np_z.reshape(-1, np_z.shape[-1])
    np_weight = np_weight.reshape(-1) if np_weight.ndim > 1 else np_weight
    if np_bias is not None:
        np_bias = np_bias.reshape(-1) if np_bias.ndim > 1 else np_bias

    M, N = np_x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size

    if not is_rms_norm:
        if group_size == N:
            mean = np.mean(np_x, axis=-1, keepdims=True)
            var = np.var(np_x, axis=-1, keepdims=True)
            xbar = np_x - mean
            rstd = 1.0 / np.sqrt(var + eps)
            x_hat = xbar * rstd
        else:
            x_group = np_x.reshape(M, ngroups, group_size)
            mean = np.mean(x_group, axis=-1, keepdims=True)
            var = np.var(x_group, axis=-1, keepdims=True)
            xbar = x_group - mean
            rstd = 1.0 / np.sqrt(var + eps)
            x_hat = (xbar * rstd).reshape(M, N)
        y = x_hat * np_weight + (np_bias if np_bias is not None else 0)
    else:
        x_group = np_x.reshape(M, ngroups, group_size)
        var = np.mean(x_group.square(), axis=-1, keepdims=True)
        rstd = 1.0 / np.sqrt(var + eps)
        y = (x_group * rstd).reshape(M, N) * np_weight + (np_bias if np_bias is not None else 0)

    if np_z is not None and norm_before_gate:
        y = y * np_silu(np_z)

    original_dtype = np.asarray(x).dtype
    return y.astype(original_dtype).reshape(x_shape_og)


def rmsnorm_fn(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True):
    return layernorm_fn(x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm=True)


def layernorm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    result = layernorm_fn(x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm)
    if out is not None:
        out_shape = out.shape
        result = result.reshape(out_shape)
        np.copyto(np.asarray(out), result)
        return out, None, None
    return result, None, None