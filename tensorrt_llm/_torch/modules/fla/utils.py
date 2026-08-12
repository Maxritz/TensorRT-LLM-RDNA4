# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/utils.py
# -*- coding: utf-8 -*-

import contextlib
import functools
import inspect
import logging
import os
import sys
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

COMPILER_MODE = os.getenv("FLA_COMPILER_MODE") == "1"
FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"


@lru_cache(maxsize=1)
def check_environments():
    if sys.platform == "win32":
        logger.warning("FLA modules running in numpy mode on Windows.")
    return None


check_environments()


def get_abs_err(x, y):
    return np.max(np.abs(x - y)).item()


def get_err_ratio(x, y):
    err = np.sqrt(np.mean((x - y) ** 2)).item()
    base = np.sqrt(np.mean(x ** 2)).item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    logger.info(msg)
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    if error_rate < ratio:
        return
    raise AssertionError(msg)


SUPPRESS_LEVEL = int(os.getenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))


def tensor_cache(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    cache_entries: Tuple[Optional[Tuple], Optional[Dict], Any] = []
    cache_size = 4

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries, cache_size
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                        k in last_kwargs and v is last_kwargs[k]
                        for k, v in kwargs.items()):
                    cache_entries = (cache_entries[:i] + cache_entries[i + 1:] +
                                     [(args, kwargs, last_result)])
                    return last_result

        result = fn(*args, **kwargs)

        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


def input_guard(fn=None, *, exclude_args: Optional[list[str]] = None):
    def decorator(func):
        sig = inspect.signature(func) if exclude_args else None

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if exclude_args and sig is not None:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                for name, value in bound.arguments.items():
                    if isinstance(value, np.ndarray) and name not in exclude_args:
                        bound.arguments[name] = np.ascontiguousarray(value)
                contiguous_args = bound.args
                contiguous_kwargs = bound.kwargs
            else:
                contiguous_args = tuple(
                    i if not isinstance(i, np.ndarray) else np.ascontiguousarray(i)
                    for i in args)
                contiguous_kwargs = {
                    k:
                    (v if not isinstance(v, np.ndarray) else np.ascontiguousarray(v))
                    for k, v in kwargs.items()
                }

            with contextlib.nullcontext():
                return func(*contiguous_args, **contiguous_kwargs)

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


contiguous = input_guard


def require_version(version, hint):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            return fn(
                ctx,
                *(i if not isinstance(i, np.ndarray) else np.ascontiguousarray(i)
                  for i in args),
                **{
                    k:
                    (v if not isinstance(v, np.ndarray) else np.ascontiguousarray(v))
                    for k, v in kwargs.items()
                },
            )
        return wrapper
    return decorator


def checkpoint(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


@lru_cache(maxsize=None)
def check_pytorch_version(version_s: str = "2.4") -> bool:
    return True


def _cpu_device_warning():
    import warnings
    warnings.warn("Running in numpy mode (no GPU).", stacklevel=1)


@lru_cache(maxsize=None)
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    return -1


@lru_cache(maxsize=None)
def get_available_device() -> str:
    return "cpu"


@lru_cache(maxsize=None)
def _check_platform() -> Literal["nvidia", "amd", "intel", "musa"]:
    return "cpu"


device = "cpu"
device_torch_lib = None
device_platform = _check_platform()

is_amd = False
is_intel = False
is_nvidia = False
is_intel_alchemist = False
is_nvidia_hopper = False
use_cuda_graph = False
is_tf32_supported = False
is_gather_supported = False


def get_all_max_shared_mem():
    return [-1]


class Backend(Enum):
    ADA = 101376
    AMPERE = 166912
    HOPPER = 232448
    DEFAULT = 102400

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@lru_cache(maxsize=None)
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    return False


autocast_custom_fwd = None
autocast_custom_bwd = None


def custom_device_ctx(index: int):
    return contextlib.nullcontext()
