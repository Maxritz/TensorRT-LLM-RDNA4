import numpy as np

from tensorrt_llm._torch.modules.fla.utils import is_gather_supported


def safe_exp(x):
    return np.exp(np.where(x <= 0, x, float("-inf")))


def exp(x):
    return np.exp(x)


def exp2(x):
    return np.exp2(x)


def log(x):
    return np.log(x)


def log2(x):
    return np.log2(x)


def gather(src, index, axis, _builder=None):
    return np.take_along_axis(src, index, axis)


make_tensor_descriptor = None
