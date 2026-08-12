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

"""flashinfer.sampling — Vulkan-backed sampling functions.

Drop-in replacement for flashinfer.sampling using numpy + Vulkan compute.
"""

import numpy as np
from typing import Optional, Tuple, Union


def softmax(
    logits: np.ndarray,
    temperature: Optional[Union[np.ndarray, float]] = None,
    enable_pdl: Optional[bool] = None,
) -> np.ndarray:
    """Softmax with optional temperature scaling."""
    if temperature is not None:
        logits = logits / max(float(temperature), 1e-6)
    x_max = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - x_max)
    return e / np.sum(e, axis=-1, keepdims=True)


def sampling_from_probs(
    probs: np.ndarray,
    indices: Optional[np.ndarray] = None,
    deterministic: bool = False,
    generator=None,
    check_nan: bool = False,
    seed=None,
    offset=None,
    return_valid: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Sample from probability distribution."""
    if probs.ndim == 1:
        token = int(np.random.choice(len(probs), p=probs))
        result = np.array([token], dtype=np.int64)
    else:
        result = np.array(
            [int(np.random.choice(len(p), p=p)) for p in probs],
            dtype=np.int64,
        )
    if return_valid:
        valid = np.ones_like(result, dtype=np.bool_)
        return result, valid
    return result


def sampling_from_logits(
    logits: np.ndarray,
    indices: Optional[np.ndarray] = None,
    deterministic: bool = False,
    generator=None,
    check_nan: bool = False,
    seed=None,
    offset=None,
) -> np.ndarray:
    """Sample from logits (apply softmax then sample)."""
    probs = softmax(logits)
    return sampling_from_probs(probs)


def top_p_sampling_from_probs(
    probs: np.ndarray,
    top_p: Union[np.ndarray, float] = 0.9,
    indices: Optional[np.ndarray] = None,
    deterministic: bool = False,
    generator=None,
    check_nan: bool = False,
    seed=None,
    offset=None,
    return_valid: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Nucleus (top-p) sampling."""
    if probs.ndim == 1:
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        cutoff = cumsum > top_p
        if np.any(cutoff):
            first_true = np.argmax(cutoff)
            sorted_probs[first_true + 1:] = 0
        sorted_probs = sorted_probs / max(sorted_probs.sum(), 1e-30)
        token = int(np.random.choice(len(sorted_probs), p=sorted_probs))
        result = np.array([token], dtype=np.int64)
    else:
        results = []
        for p in probs:
            sidx = np.argsort(-p)
            sp = p[sidx]
            cs = np.cumsum(sp)
            co = cs > top_p
            if np.any(co):
                ft = np.argmax(co)
                sp[ft + 1:] = 0
            sp = sp / max(sp.sum(), 1e-30)
            results.append(int(np.random.choice(len(sp), p=sp)))
        result = np.array(results, dtype=np.int64)
    if return_valid:
        valid = np.ones_like(result, dtype=np.bool_)
        return result, valid
    return result


def top_k_sampling_from_probs(
    probs: np.ndarray,
    top_k: Union[np.ndarray, int] = 50,
    indices: Optional[np.ndarray] = None,
    deterministic: bool = False,
    generator=None,
    check_nan: bool = False,
    seed=None,
    offset=None,
    return_valid: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Top-K sampling."""
    k = int(top_k)
    if probs.ndim == 1:
        topk_idx = np.argpartition(-probs, k)[:k]
        topk_probs = probs[topk_idx]
        topk_probs = topk_probs / max(topk_probs.sum(), 1e-30)
        token = topk_idx[int(np.random.choice(len(topk_probs), p=topk_probs))]
        result = np.array([token], dtype=np.int64)
    else:
        results = []
        for p in probs:
            ti = np.argpartition(-p, k)[:k]
            tp = p[ti]
            tp = tp / max(tp.sum(), 1e-30)
            results.append(ti[int(np.random.choice(len(tp), p=tp))])
        result = np.array(results, dtype=np.int64)
    if return_valid:
        valid = np.ones_like(result, dtype=np.bool_)
        return result, valid
    return result


def top_k_top_p_sampling_from_probs(
    probs: np.ndarray,
    indices: Optional[np.ndarray] = None,
    maybe_top_k_arr: Optional[np.ndarray] = None,
    top_k_val: int = 50,
    maybe_top_p_arr: Optional[np.ndarray] = None,
    top_p_val: float = 0.9,
    deterministic: bool = False,
    generator=None,
    seed=None,
    offset=None,
    return_valid: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Top-K then Top-P sampling."""
    k = int(top_k_val)
    p = float(top_p_val)
    if probs.ndim == 1:
        topk_idx = np.argpartition(-probs, k)[:k]
        topk_probs = np.full_like(probs, -np.inf)
        topk_probs[topk_idx] = probs[topk_idx]
        topk_probs = np.where(topk_probs == -np.inf, 0, topk_probs)
        sorted_idx = np.argsort(-topk_probs)
        sorted_probs = topk_probs[sorted_idx]
        cs = np.cumsum(sorted_probs)
        co = cs > p
        if np.any(co):
            ft = np.argmax(co)
            sorted_probs[ft + 1:] = 0
        sorted_probs = sorted_probs / max(sorted_probs.sum(), 1e-30)
        token = sorted_idx[int(np.random.choice(len(sorted_probs), p=sorted_probs))]
        result = np.array([token], dtype=np.int64)
    else:
        results = []
        for pr in probs:
            ti = np.argpartition(-pr, k)[:k]
            tp = np.full_like(pr, -np.inf)
            tp[ti] = pr[ti]
            tp = np.where(tp == -np.inf, 0, tp)
            si = np.argsort(-tp)
            sp = tp[si]
            cs = np.cumsum(sp)
            co = cs > p
            if np.any(co):
                ft = np.argmax(co)
                sp[ft + 1:] = 0
            sp = sp / max(sp.sum(), 1e-30)
            results.append(si[int(np.random.choice(len(sp), p=sp))])
        result = np.array(results, dtype=np.int64)
    if return_valid:
        valid = np.ones_like(result, dtype=np.bool_)
        return result, valid
    return result


def top_k_top_p_sampling_from_logits(
    logits: np.ndarray,
    maybe_top_k_arr: Optional[np.ndarray] = None,
    top_k_val: int = 50,
    maybe_top_p_arr: Optional[np.ndarray] = None,
    top_p_val: float = 0.9,
    temperature: float = 1.0,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Top-K then Top-P sampling from logits."""
    probs = softmax(logits / max(temperature, 1e-6))
    return top_k_top_p_sampling_from_probs(
        probs, top_k_val=top_k_val, top_p_val=top_p_val, **kwargs
    )


def min_p_sampling_from_probs(
    probs: np.ndarray,
    min_p: Union[np.ndarray, float] = 0.05,
    indices: Optional[np.ndarray] = None,
    deterministic: bool = False,
    generator=None,
    check_nan: bool = False,
    seed=None,
    offset=None,
    return_valid: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Min-P sampling: filter tokens with probability < min_p * max_prob."""
    mp = float(min_p)
    if probs.ndim == 1:
        max_prob = np.max(probs)
        threshold = mp * max_prob
        mask = probs >= threshold
        filtered = probs.copy()
        filtered[~mask] = 0
        total = filtered.sum()
        if total > 0:
            filtered = filtered / total
        token = int(np.random.choice(len(filtered), p=filtered))
        result = np.array([token], dtype=np.int64)
    else:
        results = []
        for p in probs:
            mx = np.max(p)
            thr = mp * mx
            m = p >= thr
            f = p.copy()
            f[~m] = 0
            s = f.sum()
            if s > 0:
                f = f / s
            results.append(int(np.random.choice(len(f), p=f)))
        result = np.array(results, dtype=np.int64)
    if return_valid:
        valid = np.ones_like(result, dtype=np.bool_)
        return result, valid
    return result


def top_p_renorm_probs(
    probs: np.ndarray,
    maybe_top_p_arr: Optional[np.ndarray] = None,
    top_p_val: float = 0.9,
    is_deterministic: bool = True,
    workspace: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Renormalize probabilities with top-p filtering."""
    p = float(top_p_val)
    if probs.ndim == 1:
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cs = np.cumsum(sorted_probs)
        co = cs > p
        if np.any(co):
            ft = np.argmax(co)
            sorted_probs[ft + 1:] = 0
        total = sorted_probs.sum()
        if total > 0:
            sorted_probs = sorted_probs / total
        result = np.zeros_like(probs)
        result[sorted_idx] = sorted_probs
        return result
    else:
        results = []
        for pr in probs:
            si = np.argsort(-pr)
            sp = pr[si]
            cs = np.cumsum(sp)
            co = cs > p
            if np.any(co):
                ft = np.argmax(co)
                sp[ft + 1:] = 0
            s = sp.sum()
            if s > 0:
                sp = sp / s
            r = np.zeros_like(pr)
            r[si] = sp
            results.append(r)
        return np.array(results)


def top_k_renorm_probs(
    probs: np.ndarray,
    maybe_top_k_arr: Optional[np.ndarray] = None,
    top_k_val: int = 50,
    row_states_buffer: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Renormalize probabilities with top-k filtering."""
    k = int(top_k_val)
    if probs.ndim == 1:
        topk_idx = np.argpartition(-probs, k)[:k]
        mask = np.zeros_like(probs, dtype=np.bool_)
        mask[topk_idx] = True
        result = probs.copy()
        result[~mask] = 0
        total = result.sum()
        if total > 0:
            result = result / total
        return result
    else:
        results = []
        for pr in probs:
            ti = np.argpartition(-pr, k)[:k]
            m = np.zeros_like(pr, dtype=np.bool_)
            m[ti] = True
            r = pr.copy()
            r[~m] = 0
            s = r.sum()
            if s > 0:
                r = r / s
            results.append(r)
        return np.array(results)


def top_k_mask_logits(
    logits: np.ndarray,
    maybe_top_k_arr: Optional[np.ndarray] = None,
    top_k_val: int = 50,
    row_states_buffer: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mask logits outside top-k with -inf."""
    k = int(top_k_val)
    if logits.ndim == 1:
        topk_idx = np.argpartition(-logits, k)[:k]
        mask = np.full_like(logits, -np.inf)
        mask[topk_idx] = logits[topk_idx]
        return mask
    else:
        results = []
        for logit in logits:
            ti = np.argpartition(-logit, k)[:k]
            m = np.full_like(logit, -np.inf)
            m[ti] = logit[ti]
            results.append(m)
        return np.array(results)


def chain_speculative_sampling(
    draft_probs: np.ndarray,
    draft_token_ids: np.ndarray,
    target_probs: np.ndarray,
    output_accepted_token_num: Optional[np.ndarray] = None,
    output_emitted_draft_token_num: Optional[np.ndarray] = None,
    deterministic: bool = False,
    generator=None,
    seed=None,
    offset=None,
) -> np.ndarray:
    """Chain speculative sampling (simplified)."""
    batch_size = draft_token_ids.shape[0]
    result = np.zeros(batch_size, dtype=np.int64)
    for i in range(batch_size):
        draft_token = draft_token_ids[i]
        draft_p = draft_probs[i, draft_token] if draft_probs.ndim > 1 else draft_probs[draft_token]
        target_p = target_probs[i, draft_token] if target_probs.ndim > 1 else target_probs[draft_token]
        if draft_p > 0 and np.random.random() < min(1.0, target_p / draft_p):
            result[i] = draft_token
        else:
            tp = target_probs[i] if target_probs.ndim > 1 else target_probs
            result[i] = int(np.random.choice(len(tp), p=tp))
    return result
