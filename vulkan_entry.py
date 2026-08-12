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

"""Standalone torch-free Vulkan inference entry point.

Bypasses tensorrt_llm/__init__.py entirely — no torch import, no C++ chain.
Pure Python + ctypes + numpy + Vulkan.

Usage:
    python vulkan_entry.py model.gguf "Hello world"
    python vulkan_entry.py model.safetensors "Hello world"
    python -c "from vulkan_entry import run; run('model.gguf', 'Hi')"
"""

import os
import sys
import time
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: add the vulkan_backend package to sys.path so we can import
# its submodules directly without going through tensorrt_llm/__init__.py.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.join(_HERE, "tensorrt_llm", "_torch", "vulkan_backend")
# We need the parent of vulkan_backend so relative imports work
_BACKEND_PARENT = os.path.join(_HERE, "tensorrt_llm", "_torch")
if _BACKEND_PARENT not in sys.path:
    sys.path.insert(0, _BACKEND_PARENT)
# Also add the repo root so we can import the vulkan_backend package directly
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Also ensure the gguf package from the sglang venv is importable.
_SGLANG_SITE = r"F:\AI-sglang\sglang-windows\.venv312\Lib\site-packages"
if os.path.isdir(_SGLANG_SITE) and _SGLANG_SITE not in sys.path:
    sys.path.insert(0, _SGLANG_SITE)

# Force Vulkan backend mode
os.environ.setdefault("TLLM_VULKAN_BACKEND", "1")


def _tokenize_simple(text: str, vocab: dict) -> np.ndarray:
    """Simple whitespace + byte-level tokenizer."""
    tokens: list[int] = []
    for word in text.split():
        if word in vocab:
            tokens.append(vocab[word])
        else:
            for ch in word:
                key = f"<0x{ord(ch):02X}>"
                if key in vocab:
                    tokens.append(vocab[key])
                else:
                    tokens.append(0)
    return np.array(tokens, dtype=np.int32)


def run(
    model_path: str,
    prompt: str = "",
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    verbose: bool = True,
) -> str:
    """Run torch-free Vulkan inference on a GGUF or safetensors model.

    Args:
        model_path: path to .gguf or .safetensors file/directory
        prompt: text prompt
        max_tokens: max tokens to generate
        temperature: sampling temperature
        top_k: top-k filtering
        top_p: nucleus sampling
        verbose: print progress
    Returns:
        Generated text
    """
    # Lazy imports — only after sys.path is configured
    from tensorrt_llm._torch.vulkan_backend.numpy_bridge import VulkanDevice
    from tensorrt_llm._torch.vulkan_backend.weight_loader import load_gguf, load_safetensors, load_safetensors_dir
    from tensorrt_llm._torch.vulkan_backend.forward_pass import KVCache, TransformerModel
    from tensorrt_llm._torch.vulkan_backend.vulkan_ops import sample_from_probs

    device = VulkanDevice()
    if verbose:
        print("Vulkan device initialized")

    # Load weights
    t0 = time.time()
    if model_path.endswith(".gguf"):
        weights = load_gguf(model_path, device)
    elif model_path.endswith(".safetensors"):
        weights = load_safetensors(model_path, device)
    elif model_path.endswith("/") or model_path.endswith("\\"):
        weights = load_safetensors_dir(model_path, device)
    else:
        raise ValueError(f"Unsupported model format: {model_path}")

    if verbose:
        t_load = time.time() - t0
        print(f"Loaded {len(weights.vulkan_ptrs)} tensors in {t_load:.2f}s")
        print(f"  Layers: {weights.n_layers}, Hidden: {weights.hidden_dim}, "
              f"Heads: {weights.n_heads}/{weights.n_kv_heads}, "
              f"Vocab: {weights.vocab_size}")

    model = TransformerModel(weights)
    if verbose:
        print(f"Model built ({weights.n_layers} transformer blocks)")

    # Tokenize
    if prompt:
        try:
            from gguf import GGUFReader
            reader = GGUFReader(model_path, "r")
            vocab: dict[str, int] = {}
            tokens_field = reader.fields.get("tokenizer.ggml.tokens")
            if tokens_field:
                for i, idx in enumerate(tokens_field.data):
                    val = bytes(tokens_field.parts[idx])
                    vocab[val.decode("utf-8", errors="replace")] = i
            prompt_ids = _tokenize_simple(prompt, vocab) if vocab else np.array([0], dtype=np.int32)
        except Exception:
            prompt_ids = np.array([0], dtype=np.int32)
    else:
        prompt_ids = np.array([0], dtype=np.int32)

    if verbose:
        print(f"Prompt tokens: {prompt_ids.tolist()}")

    kv_cache = KVCache(
        device, weights.n_layers, weights.n_kv_heads,
        weights.head_dim, max_seq_len=4096
    )

    # Prefill
    t0 = time.time()
    logits = model.forward(prompt_ids, position=0, kv_cache=kv_cache)
    t_prefill = time.time() - t0
    if verbose:
        print(f"Prefill: {t_prefill:.3f}s")

    # Generate
    generated: list[int] = []
    position = len(prompt_ids)
    token = int(np.argmax(logits))

    if verbose:
        print("Generating:", end=" ", flush=True)

    for step in range(max_tokens):
        t0 = time.time()
        logits = model.forward(
            np.array([token], dtype=np.int32), position=position, kv_cache=kv_cache
        )
        next_token = sample_from_probs(
            logits, temperature=temperature, top_k=top_k, top_p=top_p
        )
        t_step = time.time() - t0
        generated.append(next_token)
        position += 1
        token = next_token

        if verbose and step < 50:
            print(f"[{next_token}]({t_step:.3f}s)", end=" ", flush=True)

    if verbose:
        print(f"\nGenerated {len(generated)} tokens")

    kv_cache.free()
    model.free()
    return str(generated)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Torch-free Vulkan inference")
    parser.add_argument("model", help="Path to .gguf or .safetensors model")
    parser.add_argument("prompt", nargs="?", default="", help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        args.model, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(f"\nResult: {result}")
