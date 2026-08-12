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

"""Torch-free Vulkan inference for GGUF/safetensors models.

Usage:
    from tensorrt_llm._torch.vulkan_backend.vulkan_infer import run_inference
    output = run_inference("model.gguf", "Hello world")
"""

import time
from typing import Optional

import numpy as np

from .numpy_bridge import VulkanDevice
from .weight_loader import load_gguf, load_safetensors, load_safetensors_dir
from .forward_pass import KVCache, TransformerModel


def _tokenize_simple(text: str, vocab: dict) -> np.ndarray:
    """Simple whitespace tokenizer. Replace with real tokenizer for production."""
    tokens = []
    for word in text.split():
        if word in vocab:
            tokens.append(vocab[word])
        else:
            # Fall back to byte-level
            for ch in word:
                key = f"<0x{ord(ch):02X}>"
                if key in vocab:
                    tokens.append(vocab[key])
                else:
                    tokens.append(0)  # UNK
    return np.array(tokens, dtype=np.int32)


def run_inference(
    model_path: str,
    prompt: str = "",
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    verbose: bool = True,
) -> str:
    """Run inference on a GGUF or safetensors model using the Vulkan backend.

    Args:
        model_path: path to .gguf or .safetensors file (or directory of .safetensors)
        prompt: text prompt
        max_tokens: max tokens to generate
        temperature: sampling temperature
        top_k: top-k filtering
        top_p: nucleus sampling
        verbose: print progress
    Returns:
        Generated text
    """
    device = VulkanDevice()
    if verbose:
        print(f"Vulkan device initialized")

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

    # Build model
    model = TransformerModel(weights)
    if verbose:
        print(f"Model built ({weights.n_layers} transformer blocks)")

    # Tokenize prompt
    if prompt:
        # Use GGUF tokenizer if available
        try:
            from gguf import GGUFReader
            reader = GGUFReader(model_path, "r")
            vocab = {}
            tokens_field = reader.fields.get("tokenizer.ggml.tokens")
            if tokens_field:
                for i, idx in enumerate(tokens_field.data):
                    val = bytes(tokens_field.parts[idx])
                    vocab[val.decode("utf-8", errors="replace")] = i
            prompt_ids = _tokenize_simple(prompt, vocab) if vocab else np.array([0], dtype=np.int32)
        except Exception:
            prompt_ids = np.array([0], dtype=np.int32)  # BOS token
    else:
        prompt_ids = np.array([0], dtype=np.int32)  # BOS token

    if verbose:
        print(f"Prompt tokens: {prompt_ids.tolist()}")

    # Create KV cache
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
    generated = []
    position = len(prompt_ids)
    token = int(np.argmax(logits))

    if verbose:
        print("Generating:", end=" ", flush=True)

    for step in range(max_tokens):
        t0 = time.time()
        logits = model.forward(np.array([token], dtype=np.int32), position=position, kv_cache=kv_cache)

        from .vulkan_ops import sample_from_probs
        next_token = sample_from_probs(logits, temperature=temperature, top_k=top_k, top_p=top_p)

        t_step = time.time() - t0
        generated.append(next_token)
        position += 1
        token = next_token

        if verbose and step < 50:
            print(f"[{next_token}]({t_step:.3f}s)", end=" ", flush=True)

    if verbose:
        print(f"\nGenerated {len(generated)} tokens")

    # Cleanup
    kv_cache.free()
    model.free()

    return str(generated)
