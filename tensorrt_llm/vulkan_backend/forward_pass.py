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

"""Full transformer forward pass using Vulkan backend.

Chains Vulkan compute ops (GEMM, attention, norm, activations) with
numpy helper ops (RoPE, embedding, sampling) for a complete inference
pipeline. Zero torch dependency.
"""

import ctypes
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .numpy_bridge import (
    VulkanDevice,
    vulkan_attention,
    vulkan_elementwise_add,
    vulkan_elementwise_mul,
    vulkan_gemm,
    vulkan_rms_norm,
    vulkan_silu,
    vulkan_swiglu,
    vulkan_softmax,
)
from .vulkan_ops import (
    embedding,
    precompute_rope_freqs,
    rms_norm_cpu,
    rope,
    sample_from_probs,
    softmax_cpu,
    topk_cpu,
)
from .weight_loader import ModelWeights


class KVCache:
    """Paged KV cache stored on Vulkan device."""

    def __init__(
        self,
        device: VulkanDevice,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
    ):
        self.device = device
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Flat K/V buffers per layer: [max_seq_len, n_kv_heads * head_dim]
        self.k_ptrs: List[ctypes.c_void_p] = []
        self.v_ptrs: List[ctypes.c_void_p] = []

        kv_dim = n_kv_heads * head_dim
        for _ in range(n_layers):
            self.k_ptrs.append(device.upload(np.zeros(max_seq_len * kv_dim, dtype=np.float32)))
            self.v_ptrs.append(device.upload(np.zeros(max_seq_len * kv_dim, dtype=np.float32)))

        self.seq_len = 0

    def update(
        self,
        layer_idx: int,
        new_k: np.ndarray,
        new_v: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Append new K/V and return full K/V for this layer.

        Args:
            layer_idx: transformer layer index
            new_k: (tokens, n_kv_heads, head_dim)
            new_v: (tokens, n_kv_heads, head_dim)
        Returns:
            (full_k, full_v) each (seq_len, n_kv_heads, head_dim)
        """
        n_tokens = new_k.shape[0]
        kv_dim = self.n_kv_heads * self.head_dim
        start = self.seq_len
        end = start + n_tokens

        # Upload new K/V to the flat buffer
        k_flat = new_k.reshape(n_tokens, kv_dim).astype(np.float32)
        v_flat = new_v.reshape(n_tokens, kv_dim).astype(np.float32)

        # Copy into the correct position in the flat buffer
        # We need to download, update, re-upload (correctness-first path)
        k_buf = self.device.download(self.k_ptrs[layer_idx], (self.max_seq_len, kv_dim))
        v_buf = self.device.download(self.v_ptrs[layer_idx], (self.max_seq_len, kv_dim))

        k_buf[start:end] = k_flat
        v_buf[start:end] = v_flat

        self.device.free(self.k_ptrs[layer_idx])
        self.device.free(self.v_ptrs[layer_idx])
        self.k_ptrs[layer_idx] = self.device.upload(k_buf)
        self.v_ptrs[layer_idx] = self.device.upload(v_buf)

        # Return full K/V as (seq_len, n_kv_heads, head_dim)
        full_k = k_buf[:end].reshape(end, self.n_kv_heads, self.head_dim)
        full_v = v_buf[:end].reshape(end, self.n_kv_heads, self.head_dim)

        if layer_idx == self.n_layers - 1:
            self.seq_len = end

        return full_k, full_v

    def get(
        self, layer_idx: int, seq_len: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get K/V from cache as numpy arrays."""
        if seq_len is None:
            seq_len = self.seq_len
        kv_dim = self.n_kv_heads * self.head_dim
        k = self.device.download(self.k_ptrs[layer_idx], (self.max_seq_len, kv_dim))[:seq_len]
        v = self.device.download(self.v_ptrs[layer_idx], (self.max_seq_len, kv_dim))[:seq_len]
        return (
            k.reshape(seq_len, self.n_kv_heads, self.head_dim),
            v.reshape(seq_len, self.n_kv_heads, self.head_dim),
        )

    def free(self):
        for p in self.k_ptrs + self.v_ptrs:
            self.device.free(p)


class TransformerBlock:
    """Single transformer decoder layer using Vulkan ops."""

    def __init__(
        self,
        layer_idx: int,
        weights: ModelWeights,
        device: VulkanDevice,
    ):
        self.layer_idx = layer_idx
        self.weights = weights
        self.device = device
        self.n_heads = weights.n_heads
        self.n_kv_heads = weights.n_kv_heads
        self.head_dim = weights.head_dim
        self.hidden_dim = weights.hidden_dim
        self.intermediate_dim = weights.intermediate_dim
        self.norm_eps = weights.norm_eps

        # GQA ratio
        self.n_rep = self.n_heads // self.n_kv_heads

    def forward(
        self,
        x: np.ndarray,
        cos: np.ndarray,
        sin: np.ndarray,
        kv_cache: Optional[KVCache] = None,
    ) -> np.ndarray:
        """Forward pass through one transformer block.

        Args:
            x: (tokens, hidden_dim) input activations
            cos, sin: RoPE tables (tokens, head_dim//2) or precomputed
            kv_cache: optional KV cache for incremental decode
        Returns:
            output: (tokens, hidden_dim)
        """
        prefix = f"blk.{self.layer_idx}"

        # --- Input RMS Norm ---
        norm_weight = self.weights.get_numpy(f"{prefix}.attn_norm.weight")
        residual = x.copy()
        x_norm = rms_norm_cpu(x, norm_weight, self.norm_eps)

        # --- Self Attention ---
        q_proj = self.weights.get_numpy(f"{prefix}.attn_q.weight")
        k_proj = self.weights.get_numpy(f"{prefix}.attn_k.weight")
        v_proj = self.weights.get_numpy(f"{prefix}.attn_v.weight")

        q = vulkan_gemm(self.device, x_norm, q_proj.T)  # (tokens, n_heads * head_dim)
        k = vulkan_gemm(self.device, x_norm, k_proj.T)  # (tokens, n_kv_heads * head_dim)
        v = vulkan_gemm(self.device, x_norm, v_proj.T)

        # Reshape to (tokens, heads, head_dim)
        q = q.reshape(-1, self.n_heads, self.head_dim)
        k = k.reshape(-1, self.n_kv_heads, self.head_dim)
        v = v.reshape(-1, self.n_kv_heads, self.head_dim)

        # --- RoPE ---
        q, k = rope(q, k, cos, sin)

        # --- KV Cache update ---
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)
            seq_len = kv_cache.seq_len
        else:
            seq_len = q.shape[0]

        # --- GQA expand K/V ---
        if self.n_rep > 1:
            k = np.repeat(k, self.n_rep, axis=1)
            v = np.repeat(v, self.n_rep, axis=1)

        # --- Attention ---
        # q: (tokens, heads, head_dim) -> (1, heads, seq, head_dim)
        q_bt = q.transpose(1, 0, 2)[np.newaxis, :, :, :]  # (1, H, S_q, D)
        k_bt = k.transpose(1, 0, 2)[np.newaxis, :, :, :]  # (1, H, S_k, D)
        v_bt = v.transpose(1, 0, 2)[np.newaxis, :, :, :]

        attn_out = vulkan_attention(self.device, q_bt, k_bt, v_bt, causal=True)
        attn_out = attn_out[0].transpose(1, 0, 2)  # (tokens, heads, head_dim)
        attn_out = attn_out.reshape(-1, self.n_heads * self.head_dim)

        # --- Output projection ---
        o_proj = self.weights.get_numpy(f"{prefix}.attn_output.weight")
        attn_out = vulkan_gemm(self.device, attn_out, o_proj.T)

        # --- Residual ---
        x = vulkan_elementwise_add(self.device, residual, attn_out)

        # --- MLP ---
        residual = x.copy()
        post_norm_weight = self.weights.get_numpy(f"{prefix}.ffn_norm.weight")
        x_norm = rms_norm_cpu(x, post_norm_weight, self.norm_eps)

        # SwiGLU MLP: gate_proj + up_proj -> swiglu -> down_proj
        gate_proj = self.weights.get_numpy(f"{prefix}.ffn_gate.weight")
        up_proj = self.weights.get_numpy(f"{prefix}.ffn_up.weight")
        down_proj = self.weights.get_numpy(f"{prefix}.ffn_down.weight")

        gate = vulkan_gemm(self.device, x_norm, gate_proj.T)
        up = vulkan_gemm(self.device, x_norm, up_proj.T)

        # SwiGLU: gate * silu(up) — or fused swiglu if shapes match
        # gate and up are (tokens, intermediate_dim)
        # Concatenate along last dim for fused swiglu
        gate_up = np.concatenate([gate, up], axis=-1)  # (tokens, 2 * intermediate_dim)
        mlp_out = vulkan_swiglu(self.device, gate_up, self.intermediate_dim)

        mlp_out = vulkan_gemm(self.device, mlp_out, down_proj.T)

        # --- Residual ---
        x = vulkan_elementwise_add(self.device, residual, mlp_out)
        return x


class TransformerModel:
    """Full transformer model running on Vulkan backend."""

    def __init__(self, weights: ModelWeights):
        self.weights = weights
        self.device = weights.device

        self.n_layers = weights.n_layers
        self.hidden_dim = weights.hidden_dim
        self.n_heads = weights.n_heads
        self.n_kv_heads = weights.n_kv_heads
        self.head_dim = weights.head_dim
        self.vocab_size = weights.vocab_size
        self.norm_eps = weights.norm_eps

        self.blocks = [
            TransformerBlock(i, weights, self.device) for i in range(self.n_layers)
        ]

        # Precompute RoPE frequencies
        self.cos_table, self.sin_table = precompute_rope_freqs(
            4096, self.head_dim, weights.rope_theta
        )

    def forward(
        self,
        token_ids: np.ndarray,
        position: int = 0,
        kv_cache: Optional[KVCache] = None,
    ) -> np.ndarray:
        """Forward pass: tokens -> logits.

        Args:
            token_ids: (num_tokens,) int32 token ids
            position: starting position for RoPE
            kv_cache: optional KV cache
        Returns:
            logits: (vocab_size,) float32 (last token only)
        """
        # --- Token embedding ---
        emb_table = self.weights.get_numpy("token_embd.weight")
        x = embedding(emb_table, token_ids)  # (num_tokens, hidden_dim)

        # --- Transformer blocks ---
        seq_len = token_ids.shape[0]
        cos = self.cos_table[position:position + seq_len]
        sin = self.sin_table[position:position + seq_len]

        for block in self.blocks:
            x = block.forward(x, cos, sin, kv_cache)

        # --- Final RMS Norm ---
        norm_weight = self.weights.get_numpy("output_norm.weight")
        x = rms_norm_cpu(x, norm_weight, self.norm_eps)

        # --- Language model head ---
        lm_head = self.weights.get_numpy("output.weight")
        if lm_head is None:
            # Weight-tied: output weight = token embedding
            lm_head = emb_table

        # Last token logits
        last_hidden = x[-1:]  # (1, hidden_dim)
        logits = vulkan_gemm(self.device, last_hidden, lm_head.T)  # (1, vocab_size)
        return logits.ravel()

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        kv_cache: Optional[KVCache] = None,
    ) -> List[int]:
        """Autoregressive generation.

        Args:
            prompt_ids: (prompt_len,) int32 token ids
            max_tokens: max new tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling
            kv_cache: optional KV cache for incremental decode
        Returns:
            List of generated token ids
        """
        generated = []
        position = 0
        input_ids = prompt_ids

        for step in range(max_tokens):
            logits = self.forward(input_ids, position=position, kv_cache=kv_cache)
            token = sample_from_probs(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            generated.append(token)
            position += len(input_ids)
            input_ids = np.array([token], dtype=np.int32)

        return generated

    def free(self):
        for block in self.blocks:
            pass  # blocks share weights from ModelWeights
        self.weights.free()
