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

"""Weight loader for GGUF and safetensors formats.

Loads model weights as numpy arrays, dequantizes quantized formats,
and stages them to Vulkan device memory. Zero torch dependency.
"""

import ctypes
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .numpy_bridge import VulkanDevice


@dataclass
class ModelWeights:
    """Loaded model weights in Vulkan device memory."""
    device: VulkanDevice
    # weight_name -> Vulkan device pointer
    vulkan_ptrs: Dict[str, ctypes.c_void_p] = field(default_factory=dict)
    # weight_name -> numpy array (host copy)
    numpy_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    # Model metadata
    n_layers: int = 0
    hidden_dim: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    intermediate_dim: int = 0
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    def get_numpy(self, name: str) -> np.ndarray:
        """Get weight as numpy array (from cache or Vulkan)."""
        return self.numpy_cache.get(name)

    def get_vulkan(self, name: str) -> ctypes.c_void_p:
        """Get weight as Vulkan device pointer."""
        return self.vulkan_ptrs.get(name)

    def stage_to_vulkan(self, name: str, arr: np.ndarray):
        """Stage a numpy array to Vulkan device memory."""
        arr = np.ascontiguousarray(arr).astype(np.float32)
        self.numpy_cache[name] = arr
        self.vulkan_ptrs[name] = self.device.upload(arr)

    def free(self):
        """Free all Vulkan device memory."""
        for ptr in self.vulkan_ptrs.values():
            self.device.free(ptr)
        self.vulkan_ptrs.clear()
        self.numpy_cache.clear()


def _dequantize_gguf(data: np.ndarray, qtype: int) -> np.ndarray:
    """Dequantize GGUF quantized data to float32 using the gguf library."""
    from gguf import GGMLQuantizationType, dequantize
    qt = GGMLQuantizationType(qtype)
    deq = dequantize(np.asarray(data, dtype=np.uint8), qt)
    return np.asarray(deq, dtype=np.float32)


def load_gguf(path: str, device: VulkanDevice) -> ModelWeights:
    """Load a GGUF model file into Vulkan device memory.

    Args:
        path: Path to .gguf file
        device: VulkanDevice to stage weights to
    Returns:
        ModelWeights with all tensors staged to Vulkan
    """
    from gguf import GGMLQuantizationType, GGUFReader

    reader = GGUFReader(path, "r")
    weights = ModelWeights(device=device)

    # Extract metadata
    arch = ""
    for name, field in reader.fields.items():
        if name == "general.architecture":
            val = field.parts[list(field.data)[0]]
            if isinstance(val, (bytes, bytearray, memoryview)):
                arch = str(val, encoding="utf-8")
            elif hasattr(val, "tobytes"):
                # numpy array/memmap - convert via tobytes
                arch = val.tobytes().decode("utf-8")
            else:
                arch = str(val)

    if arch:
        prefix = arch
        meta_get = lambda key: _get_gguf_metadata(reader, f"{prefix}.{key}")
        weights.n_layers = int(meta_get("block_count") or 0)
        weights.hidden_dim = int(meta_get("embedding_length") or 0)
        weights.n_heads = int(meta_get("attention.head_count") or 0)
        weights.n_kv_heads = int(meta_get("attention.head_count_kv") or weights.n_heads)
        weights.head_dim = weights.hidden_dim // weights.n_heads if weights.n_heads else 0
        weights.norm_eps = float(meta_get("attention.layer_norm_rms_epsilon") or 1e-6)
        weights.rope_theta = float(meta_get("rope.freq_base") or 10000.0)

        # Try to get vocab size from tokenizer
        vocab_key = f"{prefix}.vocab_size"
        weights.vocab_size = int(_get_gguf_metadata(reader, vocab_key) or 0)
        if weights.vocab_size == 0:
            # Count tokenizer tokens
            tok_key = "tokenizer.ggml.tokens"
            tok_field = reader.fields.get(tok_key)
            if tok_field:
                weights.vocab_size = len(list(tok_field.data))

        # Intermediate dim
        weights.intermediate_dim = int(meta_get("feed_forward_length") or 0)

    # Load tensors
    for tensor in reader.tensors:
        name = tensor.name
        shape = list(reversed(tensor.shape.tolist()))
        ttype = int(tensor.tensor_type)

        if ttype in (GGMLQuantizationType.F32, 0):
            arr = np.frombuffer(tensor.data, dtype=np.float32).reshape(shape).copy()
        elif ttype in (GGMLQuantizationType.F16, 1):
            arr = np.frombuffer(tensor.data, dtype=np.float16).reshape(shape).copy().astype(np.float32)
        elif ttype in (GGMLQuantizationType.BF16, 30):
            # BF16: 1 sign + 8 exp + 7 mantissa, stored as uint16
            raw = np.frombuffer(tensor.data, dtype=np.uint16).copy()
            # Convert BF16 -> FP32: shift mantissa up, reconstruct exponent
            sign = (raw >> 15).astype(np.uint32) << 31
            exp = ((raw >> 7) & 0xFF).astype(np.uint32) << 23
            mantissa = (raw & 0x7F).astype(np.uint32) << 16
            fp32_bits = sign | exp | mantissa
            arr = fp32_bits.view(np.float32).reshape(shape).copy()
        elif ttype in (12, 13, 14):  # Q4_K, Q5_K, Q6_K
            arr = _dequantize_gguf(tensor.data, ttype).reshape(shape)
        elif ttype == 8:  # Q8_0
            arr = _dequantize_gguf(tensor.data, ttype).reshape(shape)
        else:
            # Try generic dequantize
            try:
                arr = _dequantize_gguf(tensor.data, ttype).reshape(shape)
            except Exception:
                # Fall back to raw bytes
                arr = np.frombuffer(tensor.data, dtype=np.float32).reshape(shape).copy()

        weights.stage_to_vulkan(name, arr)

    return weights


def load_safetensors(path: str, device: VulkanDevice) -> ModelWeights:
    """Load a safetensors file into Vulkan device memory.

    Args:
        path: Path to .safetensors file
        device: VulkanDevice to stage weights to
    Returns:
        ModelWeights with all tensors staged to Vulkan
    """
    from safetensors.numpy import load_file

    tensor_dict = load_file(path)
    weights = ModelWeights(device=device)

    for name, arr in tensor_dict.items():
        weights.stage_to_vulkan(name, arr.astype(np.float32))

    return weights


def load_safetensors_dir(
    dir_path: str, device: VulkanDevice, pattern: str = "*.safetensors"
) -> ModelWeights:
    """Load all safetensors files from a directory.

    Args:
        dir_path: Directory containing .safetensors files
        device: VulkanDevice
        pattern: glob pattern (default: *.safetensors)
    Returns:
        ModelWeights with all tensors staged to Vulkan
    """
    import glob as globmod

    weights = ModelWeights(device=device)
    files = sorted(globmod.glob(os.path.join(dir_path, pattern)))

    for f in files:
        from safetensors.numpy import load_file
        tensor_dict = load_file(f)
        for name, arr in tensor_dict.items():
            weights.stage_to_vulkan(name, arr.astype(np.float32))

    return weights


def _get_gguf_metadata(reader, key: str):
    """Get a metadata value from a GGUFReader."""
    field = reader.fields.get(key)
    if field is None:
        return None
    idx = list(field.data)[0]
    val = field.parts[idx]
    if field.types[0] == 3:  # STRING
        if isinstance(val, (bytes, bytearray, memoryview)):
            return str(val, encoding="utf-8")
        elif hasattr(val, "tobytes"):
            return val.tobytes().decode("utf-8")
        else:
            return str(val)
    elif field.types[0] == 4:  # ARRAY
        # field.data is a list of indices into field.parts
        # Each part may be a numpy memmap array
        result = [field.parts[i] for i in field.data]
        # If it is a single-element array, return the scalar
        if len(result) == 1:
            item = result[0]
            if hasattr(item, "tolist"):
                val = item.tolist()
                # If tolist returned a single-element list, extract the element
                if isinstance(val, list) and len(val) == 1:
                    return val[0]
                return val
            elif hasattr(item, "item"):
                return item.item()
            else:
                return item
        return result
    else:
        return val[0].tolist() if hasattr(val[0], "tolist") else val[0]
        return val[0] if hasattr(val, '__getitem__') else val
