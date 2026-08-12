"""Smoke test for all Vulkan ops (no torch)."""
import sys
sys.path.insert(0, r"F:\NV\TensorRT-LLM")
import numpy as np

from tensorrt_llm._torch.vulkan_backend.numpy_bridge import (
    VulkanDevice, vulkan_gemm, vulkan_silu, vulkan_softmax,
    vulkan_rms_norm, vulkan_elementwise_add, vulkan_elementwise_mul,
    vulkan_swiglu, vulkan_attention, vulkan_sigmoid, vulkan_gelu,
)
from tensorrt_llm._torch.vulkan_backend.vulkan_ops import (
    embedding, rope, precompute_rope_freqs, softmax_cpu, sample_from_probs,
)
from tensorrt_llm._torch.vulkan_backend.forward_pass import TransformerModel, KVCache

dev = VulkanDevice()
print("VulkanDevice: OK")
print()

tests = []

# GEMM
a = np.random.randn(4, 8).astype(np.float32)
b = np.random.randn(8, 16).astype(np.float32)
tests.append(("GEMM", np.allclose(vulkan_gemm(dev, a, b), a @ b, atol=1e-4)))

# SiLU
x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
tests.append(("SiLU", np.allclose(vulkan_silu(dev, x), x / (1 + np.exp(-x)), atol=1e-5)))

# GELU
x2 = np.random.randn(256).astype(np.float32)
GELU_COEF_A = 0.044715
SQRT_2_OVER_PI = 0.79788456080286535587989211986876
val = SQRT_2_OVER_PI * x2 * (1.0 + GELU_COEF_A * x2**2)
expected_gelu = 0.5 * x2 * (2.0 - 2.0 / (np.exp(2.0 * val) + 1.0))
tests.append(("GELU", np.allclose(vulkan_gelu(dev, x2), expected_gelu, atol=1e-5)))

# SwiGLU
tests.append(("SwiGLU", vulkan_swiglu(dev, np.random.randn(4, 16).astype(np.float32), 8).shape == (4, 8)))

# RoPE
cos, sin = precompute_rope_freqs(128, 64)
q = np.random.randn(1, 4, 128, 64).astype(np.float32)
k = np.random.randn(1, 4, 128, 64).astype(np.float32)
tests.append(("RoPE", rope(q, k, cos[:128], sin[:128])[0].shape == q.shape))

# KV cache
kv = KVCache(dev, 2, 4, 64, 256)
kn = np.random.randn(3, 4, 64).astype(np.float32)
vn = np.random.randn(3, 4, 64).astype(np.float32)
kf, vf = kv.update(0, kn, vn)
tests.append(("KV cache", kf.shape == (3, 4, 64)))
kv.free()

# RMS norm
tests.append(("RMS norm", vulkan_rms_norm(
    dev, np.random.randn(4, 64).astype(np.float32),
    np.ones(64, dtype=np.float32)).shape == (4, 64)))

# Softmax
s = np.random.randn(4, 128).astype(np.float32)
sm = vulkan_softmax(dev, s)
tests.append(("Softmax", sm.shape == (4, 128) and np.allclose(sm.sum(axis=-1), 1.0, atol=1e-3)))

# Attention
q4 = np.random.randn(1, 4, 16, 64).astype(np.float32)
k4 = np.random.randn(1, 4, 16, 64).astype(np.float32)
v4 = np.random.randn(1, 4, 16, 64).astype(np.float32)
tests.append(("Attention", vulkan_attention(dev, q4, k4, v4, causal=True).shape == (1, 4, 16, 64)))

# Elementwise add
a2 = np.random.randn(4, 8).astype(np.float32)
b2 = np.random.randn(4, 8).astype(np.float32)
tests.append(("Elem add", np.allclose(vulkan_elementwise_add(dev, a2, b2), a2 + b2, atol=1e-5)))

# Elementwise mul
tests.append(("Elem mul", np.allclose(vulkan_elementwise_mul(dev, a2, b2), a2 * b2, atol=1e-5)))

# Embedding
emb = np.random.randn(100, 64).astype(np.float32)
tests.append(("Embedding", embedding(emb, np.array([0, 1, 2], dtype=np.int32)).shape == (3, 64)))

# Sigmoid
sx = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
tests.append(("Sigmoid", np.allclose(vulkan_sigmoid(dev, sx), 1.0 / (1.0 + np.exp(-sx)), atol=1e-5)))

for name, ok in tests:
    status = "PASS" if ok else "FAIL"
    print("  %s: %s" % (name, status))

passed = sum(1 for _, ok in tests if ok)
print()
print("%d/%d PASS" % (passed, len(tests)))
