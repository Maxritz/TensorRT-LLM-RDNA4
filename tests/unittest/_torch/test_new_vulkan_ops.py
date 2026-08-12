"""Quick smoke test for new Vulkan ops: silu, gelu, elementwise_mul, scale_rows"""
import sys, ctypes
import numpy as np

sys.path.insert(0, r"F:\NV\TensorRT-LLM")
from tensorrt_llm._torch.vulkan_backend import vulkan_compute as vc

vk = vc._vk
print("available:", vc.is_available())
print("init:", vk.init(0))
print("active:", vk.is_active())

N = 256
results = []

# elementwise_mul
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
expected = a * b
pa = vk.malloc(N * 4); pb = vk.malloc(N * 4); pout = vk.malloc(N * 4)
vk.memcpy_h2d(pa, a.ctypes.data, N * 4)
vk.memcpy_h2d(pb, b.ctypes.data, N * 4)
vk.elementwise_mul(pa, pb, pout, N)
result = np.empty(N, dtype=np.float32)
vk.memcpy_d2h(result.ctypes.data, pout, N * 4)
vk.device_synchronize()
ok = np.allclose(expected, result, atol=1e-5)
print("elementwise_mul:", "PASS" if ok else "FAIL", "max_diff=", np.max(np.abs(expected-result)))
results.append(ok)
vk.free(pa); vk.free(pb); vk.free(pout)

# silu
x = np.random.randn(N).astype(np.float32)
expected_silu = x / (1.0 + np.exp(-x))
px = vk.malloc(N * 4); po = vk.malloc(N * 4)
vk.memcpy_h2d(px, x.ctypes.data, N * 4)
vk.silu(px, po, N)
result_silu = np.empty(N, dtype=np.float32)
vk.memcpy_d2h(result_silu.ctypes.data, po, N * 4)
vk.device_synchronize()
ok = np.allclose(expected_silu, result_silu, atol=1e-5)
print("silu:", "PASS" if ok else "FAIL", "max_diff=", np.max(np.abs(expected_silu-result_silu)))
results.append(ok)
vk.free(px); vk.free(po)

# gelu
x2 = np.random.randn(N).astype(np.float32)
GELU_COEF_A = 0.044715
SQRT_2_OVER_PI = 0.79788456080286535587989211986876
val = SQRT_2_OVER_PI * x2 * (1.0 + GELU_COEF_A * x2**2)
expected_gelu = 0.5 * x2 * (2.0 - 2.0 / (np.exp(2.0 * val) + 1.0))
px2 = vk.malloc(N * 4); po2 = vk.malloc(N * 4)
vk.memcpy_h2d(px2, x2.ctypes.data, N * 4)
vk.gelu(px2, po2, N)
result_gelu = np.empty(N, dtype=np.float32)
vk.memcpy_d2h(result_gelu.ctypes.data, po2, N * 4)
vk.device_synchronize()
ok = np.allclose(expected_gelu, result_gelu, atol=1e-5)
print("gelu:", "PASS" if ok else "FAIL", "max_diff=", np.max(np.abs(expected_gelu-result_gelu)))
results.append(ok)
vk.free(px2); vk.free(po2)

# scale_rows
rows, cols = 4, 64
inp = np.random.randn(rows, cols).astype(np.float32)
scale = np.random.randn(rows).astype(np.float32)
expected_sr = inp * scale[:, None]
pinp = vk.malloc(rows * cols * 4); pscale = vk.malloc(rows * 4); psr = vk.malloc(rows * cols * 4)
vk.memcpy_h2d(pinp, inp.ctypes.data, rows * cols * 4)
vk.memcpy_h2d(pscale, scale.ctypes.data, rows * 4)
vk.scale_rows(pinp, pscale, psr, rows, cols)
result_sr = np.empty(rows * cols, dtype=np.float32)
vk.memcpy_d2h(result_sr.ctypes.data, psr, rows * cols * 4)
vk.device_synchronize()
ok = np.allclose(expected_sr.ravel(), result_sr, atol=1e-5)
print("scale_rows:", "PASS" if ok else "FAIL", "max_diff=", np.max(np.abs(expected_sr.ravel()-result_sr)))
results.append(ok)
vk.free(pinp); vk.free(pscale); vk.free(psr)

print()
print("ALL PASS" if all(results) else "SOME FAILED")
