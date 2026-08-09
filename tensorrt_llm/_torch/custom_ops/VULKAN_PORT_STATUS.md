# Vulkan Port Status (TensorRT-LLM-RDNA4)

## Overview
Ports TensorRT-LLM's PyTorch backend custom ops to run without the TRT-LLM C++
extension (`torch.classes.trtllm.*`), using Vulkan compute exclusively on
AMD RDNA4 GPUs.

## Completed
| File | Change |
|------|--------|
| `torch_custom_ops.py` | Replaced all 7 unconditional `torch.classes.trtllm.*` calls + 1 `FusedMoeRunner` call with `_TLLM_CPP_AVAILABLE` guarded conditionals. Added `_VulkanGemmRunner` and `_VulkanMoERunner` fallback classes that delegate to `vulkan_backend.torch_bridge`. |
| `trtllm_gen_custom_ops.py` | Replaced 5 `torch.classes.trtllm.*` MoE runner calls with guarded conditionals using `_VulkanMoERunner` fallback. |
| `communicator.py` | Wrapped `NcclCommunicatorOp` in `_TLLM_CPP_AVAILABLE` guard; falls back to torch.distributed `PPCommTorch` path. |
| `test_moe.py` | Added `_skip_if_no_cpp_ext` decorator for tests requiring C++ extension. |
| `test_kimi_k3_situ_moe.py` | Added `_cpp_ext_required` skipif decorator. |
| `.gitignore` | Added `build_trace/` and `*.pyd` patterns. |

## Fallback Classes
- **`_VulkanGemmRunner`** — GEMM fallback via `vulkan_backend.torch_bridge.vulkan_gemm`
- **`_VulkanMoERunner`** — MoE fallback: loops over experts, dispatches GEMM and
  activation through `vulkan_backend.torch_bridge`
- **`_VulkanGemmFallback`** — Static GEMM fallback for `TunableRunner` autotuning

## C++ Extension Detection
```python
_TLLM_CPP_AVAILABLE = False
try:
    torch.classes.trtllm.FusedMoeRunner
    _TLLM_CPP_AVAILABLE = True
except (AttributeError, RuntimeError, ModuleNotFoundError):
    pass
```

## ZLUDA Reference
`F:\AI-sglang\zluda` maps CUDA Driver API → HIP via `FromCuda<T,E>` trait conversion
at the FFI boundary. Analogous pattern: we intercept `torch.classes.trtllm.*` calls
at the Python boundary and redirect to `vulkan_backend` (Vulkan compute shaders).
The Vulkan backend stages GPU tensors through host memory (GPU→host→Vulkan device
buffer→shader→host→GPU) — documented in `torch_bridge.py`.

## Remaining Work
1. Add Vulkan compute shader support for FP4/FP8/weight-only quantized GEMM
   (currently only `tllm_vulkan_gemm` and `tllm_vulkan_q8_0_gemm` exist in C++)
2. Add Vulkan compute shaders for MoE routing/permutation/activation
3. Implement direct device-pointer Vulkan path (VK_EXTERNAL_MEMORY) to remove
   host round-trip staging
4. Wire `__init__.py` to auto-import vulkan_backend when C++ extension absent
