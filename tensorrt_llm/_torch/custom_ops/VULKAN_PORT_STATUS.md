# Vulkan Port Status (TensorRT-LLM-RDNA4)

## Overview
This branch ports TensorRT-LLM's PyTorch backend custom ops to work without the
TRT-LLM C++ extension (`torch.classes.trtllm.*`), enabling operation on AMD RDNA4
GPUs via Vulkan or fallback PyTorch paths.

## Completed
| File | Change |
|------|--------|
| `torch_custom_ops.py` | Replaced all 7 unconditional `torch.classes.trtllm.*` calls with `_TLLM_CPP_AVAILABLE` guarded conditionals. Added fallback runner classes: `_VulkanGemmRunner`, `_VulkanMoERunner`, `_TorchGemmFallback`. |
| `trtllm_gen_custom_ops.py` | (Pending — 5 call sites to fix) |

## Fallback Classes
- **`_VulkanGemmRunner`** — GEMM fallback; delegates to `vulkan_compute` bridge when available, falls back to `torch.nn.functional.linear`.
- **`_VulkanMoERunner`** — MoE fusion fallback; uses PyTorch ops for routing/GEMM/activation (correctness-first).
- **`_TorchGemmFallback`** — Static GEMM fallback for the `TunableRunner` autotuning path.

## C++ Extension Detection
```python
_TLLM_CPP_AVAILABLE = False
try:
    torch.classes.trtllm.FusedMoeRunner  # noqa: B018
    _TLLM_CPP_AVAILABLE = True
except (AttributeError, RuntimeError, ModuleNotFoundError):
    pass
```

## Remaining Work
1. Fix `trtllm_gen_custom_ops.py` — 5 `torch.classes.trtllm.*` call sites.
2. Fix `tensorrt_llm/_torch/distributed/communicator.py` — `NcclCommunicatorOp` call site.
3. Replace `_VulkanMoERunner` torch ops with Vulkan compute shader dispatches.
4. Add `torch.classes.trtllm` references in tests to fall back gracefully.

## ZLUDA Reference
`F:\AI-sglang\zluda` translates CUDA→ROCr/ROCm. Its patterns for `cudaMalloc`,
`cudaLaunchKernel`→Vulkan `vkCmdDispatch` provide guidance for our Vulkan compute
bridge (`vulkan_backend/vulkan_compute.py`).
