# Vulkan Port Status (TensorRT-LLM-RDNA4)

## Overview
Ports TensorRT-LLM's PyTorch backend custom ops to run without the TRT-LLM C++
extension (`torch.classes.trtllm.*`), using Vulkan compute exclusively on
AMD RDNA4 GPUs.

## Completed
| Area | File | Change |
|------|------|--------|
| C++ Ext Guard | `torch_custom_ops.py` | Guarded all 8 `torch.classes.trtllm.*` runner-creation calls with `_TLLM_CPP_AVAILABLE` |
| C++ Ext Guard | `trtllm_gen_custom_ops.py` | Guarded 6 `torch.classes.trtllm.*` MoE runner calls |
| C++ Ext Guard | `communicator.py` | Guarded `NcclCommunicatorOp` call + probe |
| C++ Ext Guard | `cublaslt_utils.py` | Already guarded via `hasattr` check |
| Fallback Runners | `torch_custom_ops.py` | `_VulkanGemmRunner`, `_VulkanMoERunner`, `_VulkanGenMoERunner`, `_VulkanGemmFallback` |
| Activation Shaders | `silu.comp`, `gelu.comp`, `swiglu.comp` | New Vulkan compute shaders |
| C++ Backend | `vulkanBackend.h/cpp`, `vulkan_c_api.h` | Added `launchSilu`/`launchGelu`/`launchSwiglu` + C API |
| Kernel Registry | `vulkanKernelRegistry.h/cpp` | Added `dispatchSilu`/`dispatchGelu`/`dispatchSwiglu` + kernel descriptors |
| Python Bridge | `vulkan_compute.py` | ctypes bindings for `tllm_vulkan_silu/gelu/swiglu` |
| Python Bridge | `torch_bridge.py` | `vulkan_silu`, `vulkan_gelu`, `vulkan_swiglu` wrappers |
| Tests | `test_moe.py`, `test_kimi_k3_situ_moe.py` | Added C++ extension skip decorators |
| Git | `.gitignore` | Added `build_trace/` and `*.pyd` patterns |

## Vulkan Compute Functions (C++ Bridge)
| Function | Shader | Purpose |
|----------|--------|---------|
| `tllm_vulkan_gemm` | `fp16_gemm.comp` | FP16/FP32 GEMM |
| `tllm_vulkan_q8_0_gemm` | `q8_0_gemm.comp` | Quantized GEMM (Q8_0) |
| `tllm_vulkan_elementwise_add` | `elementwise_add.comp` | Elementwise add |
| `tllm_vulkan_rms_norm` | `rms_norm.comp` | RMS normalization |
| `tllm_vulkan_softmax` | `softmax.comp` | Softmax along last dim |
| `tllm_vulkan_attention` | `attention.comp` | SDPA attention |
| `tllm_vulkan_topk` | `topk.comp` | Top-K (sparse attention) |
| `tllm_vulkan_silu` | `silu.comp` | SiLU/swish activation |
| `tllm_vulkan_sigmoid` | `sigmoid.comp` | Sigmoid |
| `tllm_vulkan_gelu` | `gelu.comp` | GELU (tanh approx) |
| `tllm_vulkan_relu` | `relu.comp` | ReLU |
| `tllm_vulkan_swiglu` | `swiglu.comp` | SwiGLU activation |
| `tllm_vulkan_topk_general` | `topk_general.comp` | General top-K per row |

## Fallback Classes
- **`_VulkanGemmRunner`** — GEMM fallback; delegates to `vulkan_backend.torch_bridge.vulkan_gemm`. Implements `get_num_configs`, `run_gemm_profile`, `run_gemm`, `run_batched_gemm`, `get_valid_configs`, `get_num_heuristic_algos`, `get_tactic_num`, `clear_cache`.
- **`_VulkanMoERunner`** — MoE fallback for `fused_moe` path; implements `run_moe` (token routing + expert dispatch + Vulkan GEMM + SwiGLU), `run_moe_min_latency`, `run_gemm_profile`, `_apply_activation` (via Vulkan silu/gelu/swiglu shaders).
- **`_VulkanGenMoERunner`** — MoE fallback for `trtllm_gen_custom_ops` path; implements the gen-style `run_moe` signature with routing_logits/topk_ids/topk_weights.
- **`_VulkanGemmFallback`** — Static GEMM fallback for `TunableRunner` autotuning.

## Staging Model
GPU tensors → host (fp32) → Vulkan device buffer → shader → host → GPU.
Direct device-pointer path (VK_EXTERNAL_MEMORY) is the documented performance follow-up.

## Remaining Work
1. Q8_0 quantized GEMM in MoE path (only `tllm_vulkan_q8_0_gemm` exists; need to wire into `_VulkanGemmRunner.run_gemm`)
2. FP4/MXPF4 quantized MoE (needs dequant + Vulkan gemm)
3. Direct device-pointer Vulkan path (VK_EXTERNAL_MEMORY)
4. Auto-import vulkan_backend when C++ extension absent

## Python Elimination Status
All **compute** operations in fallback runners now go through Vulkan:
- GEMM: `torch.matmul`/`torch.bmm` → `vulkan_gemm`
- Activations: `torch.sigmoid`/`torch.relu`/etc → `vulkan_silu/sigmoid/gelu/relu/swiglu`
- Routing: `torch.topk` → `vulkan_topk_general`
- Softmax: PyTorch → `vulkan_softmax`

Remaining torch calls are **tensor orchestration** only (zeros, ones, chunk,
expand_as, index_add_) — no compute kernels remain in Python fallback paths.
