# Vulkan LLM Port Audit: Python → C++

**Last updated:** Based on codebase analysis of `tensorrt_llm/_torch/vulkan_backend/` (Python) vs `cpp/tensorrt_llm/common/` (C++ Vulkan).

---

## Table of Contents

1. [Already Ported in C++](#already-ported-in-c)
2. [Tier 1: Foundation Ops — Status Summary](#tier-1-foundation-ops--status-summary)
3. [Tier 2: Mid-Level Features](#tier-2-mid-level-features)
4. [Tier 3: Advanced Features](#tier-3-advanced-features)
5. [Porting Plan & Effort Estimates](#porting-plan--effort-estimates)

---

## Already Ported in C++

### Core files (`cpp/tensorrt_llm/common/`)
| File | Role |
|------|------|
| `vulkan_inference.{h,cpp}` | `Qwen2VulkanInference` class — model loading, forward, transformerBlock, generate |
| `vulkanBackend.{h,cpp}` | Vulkan backend singleton with `launch*` methods |
| `vulkan_c_api.h` | ctypes `extern "C"` bridge functions (host-facing API) |
| `vulkanKernelRegistry.cpp` | 27 SPIR-V shaders registered and dispatched |
| `gguf_loader.h/.cpp` | GGUF parsing + Q4_K/Q5_K/Q6_K/Q8_0 dequantization |
| `safetensors_loader.h/.cpp` | Safetensors loading |
| `vulkan_tokenizer.h/.cpp` | Simple BPE tokenizer |

### Shaders registered (`cpp/tensorrt_llm/kernels/vulkanKernelRegistry.cpp`)
`elementwise_add.comp`, `rms_norm.comp`, `layer_norm.comp`, `q8_0_gemm.comp`, `fp16_gemm.comp`,
`softmax.comp`, `attention.comp`, `topk.comp`, `mla_fmha.comp`, `mla_fmha_prefill.comp`,
`spec_accept.comp`, `tree_spec_build.comp`, `tree_spec_greedy_verify.comp`, `tree_spec_rejection.comp`,
`kv_cache_update_2d.comp`, `silu.comp`, `sigmoid.comp`, `gelu.comp`, `swiglu.comp`,
`relu.comp`, `sigmoid_mul.comp`, `elementwise_mul.comp`, `scale_rows.comp`, `cast.comp`,
`index_add.comp`, `topk_general.comp`, `gather.comp`, `fill.comp`, `compare_eq.comp`,
`append_paged_kv_cache.comp`

---

## Tier 1: Foundation Ops — Status Summary

| Python Op (numpy_bridge.py) | C++ Export | Shader | Status |
|----------------------------|-----------|--------|--------|
| `vulkan_gemm` | `tllm_vulkan_gemm` | `fp16_gemm.comp` | DONE |
| `vulkan_softmax` | `tllm_vulkan_softmax` | `softmax.comp` | DONE |
| `vulkan_rms_norm` | `tllm_vulkan_rms_norm` | `rms_norm.comp` | DONE |
| `vulkan_layer_norm` | `tllm_vulkan_layer_norm` | `layer_norm.comp` | DONE |
| `vulkan_elementwise_add` | `tllm_vulkan_elementwise_add` | `elementwise_add.comp` | DONE |
| `vulkan_elementwise_mul` | `tllm_vulkan_elementwise_mul` | `elementwise_mul.comp` | DONE |
| `vulkan_silu` | `tllm_vulkan_silu` | `silu.comp` | DONE |
| `vulkan_sigmoid` | `tllm_vulkan_sigmoid` | `sigmoid.comp` | DONE |
| `vulkan_gelu` | `tllm_vulkan_gelu` | `gelu.comp` | DONE |
| `vulkan_swiglu` | `tllm_vulkan_swiglu` | `swiglu.comp` | DONE |
| `vulkan_relu` | `tllm_vulkan_relu` | `relu.comp` | DONE |
| `vulkan_sigmoid_mul` | `tllm_vulkan_sigmoid_mul` | `sigmoid_mul.comp` | DONE |
| `vulkan_scale_rows` | `tllm_vulkan_scale_rows` | `scale_rows.comp` | DONE |
| `vulkan_cast` | `tllm_vulkan_cast` | `cast.comp` | DONE |
| `vulkan_index_add` | `tllm_vulkan_index_add` | `index_add.comp` | DONE |
| `vulkan_gather` | `tllm_vulkan_gather` | `gather.comp` | DONE |
| `vulkan_fill` | `tllm_vulkan_fill` | `fill.comp` | DONE |
| `vulkan_compare_eq` | `tllm_vulkan_compare_eq` | `compare_eq.comp` | DONE |
| `vulkan_topk_general` | `tllm_vulkan_topk_general` | `topk_general.comp` | DONE |
| `vulkan_attention` | `tllm_vulkan_attention` | `attention.comp` | DONE |
| `vulkan_q8_0_gemm` | `tllm_vulkan_q8_0_gemm` | `q8_0_gemm.comp` | DONE |
| `vulkan_topk` | `tllm_vulkan_topk` | `topk.comp` | DONE |
| `vulkan_memory ops` | `malloc`/`free`/`memcpy_*` | N/A | DONE |

**vulkan_ops.py (CPU helpers):**
| Python Function | C++ Equivalent | Status |
|----------------|---------------|--------|
| `embedding()` | Inline lookup in `forward()` | DONE (inline) |
| `rope()` | `apply_rope()` in `vulkan_inference.cpp:54` | DONE |
| `precompute_rope_freqs()` | `precompute_rope_freqs()` in `vulkan_inference.h:25` | DONE |
| `softmax_cpu()` | `softmax_cpu()` in `vulkan_inference.cpp:76` | DONE |
| `sample_top_k_top_p()` | `sample_top_k_top_p()` in `vulkan_inference.cpp:109` | DONE |
| `rms_norm_cpu()` | CPU fallback in `vulkan_ops.py`; C++ uses shader | DONE |
| `topk_cpu()` | N/A (GPU `topk.comp`) | DONE |
| `sample_from_probs()` | `sample_top_k_top_p()` + temperature | DONE |
| `append_paged_kv_cache()` | N/A (see Tier 3) | N/A |

**weight_loader.py:**
| Feature | Status |
|---------|--------|
| GGUF loading (F32, F16, BF16, Q4_K, Q5_K, Q6_K, Q8_0) | DONE |
| Safetensors loading | DONE |
| Metadata extraction (layers, hidden, heads, vocab, rope_theta, norm_eps) | DONE |

**vulkan_infer.py / vulkan_entry.py:**
Full end-to-end GGUF/safetensors inference pipeline — DONE

---

## Tier 2: Mid-Level Features

| Python File | Feature | C++ Status | Notes |
|-------------|---------|------------|-------|
| `sampling.py` | Softmax w/ temp scaling | PARTIAL | C++ has `softmax_cpu`; Python wraps with torch for Vulkan path |
| `sampling.py` | Top-k/Top-p from probs | PARTIAL | C++ has `sample_top_k_top_p` (CPU); GPU `topk.comp` exists |
| `sampling.py` | Top-k/Top-p from logits | MISSING | C++ applies temp then uses `sample_top_k_top_p` — covered |
| `attention.py` | Paged KV prefill wrapper | PARTIAL | Vulkan `attention.comp` is basic (no paging) |
| `attention.py` | Paged KV decode wrapper | PARTIAL | `append_paged_kv_cache.comp` exists but not wired in `vulkan_inference.cpp` |
| `attention.py` | Ragged KV prefill | MISSING | No shader or C++ path |
| `kv_cache.py` | Paged KV cache append | PARTIAL | `tllm_vulkan_append_paged_kv_cache` C API exists; shader exists |
| `kv_cache.py` | MLA paged KV | MISSING | `mla_fmha*.comp` exist but not wired to paged KV |
| `norm.py` | RMS norm w/ fp4 quant | MISSING | `fp4` quantization not in C++; C++ uses basic `rms_norm.comp` |
| `norm.py` | `gemma_rmsnorm` | MISSING | Not needed for Qwen2; add if Gemma support needed |
| `norm.py` | `fused_add_rmsnorm` | MISSING | Add if transformer-block fusion needed |
| `fused_moe.py` | B12xMoE wrapper | PARTIAL | `VulkanMoERunner` in C++ (`vulkanMoERunner.cpp`); `tllm_vulkan_moe_runner_*` C API exists |
| `mla.py` | MLA attention wrapper | MISSING | `mla_fmha.comp`/`mla_fmha_prefill.comp` exist; no C++ wrapper or Python wiring |

---

## Tier 3: Advanced Features

### Speculative Decoding
| C++ File | Export | Status |
|----------|--------|--------|
| `vulkan_c_api.h:489` | `tllm_vulkan_tree_spec_build` | DONE (C API) |
| `vulkan_c_api.h:510` | `tllm_vulkan_tree_spec_greedy_verify` | DONE |
| `vulkan_c_api.h:525` | `tllm_vulkan_tree_spec_rejection` | DONE |
| `vulkan_c_api.h:455` | `tllm_vulkan_spec_accept` | DONE |
| `vulkan_c_api.h:475` | `tllm_vulkan_kv_cache_update_2d` | DONE |
| `kernelRegistry.cpp` | `spec_accept.comp`, `tree_spec_*.comp` | DONE |
| `vulkanBackend.h:166-199` | `launchSpecDecode...` / `launchTreeSpec...*` | DONE |
| `vulkanMoERunner.cpp` | N/A | N/A |

**Gap:** C++ has full API surface but `vulkan_inference.cpp` `generate()` does NOT use speculative decoding. Python `forward_pass.py` also has no spec-decoding integration. Both are standalone.

### Mixture of Experts (MoE)
| C++ File | Export | Status |
|----------|--------|--------|
| `vulkanMoERunner.h/.cpp` | `VulkanMoERunner` class | DONE |
| `vulkan_c_api.h:540-620` | `tllm_vulkan_moe_runner_create/destroy/run/gemm_profile` | DONE |
| `kernelRegistry.cpp` | (uses GEMM for experts) | DONE |

**Gap:** Python `fused_moe.py` `B12xMoEWrapper.__call__` uses PyTorch implementation, not Vulkan C API. Needs bridging to `tllm_vulkan_moe_runner_run`.

### Multi-Layer Attention (MLA)
| C++ File | Export | Status |
|----------|--------|--------|
| `vulkanBackend.h:210-217` | `launchMlaFmha`, `launchMlaFmhaPrefill` | DONE |
| `kernelRegistry.cpp` | `mla_fmha.comp`, `mla_fmha_prefill.comp` | DONE |
| `vulkan_c_api.h` | NOT EXPORTED | MISSING |
| `vulkan_inference.cpp` | Not used | MISSING |

**Gap:** MLA shaders exist but are NOT exported via C API and NOT used in `Qwen2VulkanInference`. Python `mla.py` has wrapper class but uses PyTorch fallback.

### Quantization (FP4/FP8/MX)
| Python File | Feature | C++ Status |
|-------------|---------|------------|
| `fp4_quantization.py` | FP4/MXFP4 quantization | MISSING (stubs only: `return x`) |
| `fp8_quantization.py` | MXFP8 quantization | MISSING (stubs only) |
| `norm.py` | `rmsnorm_fp4quant` | MISSING |

**Gap:** Full FP4/FP8 quantization is stubbed in Python and absent from C++. These are advanced quantization formats used in TRT-LLM for weight compression; would require new Vulkan shaders.

### Communication / Distributed
| Python File | Feature | C++ Status |
|-------------|---------|------------|
| `comm/mnnvl.py` | GPUDirect + NIXL | MISSING |
| `comm/trtllm_alltoall.py` | All-to-all | MISSING |
| `comm/__init__.py` | Comm wrappers | MISSING |

**Gap:** Distributed comm (NIXL, UCX, MPI all-reduce) is entirely Python-only. C++ has no comm layer.

---

## Porting Plan & Effort Estimates

### Phase 1: Quick Wins (already mostly done)
- Fix `Qwen2VulkanInference::generate()` to use GPU sampling (`sample_top_k_top_p`) instead of always downloading logits to CPU.
- Wire `append_paged_kv_cache` C API into `forward()` for paged KV support.

### Phase 2: Intermediate (1-2 weeks)
| Task | Files | Est. Effort |
|------|-------|-------------|
| Paged KV cache integration in forward path | `vulkan_inference.cpp`, `vulkanMemoryAllocator.cpp` | 3-4 days |
| Spec decode integration in `Qwen2VulkanInference::generate()` | `vulkan_inference.cpp`, `vulkan_c_api.h` | 2-3 days |
| MLA C API export (`tllm_vulkan_mla_fmha_*`) | `vulkan_c_api.h` | 1-2 days |
| Fuse SwiGLU in transformerBlock | `vulkan_inference.cpp` | 1 day |

### Phase 3: Advanced (2-4 weeks)
| Task | Files | Est. Effort |
|------|-------|-------------|
| FP4 quantization kernels | New `.comp` shaders, `fp4_quantization.py` | 1 week |
| FP8 (MXFP8) quantization | New `.comp` shaders, `fp8_quantization.py` | 1 week |
| B12xMoE Vulkan path | Wire `fused_moe.py` to `tllm_vulkan_moe_runner_run` | 3-5 days |
| Distributed comm (NIXL/UCX) | New C++ comm layer, `comm/*.py` | 2+ weeks |
| MLA wrapper class integration | `mla.py`, `vulkan_inference.cpp` | 1 week |

### File Mapping Reference
| Python file | C++ counterpart |
|-------------|-----------------|
| `vulkan_compute.py` | `vulkan_c_api.h` + `vulkanBackend.cpp` |
| `numpy_bridge.py` | `vulkan_c_api.h` + inline in `vulkan_inference.cpp` |
| `forward_pass.py` | `vulkan_inference.cpp` |
| `weight_loader.py` | `gguf_loader.cpp` + `safetensors_loader.cpp` |
| `vulkan_ops.py` | CPU helpers in `vulkan_inference.cpp` |
| `sampling.py` | `softmax_cpu` + `sample_top_k_top_p` in `vulkan_inference.cpp` |
| `attention.py` | `vulkanBackend::launchAttention` + `launchMlaFmha` |
| `norm.py` | `launchRmsNorm` / `launchLayerNorm` |
| `kv_cache.py` | `launchAppendPagedKVCache` |
| `fused_moe.py` | `vulkanMoERunner.cpp` |
| `mlp.py` | `transformerBlock` in `vulkan_inference.cpp` |
| `fp4_quantization.py` | NEW — no C++ equivalent |
| `fp8_quantization.py` | NEW — no C++ equivalent |
| `comm/*.py` | NEW — no C++ equivalent |
| `rope.py` | `apply_rope` in `vulkan_inference.cpp` |

---

## Key Findings

1. **Foundation ops (Tier 1): Fully ported** — every numpy_bridge op has a C++ `extern "C"` export and SPIR-V shader.

2. **Advanced features (Tier 3): API surface exists but integration is incomplete** — speculative decoding, MoE, and MLA have shaders and C APIs, but `Qwen2VulkanInference` class doesn't use them.

3. **Quantization stubs: Python stubs and C++ have none** — FP4/FP8 quantization is pure stub (`return x`) in Python; C++ has no equivalent at all.

4. **Distributed comm: Pure Python** — comm layer (`comm/mnnvl.py`, `comm/trtllm_alltoall.py`) has no C++ counterpart; entire path is Python.

5. **One-warp-per-RHS column solver**: Each GEMM dispatch in `vulkan_c_api.h` handles one output column slice at a time (see `gemm` in `vulkan_compute.py`), matching the `Qwen2VulkanInference::forward()` chunked-LM-head pattern (`CHUNK_SIZE = 16384` in `vulkan_inference.cpp:768`).