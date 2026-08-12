#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

"""
Export a HuggingFace LLM model to ExecuTorch .pte format with Vulkan backend.
This replaces the TRT-LLM PyTorch engine with ExecuTorch + Vulkan at runtime.

The export phase uses torch for graph capture, but the runtime uses
ExecuTorch's C++ runtime with Vulkan backend (no torch needed at inference).

Usage:
    python export_trtllm_to_et.py --model <hf_model> --output model.pte --vulkan
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Export HF model to ExecuTorch .pte with Vulkan backend")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to HuggingFace model or model repo ID")
    parser.add_argument("-o", "--output", type=str, default="output/model.pte",
                        help="Output .pte file path")
    parser.add_argument("-V", "--vulkan", action="store_true", default=True,
                        help="Use Vulkan backend")
    parser.add_argument("--vulkan-force-fp16", action="store_true", default=True,
                        help="Force fp16 for Vulkan")
    parser.add_argument("-kv", "--use_kv_cache", action="store_true", default=True,
                        help="Use KV cache for generation")
    parser.add_argument("--use_sdpa_with_kv_cache", action="store_true", default=True,
                        help="Use SDPA with KV cache")
    parser.add_argument("-d", "--dtype-override", type=str, default="fp16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Model dtype")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--max_context_length", type=int, default=2048,
                        help="Maximum context length")
    parser.add_argument("-m", "--metadata", type=str, default=None,
                        help='Metadata JSON string')
    parser.add_argument("-v", "--verbose", action="store_true", default=True,
                        help="Verbose logging")
    parser.add_argument("--disable_dynamic_shape", action="store_true", default=False,
                        help="Disable dynamic shape")
    parser.add_argument("--generate_full_logits", action="store_true", default=False,
                        help="Generate full logits")
    parser.add_argument("-n", "--output_name", type=str, default=None,
                        help="Override output filename")
    
    args = parser.parse_args()
    
    # Build command for the existing export script
    cmd = [sys.executable, "-m", "examples.models.llama.export_llama"]
    
    # Use HF model path directly (non-hydra mode)
    cmd.extend(["--model", args.model])
    
    if args.output_name:
        cmd.extend(["-n", args.output_name])
    else:
        # Derive output name from output path
        cmd.extend(["-n", os.path.basename(args.output) if args.output else "model.pte"])
    
    cmd.extend(["--max_seq_length", str(args.max_seq_length)])
    cmd.extend(["--max_context_length", str(args.max_context_length)])
    cmd.extend(["-d", args.dtype_override])
    
    if args.use_kv_cache:
        cmd.append("-kv")
    if args.use_sdpa_with_kv_cache:
        cmd.append("--use_sdpa_with_kv_cache")
    if args.vulkan:
        cmd.append("-V")
    if args.vulkan_force_fp16:
        cmd.append("--vulkan-force-fp16")
    if args.verbose:
        cmd.append("-v")
    if args.metadata:
        cmd.extend(["-m", args.metadata])
    if not args.disable_dynamic_shape:
        pass  # dynamic shape is on by default
    else:
        cmd.append("--disable_dynamic_shape")
    if args.generate_full_logits:
        cmd.append("--generate_full_logits")
    
    # Set output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cmd.extend(["--output-dir", output_dir])
    
    et_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "executorch")
    os.chdir(et_root)
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
