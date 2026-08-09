import os
import platform

if platform.system() == "Windows" and os.environ.get("TLLM_VULKAN_BACKEND", "0") == "1":
    # Windows + Vulkan path: skip heavy LLM import chain.
    # Submodules will be imported lazily when needed.
    pass
else:
    from .llm import LLM

__all__ = ["LLM"]
