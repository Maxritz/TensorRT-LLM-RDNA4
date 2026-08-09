import os
import platform
import sys
import traceback

from ..logger import logger

IS_FLASHINFER_AVAILABLE = False


def get_env_enable_pdl() -> bool:
    enabled = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    if enabled and not getattr(get_env_enable_pdl, "_printed", False):
        logger.info("PDL enabled")
        setattr(get_env_enable_pdl, "_printed", True)
    return enabled


if platform.system() != "Windows":
    try:
        import flashinfer
        logger.info(f"flashinfer is available: {flashinfer.__version__}")
        IS_FLASHINFER_AVAILABLE = True
    except ImportError:
        traceback.print_exc()
        print(
            "flashinfer is not installed properly, please try pip install or building from source codes"
        )
else:
    _vulkan_backend = os.environ.get("TLLM_VULKAN_BACKEND", "1") == "1"
    if _vulkan_backend:
        try:
            # Import the Vulkan flashinfer shim and register it as 'flashinfer'
            from tensorrt_llm._torch.vulkan_backend import flashinfer as _vfi
            sys.modules["flashinfer"] = _vfi
            import flashinfer  # noqa: F811,E402
            logger.info("flashinfer is available via Vulkan backend")
            IS_FLASHINFER_AVAILABLE = True
        except Exception:
            traceback.print_exc()
            print(
                "Vulkan flashinfer backend is not available. "
                "Set TLLM_VULKAN_BACKEND=0 to disable."
            )
