import os
import platform

_on_windows_stub = (platform.system() == "Windows"
                    and os.environ.get("TLLM_VULKAN_BACKEND", "0") == "1")

if _on_windows_stub:
    _LAZY_NAMES = {
        "OpenAIServer": "openai_server",
        "OpenAIDisaggServer": "openai_disagg_server",
    }

    def __getattr__(name):
        if name in _LAZY_NAMES:
            import importlib
            mod = importlib.import_module(f"tensorrt_llm.serve.{_LAZY_NAMES[name]}")
            return getattr(mod, name)
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}")


else:
    from .openai_disagg_server import OpenAIDisaggServer
    from .openai_server import OpenAIServer

__all__ = ['OpenAIServer', 'OpenAIDisaggServer']
