import os
import platform

_on_windows_stub = (platform.system() == "Windows"
                    and os.environ.get("TLLM_VULKAN_BACKEND", "0") == "1")

# Lazy import mappings for the Vulkan/Windows path.
# On this path, submodules are imported on first access via __getattr__
# to break circular import chains (executor.result -> llmapi -> llm -> executor).
_LAZY_SUBMODULES = {
    "PostprocWorker": "postproc_worker",
    "PostprocWorkerConfig": "postproc_worker",
    "GenerationRequest": "postproc_worker",
    "LoRARequest": "postproc_worker",
    "PromptAdapterRequest": "postproc_worker",
    "GenerationExecutorWorker": "postproc_worker",
    "PostprocParams": "postproc_worker",
    "GenerationExecutorProxy": "proxy",
    "CompletionOutput": "result",
    "GenerationResultBase": "result",
    "DetokenizedGenerationResultBase": "result",
    "GenerationResult": "result",
    "IterationResult": "result",
    "Logprob": "result",
}

# Eager imports for the non-Vulkan path (C++ bindings available)
if not _on_windows_stub:
    from .executor import *  # noqa: F401,F403
    from .postproc_worker import *  # noqa: F401,F403
    from .proxy import *  # noqa: F401,F403
    from .request import *  # noqa: F401,F403
    from .result import *  # noqa: F401,F403
    from .utils import EngineDeadError, RequestError
    from .worker import *  # noqa: F401,F403


def __getattr__(name):
    if _on_windows_stub and name in _LAZY_SUBMODULES:
        import importlib
        submod_name = _LAZY_SUBMODULES[name]
        mod = importlib.import_module(f"tensorrt_llm.executor.{submod_name}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PostprocWorker",
    "PostprocWorkerConfig",
    "GenerationRequest",
    "LoRARequest",
    "PromptAdapterRequest",
    "GenerationExecutorWorker",
    "GenerationExecutorProxy",
    "RequestError",
    "EngineDeadError",
    "CompletionOutput",
    "GenerationResultBase",
    "DetokenizedGenerationResultBase",
    "GenerationResult",
    "IterationResult",
]
