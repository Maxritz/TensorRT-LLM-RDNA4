# Vulkan backend environment for TensorRT-LLM
# Source this after activating the venv:
#   .\.venv\Scripts\Activate.ps1
#   .\activate_vulkan_env.ps1

# Vulkan SDK
$env:PATH = "C:\VulkanSDK\1.4.357.0\Bin;F:\TENSORRT-LLM\tensorrt_llm\_torch\vulkan_backend;$env:PATH"

# Activate Vulkan backend on Windows
$env:TLLM_VULKAN_BACKEND = "1"

# TensorRT-LLM source (editable install)
$env:PYTHONPATH = "F:\TENSORRT-LLM"

Write-Host "Vulkan backend activated: TLLM_VULKAN_BACKEND=1, Vulkan SDK in PATH"
Write-Host "Run tests: pytest tests\unittest\_torch\test_vulkan_ops.py --noconftest --override-ini=addopts="
