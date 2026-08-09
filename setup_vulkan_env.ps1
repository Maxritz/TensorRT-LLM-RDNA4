<#
.SYNOPSIS
    Sets up a Vulkan/ROCm development environment for TensorRT-LLM on Windows (AMD RDNA4).

.DESCRIPTION
    Creates a Python 3.12 venv, installs all PyPI dependencies, links PyTorch ROCm
    from a local source if the public ROCm index lacks Windows wheels, and writes
    an activation script with the correct PATH / env-var configuration.

    After setup, activate with:
        F:\TENSORRT-LLM\.venv\Scripts\Activate.ps1
        .\activate_vulkan_env.ps1

.PARAMETER TorchSource
    Path to a directory whose site-packages contains torch (e.g. a sglang venv).
    If not provided, the script searches common locations.

.EXAMPLE
    .\setup_vulkan_env.ps1
    .\setup_vulkan_env.ps1 -TorchSource "F:\AI-sglang\sglang-windows\.venv312\Lib\site-packages"
#>

param(
    [string]$TorchSource,
    [string]$VulkanSdkBin = "C:\VulkanSDK\1.4.357.0\Bin",
    [string]$RepoRoot     = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [string]$VenvPath     = "F:\TENSORRT-LLM\.venv"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Vulkan/ROCm TensorRT-LLM Environment Setup ===" -ForegroundColor Cyan
Write-Host "Repo root:  $RepoRoot"
Write-Host "Venv path:  $VenvPath"
Write-Host "Vulkan SDK: $VulkanSdkBin"
Write-Host ""

# 1. Create venv
if (-Not (Test-Path "$VenvPath\Scripts\python.exe")) {
    Write-Host "[1/5] Creating venv..." -ForegroundColor Yellow
    uv venv "$VenvPath" --python 3.12
} else {
    Write-Host "[1/5] Venv already exists at $VenvPath" -ForegroundColor Green
}

$VenvPython = "$VenvPath\Scripts\python.exe"

# 2. Install PyPI dependencies
Write-Host "[2/5] Installing PyPI dependencies..." -ForegroundColor Yellow
& uv pip install --python $VenvPython @(
    "numpy>=2.0",
    "blake3",
    "nvtx",
    "StrEnum",
    "mpi4py",
    "pydantic>=2.9.1",
    "omegaconf",
    "pillow",
    "pyyaml",
    "aenum",
    "pyzmq",
    "tqdm"
) | Out-Null

# 3. Link or install PyTorch ROCm
Write-Host "[3/5] Checking for PyTorch ROCm..." -ForegroundColor Yellow
$PyTorchOk = & $VenvPython -c "import torch; assert torch.version.hip is not None; print('ok')" 2>$null
if ($PyTorchOk -ne "ok") {
    Write-Host "  PyTorch ROCm not found in venv. Attempting to link from local source..." -ForegroundColor Yellow
    if (-not $TorchSource) {
        # Search common locations
        $candidates = @(
            "F:\AI-sglang\sglang-windows\.venv312\Lib\site-packages",
            "C:\sglang-windows\.venv312\Lib\site-packages",
            "D:\sglang-windows\.venv312\Lib\site-packages"
        )
        foreach ($c in $candidates) {
            if (Test-Path $c) { $TorchSource = $c; break }
        }
    }
    if ($TorchSource -and (Test-Path $TorchSource)) {
        Write-Host "  Linking torch from: $TorchSource" -ForegroundColor Green
        $pthFile = "$VenvPath\Lib\site-packages\__rocm_torch_link__.pth"
        Set-Content -Path $pthFile -Value $TorchSource -Encoding ascii
    } else {
        Write-Host "  ERROR: No local PyTorch ROCm found." -ForegroundColor Red
        Write-Host "  PyTorch ROCm for Windows is not available on public PyPI." -ForegroundColor Red
        Write-Host "  Please provide a source venv with torch+rocm installed." -ForegroundColor Red
        Write-Host "  Example: .\setup_vulkan_env.ps1 -TorchSource 'F:\path\to\venv\Lib\site-packages'" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  PyTorch ROCm already available." -ForegroundColor Green
}

# 4. Verify torch + CUDA/ROCm
Write-Host "[4/5] Verifying PyTorch..." -ForegroundColor Yellow
& $VenvPython -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  cuda available: {torch.cuda.is_available()}')
print(f'  hip: {torch.version.hip}')
assert torch.cuda.is_available(), 'GPU not available!'
assert torch.version.hip is not None, 'Not a ROCm build!'
print('  OK')
" 2>&1

# 5. Create activation script
Write-Host "[5/5] Creating activation script..." -ForegroundColor Yellow
$VulkanBackendDir = "$RepoRoot\tensorrt_llm\_torch\vulkan_backend"
$ActivationScript = @"
# Vulkan backend environment for TensorRT-LLM
# Source this after activating the venv:
#   .\.venv\Scripts\Activate.ps1
#   .\activate_vulkan_env.ps1

# Vulkan SDK
`$env:PATH = "$VulkanSdkBin;$VulkanBackendDir;`$env:PATH"

# Activate Vulkan backend on Windows
`$env:TLLM_VULKAN_BACKEND = "1"

# TensorRT-LLM source (editable install)
`$env:PYTHONPATH = "$RepoRoot"

Write-Host "Vulkan backend activated: TLLM_VULKAN_BACKEND=1, Vulkan SDK in PATH"
Write-Host "Run tests: pytest tests\unittest\_torch\test_vulkan_ops.py --noconftest --override-ini=addopts="
"@

Set-Content -Path "$RepoRoot\activate_vulkan_env.ps1" -Value $ActivationScript -Encoding UTF8
Write-Host "  Written: $RepoRoot\activate_vulkan_env.ps1" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "To run tests:" -ForegroundColor Yellow
& $VenvPython -c "
import torch
print(f'  torch {torch.__version__} (ROCm {torch.version.hip})')
" 2>&1
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "  .\activate_vulkan_env.ps1" -ForegroundColor Gray
Write-Host "  pytest tests\unittest\_torch\test_vulkan_ops.py --noconftest --override-ini=addopts= -v" -ForegroundColor Gray
