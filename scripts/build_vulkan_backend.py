#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build the Vulkan backend shared library and stage it for the Python package.

Usage (from the repository root)::

    python scripts/build_vulkan_backend.py

Runs the CMake build of the ``vulkan_backend`` target defined in
``cpp/trace_test/CMakeLists.txt`` and copies the resulting
``vulkan_backend.dll`` / ``libvulkan_backend.so`` into
``tensorrt_llm/_torch/vulkan_backend/`` so that the ctypes loader in
``vulkan_compute.py`` (whose first search path is that package directory)
finds it automatically -- no ``TLLM_VULKAN_LIB_DIR`` required for editable/dev
installs.

The produced library exposes the ``tllm_vulkan_*`` C-ABI entry points. Loading
it does not require a GPU; only ``tllm_vulkan_init`` (vkCreateInstance) needs a
Vulkan-capable device.
"""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "cpp" / "trace_test"
BUILD_DIR = REPO_ROOT / "build_vulkan"
PKG_DIR = REPO_ROOT / "tensorrt_llm" / "_torch" / "vulkan_backend"


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _lib_names() -> tuple[str, ...]:
    if _is_windows():
        return ("vulkan_backend.dll", "libvulkan_backend.dll")
    return ("libvulkan_backend.so", "vulkan_backend.so")


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def _find_built_lib() -> Path | None:
    names = _lib_names()
    for d in (BUILD_DIR / "bin", BUILD_DIR / "lib"):
        if d.is_dir():
            for name in names:
                p = d / name
                if p.is_file():
                    return p
    for name in names:
        found = list(BUILD_DIR.rglob(name))
        if found:
            return found[0]
    return None


def build() -> None:
    _run(["cmake", "-S", str(SRC_DIR), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"])
    _run(["cmake", "--build", str(BUILD_DIR), "--target", "vulkan_backend", "--config", "Release"])


# mingw/Strawberry compiler-runtime companions that must sit next to the
# backend DLL for ctypes.CDLL to resolve transitive dependencies on Windows.
# The backend links the mingw CRT dynamically, so all of these must be
# loadable from the package directory at ctypes.CDLL time.
_RUNTIME_DLLS = ("libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll")


def _candidate_runtime_dirs(lib: Path) -> list[Path]:
    dirs = [lib.parent]
    cache = BUILD_DIR / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text(errors="replace").splitlines():
            if "CMAKE_CXX_COMPILER:" in line and "=" in line:
                compiler = Path(line.split("=", 1)[1].strip())
                dirs.append(compiler.parent)
                break
    return dirs


def stage() -> Path:
    lib = _find_built_lib()
    if lib is None:
        raise SystemExit(
            f"Built vulkan_backend library not found under {BUILD_DIR} "
            f"(expected one of {_lib_names()})."
        )
    PKG_DIR.mkdir(parents=True, exist_ok=True)
    dest = PKG_DIR / lib.name
    shutil.copy2(lib, dest)
    print(f"Staged {lib.name} -> {dest}")
    # Stage compiler-runtime companions (mingw shared CRT) from the build's
    # output dir and/or the toolchain bin dir recorded in the CMake cache.
    for name in _RUNTIME_DLLS:
        for d in _candidate_runtime_dirs(lib):
            rt = d / name
            if rt.is_file():
                shutil.copy2(rt, PKG_DIR / name)
                print(f"Staged runtime {name} -> {PKG_DIR / name}")
                break
    print("The ctypes loader in vulkan_compute.py will find the library in the package directory.")
    return dest


def main() -> int:
    build()
    stage()
    return 0


if __name__ == "__main__":
    sys.exit(main())
