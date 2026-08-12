cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(vk_executor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Vulkan SDK ---
# On Windows with the Vulkan SDK installed, VULKAN_SDK is typically set as an
# environment variable. Fall back to the standard Strawberry/V-AIST install path.
if(DEFINED ENV{VULKAN_SDK})
    set(VULKAN_SDK_DIR "$ENV{VULKAN_SDK}")
else()
    set(VULKAN_SDK_DIR "C:/VulkanSDK/1.4.357.0")
endif()

message(STATUS "Vulkan SDK: ${VULKAN_SDK_DIR}")

set(Vulkan_INCLUDE_DIR "${VULKAN_SDK_DIR}/Include")
set(Vulkan_LIBRARY     "${VULKAN_SDK_DIR}/Lib/vulkan-1.lib")

# --- Strawberry / V-AIST toolchain detection ---
# V-AIST (Vulkan-AI SDK) may be checked out alongside TensorRT-LLM.
# Strawberry Perl is needed only for shader SPIR-V cross-compilation on Windows.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../V-AIST")
    set(V_AIST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../V-AIST")
    message(STATUS "V-AIST found at: ${V_AIST_ROOT}")
    add_definitions(-DV_AIST_AVAILABLE=1)
    include_directories(${V_AIST_ROOT}/include)
else()
    message(STATUS "V-AIST not found — proceeding with base Vulkan-only build")
endif()

# --- Source paths ---
set(TLLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# --- Include directories ---
# Use own includes instead of full TLLM headers that require CUDA.
# The vk_executor only depends on the Vulkan runtime + vulkanCommon.h.
include_directories(
    ${Vulkan_INCLUDE_DIR}
    ${TLLM_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- Collect Vulkan sources ---
# vk_executor depends on VulkanRuntime + VulkanContext + VulkanMemoryAllocator,
# plus the kernel dispatch registry for GPU kernel execution.
file(GLOB_RECURSE VULKAN_RUNTIME_SRCS
    "${TLLM_ROOT}/tensorrt_llm/common/vulkanContext.cpp"
    "${TLLM_ROOT}/tensorrt_llm/common/vulkanRuntime.cpp"
    "${TLLM_ROOT}/tensorrt_llm/common/vulkanMemoryAllocator.cpp"
    "${TLLM_ROOT}/tensorrt_llm/common/vulkanFence.cpp"
    "${TLLM_ROOT}/tensorrt_llm/common/vulkanShaderCompiler.cpp"
    "${TLLM_ROOT}/tensorrt_llm/common/vulkanUtilizationTracker.cpp"
)

file(GLOB_RECURSE VULKAN_KERNEL_SRCS
    "${TLLM_ROOT}/tensorrt_llm/kernels/vulkanKernelRegistry.cpp"
)

file(GLOB_RECURSE VULKAN_BACKEND_SRCS
    "${TLLM_ROOT}/tensorrt_llm/common/vulkanBackend.cpp"
)

# --- vk_executor sources ---
set(VK_EXECUTOR_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vk_executor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sampling/sampler_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sampling/finish_reasons.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sampling/penalties.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sampling/top_p_decay.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sampling/token_ban.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sampling/sampler_strategy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sampling/sampler.cpp
)

# --- Definitions ---
# ENABLE_VK_EXT: avoids CUDA header dependencies (matches trace_test pattern)
# NOMINMAX: prevents Windows min/max macros from breaking std::min/max
add_definitions(-DENABLE_VK_EXT -DNOMINMAX)

# ============================================================================
# vk_executor static library
# ============================================================================

add_library(vk_executor STATIC
    ${VK_EXECUTOR_SRCS}
    ${VULKAN_RUNTIME_SRCS}
    ${VULKAN_KERNEL_SRCS}
    ${VULKAN_BACKEND_SRCS}
)

target_include_directories(vk_executor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/sampling
)

target_link_libraries(vk_executor PUBLIC ${Vulkan_LIBRARY})

target_compile_definitions(vk_executor PUBLIC
    ENABLE_VULKAN=1
    ENABLE_VK_EXT=1
)

# Compiler warnings
if(MSVC)
    target_compile_options(vk_executor PRIVATE
        /W4
        /wd4201   # unnamed struct/union
        /wd4459   # declaration hides global
        /wd4100   # unreferenced formal parameter
        /wd4127   # conditional expression is constant
        /wd4505   # unreferenced local function removed
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(vk_executor PRIVATE
        -Wall
        -Wno-unused-parameter
    )
endif()

set_target_properties(vk_executor PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ============================================================================
# vk_executor_test executable
# ============================================================================

add_executable(vk_executor_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vk_executor.cpp
    ${VK_EXECUTOR_SRCS}
    ${VULKAN_RUNTIME_SRCS}
    ${VULKAN_KERNEL_SRCS}
    ${VULKAN_BACKEND_SRCS}
)

target_include_directories(vk_executor_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/sampling)

target_link_libraries(vk_executor_test PUBLIC ${Vulkan_LIBRARY})

target_compile_definitions(vk_executor_test PUBLIC
    ENABLE_VULKAN=1
    ENABLE_VK_EXT=1
)

if(MSVC)
    target_compile_options(vk_executor_test PRIVATE
        /W4
        /wd4201
        /wd4459
        /wd4100
        /wd4127
        /wd4505
    )
endif()

set_target_properties(vk_executor_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)