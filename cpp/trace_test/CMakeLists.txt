cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(vulkan_trace_test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Vulkan
set(Vulkan_LIBRARY "C:/VulkanSDK/1.4.357.0/Lib/vulkan-1.lib")
set(Vulkan_INCLUDE_DIR "C:/VulkanSDK/1.4.357.0/Include")

message(STATUS "Vulkan include: ${Vulkan_INCLUDE_DIR}")
message(STATUS "Vulkan library: ${Vulkan_LIBRARY}")

# Source paths
set(TLLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Include directories - use own includes instead of TLLM headers that require CUDA
include_directories(
    ${Vulkan_INCLUDE_DIR}
    ${TLLM_ROOT}/include
)

# Collect all Vulkan source files
file(GLOB_RECURSE VULKAN_SRCS
    "${TLLM_ROOT}/tensorrt_llm/common/vulkan*.cpp"
    "${TLLM_ROOT}/tensorrt_llm/kernels/vulkanKernelRegistry.cpp"
)

# Add definitions to avoid CUDA header dependencies
add_definitions(-DENABLE_VK_EXT -DNOMINMAX)

# Main executable
add_executable(vulkan_trace_test
    ${TLLM_ROOT}/tests/vulkanTraceMain.cpp
    ${VULKAN_SRCS}
)

target_link_libraries(vulkan_trace_test PUBLIC ${Vulkan_LIBRARY})

# Compiler warnings
if(MSVC)
    target_compile_options(vulkan_trace_test PRIVATE /W4 /wd4201 /wd4459 /wd4100 /wd4127 /wd4505)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(vulkan_trace_test PRIVATE -Wall -Wno-unused-parameter)
endif()

# Enable Vulkan by default
target_compile_definitions(vulkan_trace_test PUBLIC ENABLE_VULKAN=1)

# Shader files need to be accessible at runtime
set_target_properties(vulkan_trace_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Copy real-valued DeepSeek MLA test-data assets (TEST 12/13) next to the test
# executable so the real-weights path resolves from the exe directory too, not only
# when run from cpp/ or via the source-anchored fallback in loadRealMlaData.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../tests/mla_real.bin)
    add_custom_command(TARGET vulkan_trace_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/../tests/mla_real.bin
            ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/mla_real.bin)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../tests/mla_real_prefill.bin)
    add_custom_command(TARGET vulkan_trace_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/../tests/mla_real_prefill.bin
            ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/mla_real_prefill.bin)
endif()

