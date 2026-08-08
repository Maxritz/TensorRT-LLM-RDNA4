# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 - you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

add_gtest(cudaProfilerUtilsTest cudaProfilerUtilsTest.cpp)
add_gtest(cudaUtilsTest cudaUtilsTest.cpp)
add_gtest(attentionWorkspaceTest attentionWorkspaceTest.cpp)
add_gtest(memoryUtilsTest memoryUtilsTest.cu)
add_gtest(optionalRefTest optionalRefTest.cpp)
add_gtest(quantizationTest quantizationTest.cpp)
add_gtest(stlUtilsTest stlUtilsTest.cpp)
add_gtest(stringUtilsTest stringUtilsTest.cpp)
add_gtest(timestampUtilsTest timestampUtilsTest.cpp)
add_gtest(tllmExceptionTest tllmExceptionTest.cpp)
add_gtest(cudaDriverWrapperTest cudaDriverWrapperTest.cpp)
add_gtest(loggerTest loggerTest.cpp)

# Vulkan compatibility tests (only when Vulkan is enabled)
if(ENABLE_VULKAN)
  add_gtest(vulkanCompatibilityTest vulkanCompatibilityTest.cpp)
  add_gtest(vulkanTraceValidationTest vulkanTraceValidationTest.cpp)
  target_link_libraries(vulkanCompatibilityTest PUBLIC ${Vulkan_LIBRARY})
  target_link_libraries(vulkanTraceValidationTest PUBLIC ${Vulkan_LIBRARY})
endif()
