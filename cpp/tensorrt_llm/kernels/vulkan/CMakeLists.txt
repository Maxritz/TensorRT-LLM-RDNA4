# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION &
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

if(ENABLE_VULKAN)
  file(GLOB VK_SRCS *.cpp)

  add_library(vulkan_kernels_src OBJECT ${VK_SRCS})

  target_link_libraries(vulkan_kernels_src PUBLIC common_src ${Vulkan_LIBRARY})
  target_include_directories(vulkan_kernels_src PUBLIC ${Vulkan_INCLUDE_DIR})

  if(WIN32)
    target_compile_options(vulkan_kernels_src PRIVATE /W4 /wd4201 /wd4459 /wd4127)
  else()
    target_compile_options(vulkan_kernels_src PRIVATE -Wall -Wno-unused-parameter)
  endif()
endif()
