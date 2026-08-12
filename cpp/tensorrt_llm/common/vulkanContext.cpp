/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/vulkanContext.h"

#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#else
#include <dlfcn.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <set>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

namespace
{

char const* const VULKAN_VALIDATION_LAYERS[] = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef _WIN32
HMODULE gVulkanLibrary = nullptr;
#else
void* gVulkanLibrary = nullptr;
#endif

bool loadVulkanLoader()
{
#ifdef _WIN32
    gVulkanLibrary = LoadLibraryA("vulkan-1.dll");
    if (!gVulkanLibrary)
    {
        gVulkanLibrary = LoadLibraryA("vulkan.dll");
    }
#else
    gVulkanLibrary = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!gVulkanLibrary)
    {
        gVulkanLibrary = dlopen("libvulkan.dylib", RTLD_NOW | RTLD_LOCAL);
    }
#endif
    return gVulkanLibrary != nullptr;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
     VkDebugUtilsMessageTypeFlagsEXT messageType,
     VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
     void* pUserData)
{
    // Filter based on severity
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        TLLM_LOG_ERROR(pCallbackData->pMessage);
    }
    else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        TLLM_LOG_WARNING(pCallbackData->pMessage);
    }
    return VK_FALSE;
}

VkResult createDebugUtilsMessenger(
    VkInstance instance,
    VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        return func(instance, &createInfo, nullptr, pDebugMessenger);
    }
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void destroyDebugUtilsMessenger(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        func(instance, debugMessenger, nullptr);
    }
}

} // namespace

std::shared_ptr<VulkanContext> VulkanContext::create(uint32_t gpuID)
{
    auto context = std::shared_ptr<VulkanContext>(new VulkanContext(gpuID));

    if (context->initialize() != VulkanResult::SUCCESS)
    {
        TLLM_LOG_ERROR("Failed to initialize Vulkan context for GPU %u", gpuID);
        return nullptr;
    }

    return context;
}

VulkanContext::VulkanContext(uint32_t gpuID)
    : mGpuID(gpuID)
{
}

VulkanContext::~VulkanContext()
{
    if (mDevice != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(mDevice);
        vkDestroyDevice(mDevice, nullptr);
    }

    if (mDebugMessenger != VK_NULL_HANDLE)
    {
        destroyDebugUtilsMessenger(mInstance, mDebugMessenger);
    }

    if (mInstance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(mInstance, nullptr);
    }

#ifdef _WIN32
    if (gVulkanLibrary)
    {
        FreeLibrary(gVulkanLibrary);
        gVulkanLibrary = nullptr;
    }
#else
    if (gVulkanLibrary)
    {
        dlclose(gVulkanLibrary);
        gVulkanLibrary = nullptr;
    }
#endif
}

VulkanResult VulkanContext::initialize()
{
    // Load the Vulkan loader library
    if (!loadVulkanLoader())
    {
        mLastResult = VulkanResult::INITIALIZATION_FAILED;
        TLLM_LOG_ERROR("Failed to load Vulkan loader library");
        return mLastResult;
    }

    // Create Vulkan instance
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "TensorRT-LLM";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "TensorRT-LLM-Vulkan";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_4;

    // Required extensions
    mRequiredExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    // Enable validation layers in debug builds
    std::vector<char const*> enabledLayers;
#ifdef _DEBUG
    enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(mRequiredExtensions.size());
    createInfo.ppEnabledExtensionNames = mRequiredExtensions.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
    createInfo.ppEnabledLayerNames = enabledLayers.size() > 0 ? enabledLayers.data() : nullptr;

    VkResult result = vkCreateInstance(&createInfo, nullptr, &mInstance);
    if (result != VK_SUCCESS)
    {
        mLastResult = VulkanResult::INITIALIZATION_FAILED;
        TLLM_LOG_ERROR("Failed to create Vulkan instance: %d", result);
        return mLastResult;
    }

    // Set up debug messenger
    createDebugUtilsMessenger(mInstance, &mDebugMessenger);

    // Enumerate physical devices
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(mInstance, &deviceCount, nullptr);
    if (deviceCount == 0)
    {
        mLastResult = VulkanResult::INITIALIZATION_FAILED;
        TLLM_LOG_ERROR("No Vulkan-compatible devices found");
        return mLastResult;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(mInstance, &deviceCount, devices.data());

    // Find the best compute-capable device
    // Prefer AMD devices, then any device with compute queues
    bool foundDevice = false;
    uint32_t selectedDeviceIndex = 0;

    for (uint32_t i = 0; i < deviceCount; ++i)
    {
        // Get queue family properties
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queueFamilyCount, queueFamilies.data());

        for (uint32_t q = 0; q < queueFamilyCount; ++q)
        {
            if (queueFamilies[q].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                // Check if this is the requested GPU or an AMD device
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(devices[i], &props);

                if (i == mGpuID || props.vendorID == 0x1002 /* AMD */)
                {
                    selectedDeviceIndex = i;
                    foundDevice = true;
                    break;
                }
            }
        }

        if (foundDevice)
        {
            break;
        }
    }

    if (!foundDevice)
    {
        mLastResult = VulkanResult::INITIALIZATION_FAILED;
        TLLM_LOG_ERROR("No suitable compute-capable device found (GPU count: %u)", deviceCount);
        return mLastResult;
    }

    mPhysicalDevice = devices[selectedDeviceIndex];

    // Find compute queue family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &queueFamilyCount, queueFamilies.data());

    bool foundComputeQueue = false;
    for (uint32_t i = 0; i < queueFamilyCount; ++i)
    {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
        {
            mComputeQueueFamilyIndex = i;
            mDeviceInfo.computeQueueCount = queueFamilies[i].queueCount;
            foundComputeQueue = true;
            break;
        }
    }

    if (!foundComputeQueue)
    {
        mLastResult = VulkanResult::INITIALIZATION_FAILED;
        TLLM_LOG_ERROR("No compute queue family found");
        return mLastResult;
    }

    // Enable required device extensions (only those the physical device supports).
    // Cooperative matrix, 16-bit storage, descriptor indexing and timeline semaphore
    // support vary by vendor, so each extension is queried and enabled conditionally
    // to avoid VK_ERROR_EXTENSION_NOT_PRESENT on devices that lack it (e.g. an
    // NVIDIA-only extension requested on an AMD device).
    std::vector<char const*> deviceExtensions;

    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(mPhysicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availableExts(extCount);
    if (extCount > 0)
    {
        vkEnumerateDeviceExtensionProperties(mPhysicalDevice, nullptr, &extCount, availableExts.data());
    }
    std::set<std::string> supportedExts;
    for (uint32_t i = 0; i < extCount; ++i)
    {
        supportedExts.insert(availableExts[i].extensionName);
    }

    // Cooperative matrix support (RDNA3+/CDNA2+). Enable whichever flavor the
    // device advertises; both are never guaranteed present together.
    if (supportedExts.count(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
    {
        deviceExtensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }
    if (supportedExts.count(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME))
    {
        deviceExtensions.push_back(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }

    // 16-bit storage (for FP16 support)
    if (supportedExts.count(VK_KHR_16BIT_STORAGE_EXTENSION_NAME))
    {
        deviceExtensions.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
    }

    // Descriptor indexing
    if (supportedExts.count(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME))
    {
        deviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    }

    // Timeline semaphore support
    if (supportedExts.count(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME))
    {
        deviceExtensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    }

    // Create device queues
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::vector<float> queuePriorities;
    uint32_t numQueues = std::min(mDeviceInfo.computeQueueCount, 4u); // Limit to 4 queues

    for (uint32_t i = 0; i < numQueues; ++i)
    {
        queuePriorities.push_back(1.0f);
    }

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = mComputeQueueFamilyIndex;
    queueCreateInfo.queueCount = numQueues;
    queueCreateInfo.pQueuePriorities = queuePriorities.data();

    // Get device properties
    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(mPhysicalDevice, &deviceProperties);

    mDeviceInfo.deviceName = deviceProperties.deviceName;
    mDeviceInfo.vendorID = deviceProperties.vendorID;
    mDeviceInfo.deviceID = deviceProperties.deviceID;
    mDeviceInfo.apiVersion = deviceProperties.apiVersion;
    mDeviceInfo.driverVersion = deviceProperties.driverVersion;
    mDeviceInfo.deviceType = deviceProperties.deviceType;
    mDeviceInfo.limits.maxComputeWorkGroupInvocations = deviceProperties.limits.maxComputeWorkGroupInvocations;
    memcpy(mDeviceInfo.limits.maxComputeWorkGroupCount, deviceProperties.limits.maxComputeWorkGroupCount, sizeof(uint32_t) * 3);
    memcpy(mDeviceInfo.limits.maxComputeWorkGroupSize, deviceProperties.limits.maxComputeWorkGroupSize, sizeof(uint32_t) * 3);
    mDeviceInfo.limits.maxPushConstantsSize = deviceProperties.limits.maxPushConstantsSize;
    mDeviceInfo.limits.minStorageBufferOffsetAlignment = deviceProperties.limits.minStorageBufferOffsetAlignment;
    mDeviceInfo.limits.minUniformBufferOffsetAlignment = deviceProperties.limits.minUniformBufferOffsetAlignment;
    mDeviceInfo.limits.maxMemoryAllocationCount = deviceProperties.limits.maxMemoryAllocationCount;

    // Enable features via properties2 chain
    VkPhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceVariablePointersFeatures variablePointersFeatures{};
    variablePointersFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES;

    VkPhysicalDevice16BitStorageFeatures storage16BitFeatures{};
    storage16BitFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;

    VkPhysicalDeviceShaderFloat16Int8Features float16Features{};
    float16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatrixFeatures{};
    coopMatrixFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;

    // Chain the features (queried below via GetPhysicalDeviceFeatures2, then
    // passed back to CreateDevice.pNext to actually enable the supported ones).
    deviceFeatures2.pNext = &variablePointersFeatures;
    variablePointersFeatures.pNext = &storage16BitFeatures;
    storage16BitFeatures.pNext = &float16Features;
    float16Features.pNext = &coopMatrixFeatures;

    vkGetPhysicalDeviceFeatures2(mPhysicalDevice, &deviceFeatures2);

    mDeviceInfo.hasCooperativeMatrix = coopMatrixFeatures.cooperativeMatrix;
    mDeviceInfo.hasFP16 = float16Features.shaderFloat16;
    mDeviceInfo.hasBF16 = false; // Check for BF16 support extension if needed

    // Get subgroup size
    VkPhysicalDeviceVulkan11Properties deviceProperties11{};
    deviceProperties11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;

    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &deviceProperties11;

    vkGetPhysicalDeviceProperties2(mPhysicalDevice, &properties2);
    mDeviceInfo.subgroupSize = deviceProperties11.subgroupSize;

    // Create logical device
    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    // Enable the device features that were queried above (only those the device
    // actually supports are set to VK_TRUE, so this is safe). Required for
    // float16/cooperative-matrix shaders such as elementwise_add.
    deviceCreateInfo.pNext = &deviceFeatures2;

    VkPhysicalDeviceFeatures enabledFeatures{};
    deviceCreateInfo.pEnabledFeatures = &enabledFeatures;

    result = vkCreateDevice(mPhysicalDevice, &deviceCreateInfo, nullptr, &mDevice);
    if (result != VK_SUCCESS)
    {
        mLastResult = VulkanResult::INITIALIZATION_FAILED;
        TLLM_LOG_ERROR("Failed to create Vulkan device: %d", result);
        return mLastResult;
    }

    // Get compute queues
    mComputeQueues.resize(numQueues);
    for (uint32_t i = 0; i < numQueues; ++i)
    {
        vkGetDeviceQueue(mDevice, mComputeQueueFamilyIndex, i, &mComputeQueues[i]);
    }

    mLastResult = VulkanResult::SUCCESS;
    TLLM_LOG_INFO("Vulkan context initialized for device: %s (GPU %u)",
        mDeviceInfo.deviceName.c_str(), selectedDeviceIndex);

    return mLastResult;
}

char const* VulkanContext::getErrorString(VulkanResult result)
{
    switch (result)
    {
        case VulkanResult::SUCCESS:
            return "VK_SUCCESS";
        case VulkanResult::INVALID_VALUE:
            return "VK_ERROR_INVALID_VALUE";
        case VulkanResult::OUT_OF_MEMORY:
            return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VulkanResult::INVALID_HANDLE:
            return "VK_ERROR_INVALID_HANDLE";
        case VulkanResult::INITIALIZATION_FAILED:
            return "VK_ERROR_INITIALIZATION_FAILED";
        case VulkanResult::FEATURE_NOT_PRESENT:
            return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VulkanResult::DEVICE_LOST:
            return "VK_ERROR_DEVICE_LOST";
        case VulkanResult::UNKNOWN_ERROR:
            return "VK_ERROR_UNKNOWN";
        default:
            return "VK_UNKNOWN_ERROR";
    }
}

} // namespace common
TRTLLM_NAMESPACE_END
