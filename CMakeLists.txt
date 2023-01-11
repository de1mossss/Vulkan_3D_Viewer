cmake_minimum_required(VERSION 3.24)
project(VulkanTest)

set(CMAKE_CXX_STANDARD 17)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(glfw)

find_package(Vulkan REQUIRED)

add_executable(VulkanTest main.cpp)

target_link_libraries(VulkanTest glfw)

target_link_libraries(VulkanTest Vulkan::Vulkan)

