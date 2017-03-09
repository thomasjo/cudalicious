cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Cudalicious CXX CUDA)
option(BUILD_EXAMPLES "Build the bundled examples" OFF)

if (BUILD_EXAMPLES)
  set(CMAKE_CXX_STANDARD 14)  # NOTE: Also affects CMAKE_CUDA_STANDARD.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

  # Since nvcc doesn't support -std=c++14 yet, manually set to c++11.
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50")

  add_executable(vector_add example/vector_add/vector_add.cu)
  target_include_directories(vector_add PRIVATE include)
endif()
