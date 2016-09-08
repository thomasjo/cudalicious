cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(Cudalicious CXX)
option(BUILD_EXAMPLES "Build the bundled examples" OFF)

if (BUILD_EXAMPLES)
  find_package(CUDA REQUIRED)

  set(CMAKE_CXX_STANDARD 14)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 -std=c++11")

  # Temporary "hack".
  include_directories(include)

  cuda_add_executable(vector_add
    example/vector_add/main.cpp
    example/vector_add/vector_add.cu
  )
endif()
