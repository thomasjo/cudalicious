cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(Cudalicious CXX)
set(CMAKE_CXX_STANDARD 14)

# Resolve CUDA dependency.
find_package(CUDA REQUIRED)

# Set CUDA/nvcc compilation flags.
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 -std=c++11")

# Temporary "hack".
include_directories(include)

cuda_add_executable(vector_add
  example/vector_add/main.cpp
  example/vector_add/vector_add.cu
)
