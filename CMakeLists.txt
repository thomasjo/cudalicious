cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(Cudalicious CXX)
option(BUILD_EXAMPLES "Build the bundled examples" OFF)

if (BUILD_EXAMPLES)
  find_package(CUDA REQUIRED)

  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wpedantic")

  find_package(CUDA REQUIRED)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 -std=c++11 -Xcompiler -Wno-unused,-Wno-unused-parameter,-Wno-language-extension-token")

  # Temporary "hack"...
  include_directories(include)

  cuda_add_executable(vector_add example/vector_add/vector_add.cu)
endif()
