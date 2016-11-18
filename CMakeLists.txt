cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(Cudalicious CXX)
option(BUILD_EXAMPLES "Build the bundled examples" OFF)

if (BUILD_EXAMPLES)
  find_package(CUDA REQUIRED)

  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wpedantic")

  # Provide temporary Xcode 7.3 side-by-side support via environment variable.
  if(EXISTS "$ENV{XCODE73_DIR}")
    set(MACOS_SDK_DIR "$ENV{XCODE73_DIR}/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk")
    include_directories(BEFORE SYSTEM ${MACOS_SDK_DIR}/usr/include)
  endif()

  find_package(CUDA REQUIRED)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 -std=c++11 -Xcompiler -Wno-unused,-Wno-language-extension-token")

  # Temporary "hack"...
  include_directories(include)

  cuda_add_executable(vector_add
    example/vector_add/main.cpp
    example/vector_add/vector_add.cu
  )
endif()
