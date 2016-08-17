cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(Cudalicious CXX)
set(CMAKE_CXX_STANDARD 14)

# Configure compiler warning level.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Werror \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Wcast-align \
  -Wcast-qual \
  -Wctor-dtor-privacy \
  -Wdisabled-optimization \
  -Wformat=2 \
  -Winit-self \
  -Wlogical-op \
  -Wmissing-declarations \
  -Wmissing-include-dirs \
  -Wnoexcept \
  -Wold-style-cast \
  -Woverloaded-virtual \
  -Wredundant-decls \
  -Wpointer-arith \
  -Wshadow \
  -Wsign-conversion \
  -Wsign-promo \
  -Wstrict-null-sentinel \
  -Wstrict-overflow=5 \
  -Wswitch-default \
  -Wundef \
  -Wunused \
  -Wno-unknown-warning-option \
")

# Configure linker flags.
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,O1")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--discard-all")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fstack-protector -Wl,-z,relro -Wl,-z,now -Wformat-security")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fvisibility=hidden")

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
