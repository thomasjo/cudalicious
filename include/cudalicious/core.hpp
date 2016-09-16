// Copyright (c) 2016 Thomas Johansen
// The use of this source code is governed by the MIT license, found in LICENSE.md.

#pragma once

#include <iostream>
#include <vector>

namespace cuda {

void check_error(const cudaError_t code)
{
  if (code == cudaSuccess) return;
  std::cerr << "CUDA error: " << cudaGetErrorString(code) << "\n";
}

template<typename T>
T* allocate(const size_t num)
{
  T* dev_ptr = nullptr;
  cuda::check_error(cudaMalloc(&dev_ptr, num * sizeof(T)));

  return dev_ptr;
}

template<typename T>
void free(T* dev_ptr)
{
  if (dev_ptr == nullptr) return;
  cuda::check_error(cudaFree(dev_ptr));
}

template<typename T>
void copy_to_device(T* dev_ptr, const T* host_ptr, const size_t num)
{
  cuda::check_error(cudaMemcpy(dev_ptr, host_ptr, num * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
T* copy_to_device(const T* host_ptr, const size_t num)
{
  T* dev_ptr = cuda::allocate<T>(num);
  copy_to_device(dev_ptr, host_ptr, num);

  return dev_ptr;
}

template<typename T>
T* copy_to_device(const std::vector<T>& host_vec)
{
  return copy_to_device(host_vec.data(), host_vec.size());
}

template<typename T>
void copy_to_host(T* host_ptr, const T* dev_ptr, const size_t num)
{
  cuda::check_error(cudaMemcpy(host_ptr, dev_ptr, num * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void copy_to_host(std::vector<T>& host_vec, const T* dev_ptr)
{
  copy_to_host(host_vec.data(), dev_ptr, host_vec.size());
}

template<typename T>
void copy_on_device(T* dst_ptr, const T* src_ptr, const size_t num)
{
  cuda::check_error(cudaMemcpy(dst_ptr, src_ptr, num * sizeof(T), cudaMemcpyDeviceToDevice));
}

template<typename T>
T* copy_on_device(const T* src_ptr, const size_t num)
{
  T* dst_ptr = cuda::allocate<T>(num);
  cuda::copy_on_device(dst_ptr, src_ptr, num);

  return dst_ptr;
}

void device_sync()
{
  cuda::check_error(cudaDeviceSynchronize());
}

void device_reset()
{
  cuda::check_error(cudaDeviceReset());
}

void peek_at_last_error()
{
  cuda::check_error(cudaPeekAtLastError());
}

}
