#include "vector_add.cuh"

#include <iostream>
#include <vector>

#include <cudalicious/core.hpp>

__global__
void add_kernel(float* c, const float* a, const float* b, const size_t n)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

std::vector<float> vector_add(std::vector<float> a, std::vector<float> b)
{
  auto n = a.size();
  auto d_a = cuda::copy_to_device(a);
  auto d_b = cuda::copy_to_device(b);
  auto d_c = cuda::allocate<float>(n);

  auto blocks = dim3(400);
  auto threads = dim3(1024, 1);

  std::cout << "Launching kernel..." << std::endl;
  add_kernel<<<blocks, threads>>>(d_c, d_a, d_b, n);
  cuda::peek_at_last_error();
  cuda::device_sync();

  std::vector<float> c(n);
  cuda::copy_to_host(c, d_c);

  cuda::free(d_a);
  cuda::free(d_b);
  cuda::free(d_c);

  cuda::device_reset();

  return c;
}
