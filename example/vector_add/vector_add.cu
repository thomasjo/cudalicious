#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <cudalicious/core.hpp>

template<typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
  std::vector<T> result(a.size());
  std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<T>());

  return result;
}

template<typename T>
__global__ void add_kernel(T* c, const T* a, const T* b, const size_t n)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

template<typename T>
std::vector<T> vector_add(const std::vector<T>& a, const std::vector<T>& b)
{
  // Prepare device memory.
  const auto n = a.size();
  const auto d_a = cuda::copy_to_device(a);
  const auto d_b = cuda::copy_to_device(b);
  auto d_c = cuda::allocate<T>(n);

  // Setup execution parameters.
  const dim3 threads_per_block(1024);
  const dim3 blocks_per_grid(std::ceil((n + 0.0) / threads_per_block.x));

  // Create device events that we'll use for timing.
  auto kernel_start = cuda::event_create();
  auto kernel_stop = cuda::event_create();

  // Record the start event.
  cuda::event_record(kernel_start);

  // Launch the addition kernel.
  std::cout << "Executing kernel... " << std::flush;
  add_kernel<<<blocks_per_grid, threads_per_block>>>(d_c, d_a, d_b, n);

  // Record, and wait for the stop event.
  cuda::event_record(kernel_stop);
  cuda::event_sync(kernel_stop);

  // Calculate elapsed kernel execution time.
  const auto msec_total = cuda::event_elapsed_time(kernel_start, kernel_stop);
  std::cout << "finished in " << msec_total << " ms\n";

  // Destroy timing events.
  cuda::event_destroy(kernel_start);
  cuda::event_destroy(kernel_stop);

  // Copy result from device to host.
  std::vector<T> c(n);
  cuda::copy_to_host(c, d_c);

  // Release all device memory.
  cuda::free(d_a);
  cuda::free(d_b);
  cuda::free(d_c);

#ifndef NDEBUG
  cuda::device_reset();
#endif

  return c;
}

int main()
{
  // Create some synthetic test data...
  constexpr auto n = 4000000UL;
  constexpr auto init_value = 1.0F;
  std::vector<float> a(n, init_value);
  std::vector<float> b(n);
  std::iota(b.begin(), b.end(), init_value);

  // Perform vector addition on GPU.
  const auto gpu_result = vector_add(a, b);
  // Perform vector addition on CPU.
  const auto cpu_result = a + b;

  // Verify that the CPU and GPU results are equal (to within desired epsilon).
  const auto equalish = [](float a, float b) { return std::abs(a - b) < 1e-16F; };
  const auto mismatched = std::mismatch(gpu_result.begin(), gpu_result.end(), cpu_result.begin(), equalish);
  if (mismatched.first != gpu_result.end()) {
    std::cerr << "CPU and GPU results do not correspond!\n";
    return EXIT_FAILURE;
  }
}
