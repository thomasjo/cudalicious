#include <iostream>
#include <random>
#include <vector>

#include <cudalicious/core.hpp>

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
  const auto n = a.size();
  const auto d_a = cuda::copy_to_device(a);
  const auto d_b = cuda::copy_to_device(b);
  auto d_c = cuda::allocate<T>(n);

  auto kernel_start = cuda::event_create();
  auto kernel_stop = cuda::event_create();

  const auto blocks = dim3(400);
  const auto threads = dim3(1024, 1);

  std::cout << "Launching kernel... " << std::flush;

  cuda::event_record(kernel_start);
  add_kernel<<<blocks, threads>>>(d_c, d_a, d_b, n);
  cuda::event_record(kernel_stop);
  cuda::event_sync(kernel_stop);

  auto msec_total = cuda::event_elapsed_time(kernel_start, kernel_stop);
  std::cout << "finished in " << msec_total * 1000 << "ns\n";

  std::vector<T> c(n);
  cuda::copy_to_host(c, d_c);

  cuda::event_destroy(kernel_start);
  cuda::event_destroy(kernel_stop);

  cuda::free(d_a);
  cuda::free(d_b);
  cuda::free(d_c);

  cuda::device_reset();

  return c;
}

int main()
{
  constexpr auto n = 400000ul;
  constexpr auto init_value = 1.0f;

  std::vector<float> a(n, init_value);
  std::vector<float> b(n);
  std::iota(b.begin(), b.end(), init_value);

  const auto result = vector_add(a, b);
  std::cout << result[n - 1] << "\n";
}
