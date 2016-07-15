#include <iostream>
#include <random>

#include "vector_add.cuh"

int main()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> unif(0, 1);

  auto n = 400000;
  std::vector<float> a(n);
  for (auto i = 0; i < n; ++i) {
    a[i] = unif(gen);
  }

  std::vector<float> b(n);
  std::iota(b.begin(), b.end(), 1);

  auto result = vector_add(a, b);
  std::printf("%.4f\n", result[n - 1]);

  return 0;
}
