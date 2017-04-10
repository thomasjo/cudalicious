// Copyright (c) 2016 Thomas Johansen
// The use of this source code is governed by the MIT license, found in LICENSE.md.

#pragma once

#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace cuda {
namespace blas {

using handle_t = cublasHandle_t;
using status_t = cublasStatus_t;

std::string get_status_message(status_t status)
{
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "The operation completed successfully";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "The library was not initialized.";
    case CUBLAS_STATUS_ALLOC_FAILED: return "Resource allocation failed.";
    case CUBLAS_STATUS_INVALID_VALUE: return "An unsupported value or parameter was passed.";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "The device does not support the invoked functionality.";
    case CUBLAS_STATUS_MAPPING_ERROR: return "Mapping error.";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "The GPU program failed to execute.";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "An internal operation failed.";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "Operation not supported.";
    case CUBLAS_STATUS_LICENSE_ERROR: return "Invalid license.";
  }

  return "An unknown error occured.";
}

void check_error(const status_t status)
{
  if (status == CUBLAS_STATUS_SUCCESS) return;
  std::cerr << "cuBLAS error: " << get_status_message(status) << "\n";
}

handle_t initialize()
{
  handle_t handle = nullptr;
  check_error(cublasCreate(&handle));

  return handle;
}

void release(handle_t handle)
{
  if (!handle) return;
  check_error(cublasDestroy(handle));
}

void gemm(handle_t handle,
          int n,
          int m,
          int k,
          float alpha,
          const float* a,
          int lda,
          const float* b,
          int ldb,
          float beta,
          float* c,
          int ldc,
          bool transpose_a = false,
          bool transpose_b = false)
{
  const auto operation_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  const auto operation_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  check_error(cublasSgemm(handle, operation_a, operation_b, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

}  // namespace blas
}  // namespace cuda
