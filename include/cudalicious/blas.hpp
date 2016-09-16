// Copyright (c) 2016 Thomas Johansen
// The use of this source code is governed by the MIT license, found in LICENSE.md.

#pragma once

#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace cuda {
namespace blas {

std::string get_status_message(cublasStatus_t status)
{
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "The operation completed successfully";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "The library was not initialized.";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "Resource allocation failed.";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "An unsupported value or parameter was passed.";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "The device does not support the invoked functionality.";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "Mapping error.";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "The GPU program failed to execute.";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "An internal operation failed.";
    // case CUBLAS_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    //   return "Matrix type is not supported by the invoked functionality.";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "Operation not supported.";
    // case CUBLAS_STATUS_ZERO_PIVOT:
    //   return "Zero pivot encountered.";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "Invalid license.";
  }

  return "An unknown error occured.";
}

void check_error(const cublasStatus_t status)
{
  if (status == CUBLAS_STATUS_SUCCESS) return;
  std::cerr << "cuBLAS error: " << get_status_message(status) << "\n";
}

cublasHandle_t initialize()
{
  cublasHandle_t handle = nullptr;
  check_error(cublasCreate(&handle));

  return handle;
}

void release(cublasHandle_t handle)
{
  if (!handle) return;
  check_error(cublasDestroy(handle));
}

} // blas
} // cuda
