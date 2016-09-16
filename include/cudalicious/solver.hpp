// Copyright (c) 2016 Thomas Johansen
// The use of this source code is governed by the MIT license, found in LICENSE.md.

#pragma once

#include <string>

#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace cuda {
namespace solver {

using handle_t = cusolverDnHandle_t;
using status_t = cusolverStatus_t;

std::string get_status_message(status_t status)
{
  switch (status) {
    case CUSOLVER_STATUS_SUCCESS:
      return "The operation completed successfully";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return "The library was not initialized.";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return "Resource allocation failed.";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return "An unsupported value or parameter was passed.";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return "The device does not support the invoked functionality.";
    case CUSOLVER_STATUS_MAPPING_ERROR:
      return "Mapping error.";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return "The GPU program failed to execute.";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return "An internal operation failed.";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "Matrix type is not supported by the invoked functionality.";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return "Operation not supported.";
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return "Zero pivot encountered.";
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return "Invalid license.";
  }

  return "An unknown error occured.";
}

void check_error(const status_t status)
{
  if (status == CUSOLVER_STATUS_SUCCESS) return;
  std::cerr << "cuSOLVER error: " << get_status_message(status) << "\n";
}

handle_t initialize()
{
  handle_t handle = nullptr;
  check_error(cusolverDnCreate(&handle));

  return handle;
}

void release(handle_t handle)
{
  if (!handle) return;
  check_error(cusolverDnDestroy(handle));
}

} // solver
} // cuda
