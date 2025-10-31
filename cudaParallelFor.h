#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#else
#include "hipSYCL/pcuda/cuda.h"
#include "hipSYCL/pcuda/cuda_runtime.h"
#endif

#ifndef CUDA_PARALLEL_FOR_H
#define CUDA_PARALLEL_FOR_H

using int_return_double = decltype([](int) { return 0.0; });

template <class F> __global__ void kernel_launcher(F f) { f(); }

template <class F> cudaError_t cudaParallelFor(dim3 grid, dim3 block, F f) {
#ifdef __ACPP__
  return pcudaParallelFor(grid, block, f);
#else
  kernel_launcher<<<grid, block>>>(f);
  return cudaGetLastError();
#endif
}

#endif // CUDA_PARALLEL_FOR_H
