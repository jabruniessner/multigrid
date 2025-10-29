#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#else
#include "hipSYCL/pcuda/cuda.h"
#include "hipSYCL/pcuda/cuda_runtime.h"
#endif

#ifndef CUDA_PARALLEL_FOR_H
#define CUDA_PARALLEL_FOR_H

template <class F> void cudaParallelFor(dim3 grid, dim3 block, F f) {
#ifdef __ACPP__
  pcudaParallelFor(grid, block, f);
#else
  f<<<grid, block>>>();
#endif
}

#endif // CUDA_PARALLEL_FOR_H
