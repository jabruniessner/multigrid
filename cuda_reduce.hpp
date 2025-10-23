#include "concepts.h"
#include "hipSYCL/pcuda/pcuda.hpp"
#include "hipSYCL/pcuda/pcuda_runtime.hpp"
#include <iostream>

#ifndef CUDA_REDUCE
#define CUDA_REDUCE

namespace reduction_kernel {

constexpr std::size_t gbs = 256;
constexpr std::size_t seq_size = 8;

template <typename UnaryOp, typename BinaryOp, typename T,
          std::size_t BlockSize = gbs, std::size_t seq_size = 8>
void pcudaParallelTransformReduce(std::size_t length, T *reduction, T *results,
                                  BinaryOp binary_op, UnaryOp unary_op) {

  int num_threads = (length + seq_size - 1) / seq_size;

  int num_blocks = (num_threads + BlockSize - 1) / BlockSize;

  // pcudaMalloc(&results, (1 + num_blocks) * sizeof(T));

  // results[0] = *reduction;

  pcudaParallelFor(num_blocks, BlockSize, [=]() {
    __shared__ T shared_data[BlockSize];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    T temp{};

    for (int i = 0; i < seq_size; i++) {
      if (seq_size * gid + i < length)
        temp += unary_op(seq_size * gid + i);
    }

    shared_data[tid] = temp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        shared_data[tid] = binary_op(shared_data[tid], shared_data[tid + s]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      results[blockIdx.x] = shared_data[0];
    };
  });

  do {
    length = num_blocks;
    num_threads = (length + seq_size - 1) / seq_size;
    num_blocks = (num_threads + BlockSize - 1) / BlockSize;

    pcudaParallelFor(num_blocks, BlockSize, [=]() {
      __shared__ T shared_data[BlockSize];
      const int tid = threadIdx.x;
      const int gid = blockIdx.x * blockDim.x + threadIdx.x;

      shared_data[tid] = 0;

      for (int i = 0; i < seq_size; i++)
        if (seq_size * gid + i < length)
          shared_data[tid] =
              binary_op(shared_data[tid], results[seq_size * gid + i]);

      __syncthreads();

      for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
          shared_data[tid] = binary_op(shared_data[tid], shared_data[tid + s]);
        }
        __syncthreads();
      }

      if (tid == 0) {
        results[blockIdx.x] = shared_data[0];

        if (num_blocks == 1)
          reduction[0] = shared_data[0];
      };
    });

  } while (num_blocks > 1);
}

} // namespace reduction_kernel

#endif
