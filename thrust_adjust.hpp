#include "concepts.h"
#include "hipSYCL/pcuda/pcuda.hpp"
#include "hipSYCL/pcuda/pcuda_runtime.hpp"
#include <execution>
#include <iostream>

#ifndef THRUST_ADJUST_HPP
#define THRUST_ADJUST_HPP

namespace thrust {

constexpr std::size_t gbs = 128;

template <typename RandomAccIt, typename UnaryFunction,
          std::size_t BlockSize = gbs>
void for_each(RandomAccIt first, RandomAccIt last, UnaryFunction f) {

  const std::size_t length = std::distance(first, last);

  static_assert(0 < BlockSize && BlockSize <= 1024,
                "BlockSize must be in (0, 1024]");

  if (length == 0)
    return;

  int num_blocks = (length + BlockSize - 1) / BlockSize;

  pcudaParallelFor(num_blocks, BlockSize, [=]() {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    auto my_it = first;

    if (gid < length) {
      my_it += gid;
      f(*my_it);
    }
  });

  pcudaDeviceSynchronize();
}

template <Arithmetic T, typename T2, typename BinaryOperation>
T atomicBinaryAdd(T *address, T2 val, BinaryOperation op) {

  using IntType = std::conditional_t<
      sizeof(T) == sizeof(unsigned short), unsigned short,

      std::conditional_t<
          sizeof(T) == sizeof(unsigned int), unsigned int,

          std::conditional_t<sizeof(T) == sizeof(unsigned long long),
                             unsigned long long, void>>>;

  static_assert(std::is_same_v<IntType, void> == false,
                "Unsupported type size for atomicBinaryAdd \n Only allowed "
                "sizes are 2, 4 and 8 bytes");

  union FloatInt {
    T f;
    IntType i;
  };

  FloatInt oldVal, assumed, newVal;
  oldVal.f = *address;

  do {

    assumed.i = oldVal.i;
    newVal.f = op(assumed.f, val);
    oldVal.i = atomicCAS((IntType *)address, assumed.i, newVal.i);

  } while (assumed.i != oldVal.i);

  return oldVal.f;
}

template <typename RandomAccIt, typename BinaryFunction, typename UnaryFunction,
          typename T, std::size_t BlockSize = gbs>

T transform_reduce(RandomAccIt first, RandomAccIt last, T init,
                   BinaryFunction binary_op, UnaryFunction unary_op) {
  std::size_t length = std::distance(first, last);
  static_assert(0 < BlockSize && BlockSize <= 1024,
                "BlockSize must be in (0, 1024]");

  if (length == 0)
    return init;

  int num_blocks = (length + BlockSize - 1) / BlockSize;

  // std::cout << "Hello World from inside transform_reduce" << std::endl;

  T *results;
  pcudaMallocManaged(&results, (1 + num_blocks) * sizeof(T));

  results[0] = init;

  pcudaParallelFor(num_blocks, BlockSize, [=]() {
    __shared__ T shared_data[BlockSize];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    T temp{};

    if (gid < length) {
      auto my_it = first;
      my_it += gid;
      temp = unary_op(*my_it);
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
    num_blocks = (length + BlockSize - 1) / BlockSize;

    pcudaParallelFor(num_blocks, BlockSize, [=]() {
      __shared__ T shared_data[BlockSize];
      const int tid = threadIdx.x;
      const int gid = blockIdx.x * blockDim.x + threadIdx.x;

      shared_data[tid] = gid < length ? results[gid] : 0;
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

  } while (num_blocks > 1);

  pcudaDeviceSynchronize();

  auto return_result = results[0];
  pcudaFree(results);

  return return_result;
}

template <typename RandomAccIt, typename RandomAccIt2, typename BinaryFunction,
          typename BinaryFunction2, typename T, std::size_t BlockSize = gbs>

T transform_reduce(RandomAccIt first, RandomAccIt last, RandomAccIt2 first2,
                   T init, BinaryFunction binary_op,
                   BinaryFunction2 binary_op2) {
  std::size_t length = std::distance(first, last);
  static_assert(0 < BlockSize && BlockSize <= 1024,
                "BlockSize must be in (0, 1024]");

  if (length == 0)
    return init;

  T *results;

  int num_blocks = (length + BlockSize - 1) / BlockSize;

  pcudaMallocManaged(&results, (num_blocks + 1) * sizeof(T));

  results[0] = init;

  pcudaParallelFor(num_blocks, BlockSize, [=]() {
    __shared__ T shared_data[BlockSize];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    T temp{};

    if (gid < length) {
      auto my_it = first;
      my_it += gid;
      auto my_it2 = first2;
      my_it2 += gid;
      temp = binary_op(*my_it, *my_it2);
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
    num_blocks = (length + BlockSize - 1) / BlockSize;

    pcudaParallelFor(num_blocks, BlockSize, [=]() {
      __shared__ T shared_data[BlockSize];
      const int tid = threadIdx.x;
      const int gid = blockIdx.x * blockDim.x + threadIdx.x;

      shared_data[tid] = gid < length ? results[gid] : 0;
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

  } while (num_blocks > 1);

  pcudaDeviceSynchronize();

  auto return_result = results[0];
  pcudaFree(results);

  return return_result;
}

template <typename RandomAccIt, typename RandomAccIt2, typename UnaryFunction,
          std::size_t BlockSize = gbs>
UnaryFunction transform(RandomAccIt first, RandomAccIt last,
                        RandomAccIt2 first2, UnaryFunction unary_op) {

  const std::size_t length = std::distance(first, last);
  static_assert(0 < BlockSize && BlockSize <= 1024,
                "BlockSize must be in (0, 1024]");

  if (length == 0)
    return unary_op;

  int num_blocks = (length + BlockSize - 1) / BlockSize;

  pcudaParallelFor(num_blocks, BlockSize, [=]() {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < length) {
      auto my_it = first;
      my_it += gid;
      auto my_it2 = first2;
      my_it2 += gid;
      *my_it2 = unary_op(*my_it);
    }
  });

  return unary_op;
}

template <typename RandomAccIt, typename RandomAccIt2, typename RandomAccIt3,
          typename BinaryFunction, std::size_t BlockSize = gbs>
BinaryFunction transform(RandomAccIt first, RandomAccIt last,
                         RandomAccIt2 first2, RandomAccIt3 out,
                         BinaryFunction binary_op) {

  const std::size_t length = std::distance(first, last);
  static_assert(0 < BlockSize && BlockSize <= 1024,
                "BlockSize must be in (0, 1024]");

  if (length == 0)
    return binary_op;

  int num_blocks = (length + BlockSize - 1) / BlockSize;

  pcudaParallelFor(num_blocks, BlockSize, [=]() {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < length) {
      auto my_it = first;
      my_it += gid;
      auto my_it2 = first2;
      my_it2 += gid;
      auto my_it3 = out;
      my_it3 += gid;
      *my_it3 = binary_op(*my_it, *my_it2);
    }
  });

  return binary_op;
}

template <typename RandomAccIt, typename T, std::size_t BlockSize = gbs>
void fill(RandomAccIt first, RandomAccIt last, T value) {

  const std::size_t length = std::distance(first, last);
  static_assert(0 < BlockSize && BlockSize <= 1024,
                "BlockSize must be in (0, 1024]");

  if (length == 0)
    return;

  int num_blocks = (length + BlockSize - 1) / BlockSize;

  pcudaParallelFor(num_blocks, BlockSize, [=]() {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < length) {
      auto my_it = first;
      my_it += gid;
      *my_it = value;
    }
  });
}

} // namespace thrust

#endif
