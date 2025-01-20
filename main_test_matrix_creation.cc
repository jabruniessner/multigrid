#include "FullILU.h"
#include "hipSYCL/sycl/device_selector.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include <iostream>
#include <sycl/sycl.hpp>
#include <utility>

constexpr std::size_t Dim = 2;
using OffsetType = std::array<int, Dim>;
// using DataType = double;

int main() {

  sycl::gpu_selector selector;
  sycl::queue q;

  std::cout << "Hello World! " << std::endl;
  std::array<DataType, 5> values{-1., -1., 4, -1., -1.};
  std::array<OffsetType, 5> offsets{{{-1, 0}, {0, -1}, {0, 0}, {0, 1}, {1, 0}}};

  constexpr std::size_t problem_size = 4;

  DataType *matrix =
      sycl::malloc_device<DataType>(problem_size * problem_size, q);
  q.memset(matrix, problem_size * problem_size * sizeof(DataType), 0);

  // std::cout << matrix.values << std::endl;

  create_matrix_from_stencil<Dim, problem_size>(matrix, values, offsets, q,
                                                std::index_sequence<2, 2>{});

  std::cout << "Before factorization: " << std::endl;
  print_matrix<problem_size>(matrix, q);

  // std::cout << "After factprization: " << std::endl;
  // Factorize_ILU<problem_size>(matrix);
  // print_matrix<problem_size>(matrix);
}
