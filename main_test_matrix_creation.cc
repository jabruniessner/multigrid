#include "FullILU.h"
#include <iostream>
#include <utility>

constexpr std::size_t Dim = 2;
using OffsetType = std::array<int, Dim>;
// using DataType = double;

int main() {

  std::cout << "Hello World! " << std::endl;
  std::array<DataType, 5> values{-1., -1., 4, -1., -1.};
  std::array<OffsetType, 5> offsets{{{-1, 0}, {0, -1}, {0, 0}, {0, 1}, {1, 0}}};

  constexpr std::size_t problem_size = 4;

  DataType *matrix = new DataType[problem_size * problem_size]{};

  // std::cout << matrix.values << std::endl;

  create_matrix_from_stencil<Dim, problem_size>(matrix, values, offsets,
                                                std::index_sequence<2, 2>{});

  std::cout << "Before factorization: " << std::endl;
  print_matrix<problem_size>(matrix);

  // std::cout << "After factprization: " << std::endl;
  // Factorize_ILU<problem_size>(matrix);
  // print_matrix<problem_size>(matrix);
}
