#include "CG_Solver.h"
#include "Convolution.h"
#include "FullILU.h"
#include "cycles.h"
// #include "hipSYCL/sycl/usm.hpp"
#include <chrono>
#include <iostream>
#include <string>
#include <utility>

template <std::size_t start, std::size_t end> struct Range {};

int main(int argc, char *argv[]) {

  // int num_iters = std::stoi(argv[1]);

  using namespace cg_solver;
  using namespace convolution;

  constexpr std::size_t length = 4;
  constexpr std::size_t problem_size = length * length;

  Domain<2, length, length> sol(Paddings::PERIODIC, 1);
  Domain<2, length, length> init_guess(Paddings::PERIODIC, 1);
  Domain<2, length, length> rhs(Paddings::PERIODIC, 1);
  Domain<2, length, length> defect_r(Paddings::PERIODIC, 1);
  Domain<2, length, length> defect_p(Paddings::PERIODIC, 1);

  DataType *matrix = new DataType[utils::Power<problem_size, 2>::value]{};

  constexpr std::array<DataType, 5> values = {4, -1, -1, -1, -1};
  constexpr std::array<OffsetType, 5> offsets = {
      {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

  create_matrix_from_stencil<2, problem_size>(
      matrix, values, offsets, std::index_sequence<length, length>{});

  // std::cout << "#========= Original Matrix =======#" << std::endl;
  // print_matrix<problem_size>(matrix, q);

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "#========== The LU factorization reads=======#" << std::endl;
  Factorize_ILU<problem_size>(matrix);

  // print_matrix<problem_size>(matrix, q);

  const int m = 16;

  for (int i = 0; i < length + 2; i++) {
    sol(i, 0) = -(DataType)i / (length + 1.);
    sol(i, length + 1) = -(DataType)i / (length + 1.);
    sol(length + 1, i) = -1.;
  }

  // std::cout << "#============= The initial set up is ==============#"
  //           << std::endl;

  // std::cout << "The inhomogenous part is given by: " << std::endl;
  sol.print_domain();

  Convolve(rhs, sol, values, offsets);
  std::cout << "The right hand side is given by: " << std::endl;
  rhs.print_domain();
}
