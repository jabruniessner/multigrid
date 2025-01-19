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

#ifdef DEBUGMODE
  sycl::cpu_selector selector;
#else  // DEBUGMODE
  sycl::gpu_selector selector;
#endif // DEBUGMODE
  sycl::queue q(selector,
                sycl::property_list{sycl::property::queue::in_order{}});

  constexpr std::size_t length = 4;
  constexpr std::size_t problem_size = length * length;

  Domain<2, length, length> sol(Paddings::PERIODIC, q, 1);
  Domain<2, length, length> init_guess(Paddings::PERIODIC, q, 1);
  Domain<2, length, length> rhs(Paddings::PERIODIC, q, 1);
  Domain<2, length, length> defect_r(Paddings::PERIODIC, q, 1);
  Domain<2, length, length> defect_p(Paddings::PERIODIC, q, 1);

  DataType *matrix =
      sycl::malloc_device<DataType>(utils::Power<problem_size, 2>::value, q);

  q.memset(matrix, utils::Power<problem_size, 2>::value * sizeof(DataType), 0)
      .wait();

  constexpr std::array<DataType, 5> values = {4, -1, -1, -1, -1};
  constexpr std::array<OffsetType, 5> offsets = {
      {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

  create_matrix_from_stencil<2, problem_size>(
      matrix, values, offsets, q, std::index_sequence<length, length>{});

  // std::cout << "#========= Original Matrix =======#" << std::endl;
  // print_matrix<problem_size>(matrix, q);

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "#========== The LU factorization reads=======#" << std::endl;
  q.submit([=](sycl::handler &h) {
     h.single_task([=]() { Factorize_ILU<problem_size>(matrix); });
   }).wait();

  // print_matrix<problem_size>(matrix, q);

  const int m = 16;

  for (int i = 0; i < length + 2; i++) {
    sol.set_value(-(DataType)i / (length + 1.), i, 0);
    sol.set_value(-(DataType)i / (length + 1.), i, length + 1);
    sol.set_value(-1., length + 1, i);
  }

  // std::cout << "#============= The initial set up is ==============#"
  //           << std::endl;

  // std::cout << "The inhomogenous part is given by: " << std::endl;
  sol.print_domain();

  Convolve(rhs, sol, values, offsets);
  std::cout << "The right hand side is given by: " << std::endl;
  rhs.print_domain();
  //
  //   std::cout << "The defect_r is prior to iteration given by" <<
  // std::endl;
  //   defect_r.print_domain();
  //
  //   std::cout << "The defect_p is prior to iteration given by" <<
  // std::endl;
  //   defect_p.print_domain();
  //
  //   std::cout << "The init_guess is given by " << std::endl;
  //   init_guess.print_domain();
  //
  Subdomain sub_init{init_guess, Range<1, 5>{}, Range<1, 5>{}};

  Subdomain sub_rhs{rhs, Range<1, 5>{}, Range<1, 5>{}};

  DataType *a = sycl::malloc_device<DataType>(1, q);

  // q.submit([=](sycl::handler &h) {
  //    h.single_task([=]() { sub_rhs(0, 0) = 1.; });
  //  }).wait();

  // auto val = sub_rhs(0, 0);

  // auto &val_1 = sub_rhs[-1];

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 5000; i++) {
    q.submit([=](sycl::handler &h) {
      h.single_task([=]() { solve_ILU<problem_size>(matrix, sub_rhs); });
    });
  }

  // q.wait();

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;

  std::cout << "The required time for the solution was: " << duration.count()
            << std::endl;
  //  solve_ILU<problem_size>(matrix, sub_rhs);

  // std::cout << "The solution is: " << std::endl;
  // rhs.print_domain();

  // RangeProps<Range<1, 2>> a{};

  // auto start = std::chrono::high_resolution_clock::now();
  // CG_solver(init_guess, rhs, defect_r, defect_p, values,
  //  offsets, num_iters);
  // auto end = std::chrono::high_resolution_clock::now();

  //  DataType const residual =
  //      cycles::compute_residual(rhs, init_guess, defect_p, values,
  //      offsets);
  //
  //  std::cout << "The residual after the iterations is: " << residual
  //            << std::endl;
  //
  //  std::chrono::duration<double> duration = end - start;
  //
  //  std::cout << "The required time was: " << duration.count() << "
  //  seconds"
  //            << std::endl;
  //
  //  // std::cout << "The init_guess before subtraction is: " <<
  //  std::endl;
  //  //  init_guess.print_domain();
  //
  //  subtract_domains(init_guess, init_guess, sol);

  // std::cout << "The output Matrix is given by: " << std::endl;
  // init_guess.print_domain();
  //
  //	std::cout<<"The defect is given by: "<<std::endl;
  //
  //
  //	Convolve(defect_r, init_guess, values, offsets);
  //
  //	std::cout<<"The convolution of the output is given by: "<<
  // std::endl; 	defect_r.print_domain();
  // subtract_domains(init_guess, sol, rhs);
  //
  //	init_guess.print_domain();
}
