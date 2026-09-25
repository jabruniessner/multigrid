#include "CG_Solver.h"
#include "Convolution.h"
#include "cycles.h"
#include <chrono>
#include <iostream>
#include <string>

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

  constexpr std::size_t length = 44;

  Domain<2, length, length> sol(Paddings::PERIODIC, q, 1);
  Domain<2, length, length> init_guess(Paddings::PERIODIC, q, 1);
  Domain<2, length, length> rhs(Paddings::PERIODIC, q, 1);
  Domain<2, length, length> defect_r(Paddings::PERIODIC, q, 1);
  Domain<2, length, length> defect_p(Paddings::PERIODIC, q, 1);

  constexpr std::array<DataType, 5> values = {4, -1, -1, -1, -1};
  constexpr std::array<OffsetType, 5> offsets = {
      {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

  const int m = 16;

  for (int i = 0; i < length + 2; i++) {
    sol.set_value(-(DataType)i / (length + 1.), i, 0);
    sol.set_value(-(DataType)i / (length + 1.), i, length + 1);
    sol.set_value(-1., length + 1, i);
  }

  std::cout << "#============= The initial set up is ==============#"
            << std::endl;

  std::cout << "The inhomogenous part is given by: " << std::endl;
  //  sol.print_domain();

  Convolve(rhs, sol, values, offsets);
  //  std::cout << "The right hand side is given by: " << std::endl;
  //  rhs.print_domain();

  //  std::cout << "The defect_r is prior to iteration given by" << std::endl;
  //  defect_r.print_domain();
  //
  //  std::cout << "The defect_p is prior to iteration given by" << std::endl;
  //  defect_p.print_domain();
  //
  //  std::cout << "The init_guess is given by " << std::endl;
  //  init_guess.print_domain();

  auto start = std::chrono::high_resolution_clock::now();
  CG_solver(init_guess, rhs, defect_r, defect_p, values, offsets,
            static_cast<DataType>(1e-2));
  auto end = std::chrono::high_resolution_clock::now();

  DataType const residual =
      cycles::compute_residual(rhs, init_guess, defect_p, values, offsets);

  std::cout << "The residual after the iterations is: " << residual
            << std::endl;

  std::chrono::duration<double> duration = end - start;

  std::cout << "The required time was: " << duration.count() << " seconds"
            << std::endl;

  // std::cout << "The init_guess before subtraction is: " << std::endl;
  //  init_guess.print_domain();

  subtract_domains(init_guess, init_guess, sol);

  // std::cout << "The output Matrix is given by: " << std::endl;
  // init_guess.print_domain();
  //
  //	std::cout<<"The defect is given by: "<<std::endl;
  //
  //
  //	Convolve(defect_r, init_guess, values, offsets);
  //
  //	std::cout<<"The convolution of the output is given by: "<< std::endl;
  //	defect_r.print_domain();
  //	subtract_domains(init_guess, sol, rhs);
  //
  //	init_guess.print_domain();
}
