#include "CG_Solver.h"
#include "Convolution.h"
#include <algorithm>
#include <boost/iterator/counting_iterator.hpp>
#include <chrono>

int main() {

  using namespace cg_solver;
  using namespace convolution;

  using OffsetType = std::array<int, 3>;

  constexpr std::size_t length = 31;

  Domain<3, length, length, length> sol(Paddings::PERIODIC, 1);
  Domain<3, length, length, length> init_guess(Paddings::PERIODIC, 1);
  Domain<3, length, length, length> rhs(Paddings::PERIODIC, 1);
  Domain<3, length, length, length> defect_r(Paddings::PERIODIC, 1);
  Domain<3, length, length, length> defect_p(Paddings::PERIODIC, 1);

  constexpr std::array<DataType, 7> values = {6, -1, -1, -1, -1, -1, -1};
  constexpr std::array<OffsetType, 7> offsets{{{0, 0, 0},
                                               {1, 0, 0},
                                               {-1, 0, 0},
                                               {0, 1, 0},
                                               {0, -1, 0},
                                               {0, 0, 1},
                                               {0, 0, -1}}};

  iterator start(0);
  iterator end((length + 2) * (length + 2));

  std::for_each(std::execution::par_unseq, start, end, [=](int idx) {
    auto I = domain::flat_to_multi_index<length + 2, length + 2>(idx);
    sol(0, I[0], I[1]) = -(DataType)I[1] / (length + 1.);
    sol(length + 1, I[0], I[1]) = -(DataType)I[1] / (length + 1.);
    sol(I[0], 0, I[1]) = -(DataType)I[1] / (length + 1.);
    sol(I[0], length + 1, I[1]) = -(DataType)I[1] / (length + 1.);
    sol(I[0], I[1], length + 1) = -1.;
  });

  //  for (int k = 0; k < 5; k++) {
  //    for (int j = 0; j < 5; j++) {
  //      for (int i = 0; i < 5; i++) {
  //        std::cout << sol(i, j, k) << " ";
  //      }
  //      std::cout << std::endl;
  //    }
  //    std::cout << std::endl;
  //  }

  Convolve(rhs, sol, values, offsets);

  auto start_time = std::chrono::high_resolution_clock::now();
  CG_solver(init_guess, rhs, defect_r, defect_p, values, offsets,
            static_cast<DataType>(1e-2));
  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end_time - start_time;
  std::cout << "The required time was: " << duration.count() << " seconds"
            << std::endl;

  subtract_domains(init_guess, init_guess, sol);

  std::cout << "The output Matrix is given by: " << std::endl;

  //  for (int k = 0; k < 5; k++) {
  //    for (int j = 0; j < 5; j++) {
  //      for (int i = 0; i < 5; i++) {
  //        std::cout << init_guess(i, j, k) << " ";
  //      }
  //      std::cout << std::endl;
  //    }
  //    std::cout << std::endl;
  //  }

  std::cout << "The defect is given by: " << std::endl;
}
