#include "CG_solver_simplified.h"
#include "Convolution.h"
#include <algorithm>
#include <chrono>

#ifdef __CUDACC__
#include <thrust/iterator/counting_iterator.h>
#else
#include <boost/iterator/counting_iterator.hpp>
#endif

template <typename d_type> struct sol_set_boundary_conditions {

  __host__ __device__ void operator()(int idx) const {
    constexpr auto length = d_type::length[0];
    auto I = domain::flat_to_multi_index<length + 2, length + 2>(idx);
    sol(0, I[0], I[1]) = -(DataType)I[1] / (length + 1.);
    sol(length + 1, I[0], I[1]) = -(DataType)I[1] / (length + 1.);
    sol(I[0], 0, I[1]) = -(DataType)I[1] / (length + 1.);
    sol(I[0], length + 1, I[1]) = -(DataType)I[1] / (length + 1.);
    sol(I[0], I[1], length + 1) = -1.;
  }

  d_type sol;
};

int main() {

  using namespace cg_solver;
  using namespace convolution;

  using OffsetType = domain::array<int, 3>;

  constexpr std::size_t length = 3;

  Domain<3, length, length, length> sol(Paddings::PERIODIC, 1);
  Domain<3, length, length, length> init_guess(Paddings::PERIODIC, 1);
  Domain<3, length, length, length> rhs(Paddings::PERIODIC, 1);
  Domain<3, length, length, length> defect_r(Paddings::PERIODIC, 1);
  Domain<3, length, length, length> defect_p(Paddings::PERIODIC, 1);

  constexpr domain::array<DataType, 7> values = {6, -1, -1, -1, -1, -1, -1};
  constexpr domain::array<OffsetType, 7> offsets{{{0, 0, 0},
                                                  {1, 0, 0},
                                                  {-1, 0, 0},
                                                  {0, 1, 0},
                                                  {0, -1, 0},
                                                  {0, 0, 1},
                                                  {0, 0, -1}}};

#ifdef __CUDACC__
  using iterator = thrust::counting_iterator<int>;
#else
  using iterator = boost::iterator::counting_iterator<int>;
#endif

  iterator start(0);
  iterator end((length + 2) * (length + 2));

  sol_set_boundary_conditions<decltype(sol)> set_bnd_cond{sol};

  thrust::for_each(start, end, set_bnd_cond);

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

  for (int k = 0; k < 5; k++) {
    for (int j = 0; j < 5; j++) {
      for (int i = 0; i < 5; i++) {
        std::cout << init_guess(i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  std::cout << "The defect is given by: " << std::endl;
}
