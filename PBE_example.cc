#include "Convolution.h"
#include "MultigridDomain.h"
#include "cycles.h"
#include "fileio.h"
#include "hipSYCL/sycl/device_selector.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include "level_transition.h"
#include <array>
#include <chrono>
#include <string>
#include <utility>

using namespace cycles;
using namespace convolution;

int main(int argc, char *argv[]) {
  constexpr Dimension Dim = 3;
  constexpr std::size_t nlev = 4u;
  constexpr std::size_t base_length = 6;
  constexpr DataType omega = 4. / 5.;
  constexpr DataType box_length = 96;

  using OffsetType = std::array<int, Dim>;
  using Domain =
      Multigrid_domain<Dim, nlev, base_length, base_length, base_length>;

  if (argc < 3) {
    std::cout << "Usage: ./this_program filename x_min y_min z_min num_iters"
              << std::endl;
  }
  int num_iters = std::stoi(argv[3]);

#ifdef DEBUGMODE
  sycl::cpu_selector selector;
#else
  sycl::gpu_selector selector;
#endif

  sycl::queue q{selector,
                sycl::property_list{sycl::property::queue::in_order{}}};

  Domain lhs_domain1(q), lhs_domain2(q), rhs_domain(q), boundary_values(q),
      epsilon(q), kappa_(q);

  constexpr auto &length = Domain::length;

  {
    auto &boundary_domain = boundary_values.template get_domain<nlev>();
    q.parallel_for(
         sycl::range<2>(std::get<1>(length) + 2, std::get<2>(length) + 2),
         [=](sycl::id<2> I) {
           boundary_domain(I[0], I[1], 0) = 0.;
           boundary_domain(I[0], I[1], std::get<2>(length) + 1) = 1.;
           boundary_domain(0, I[0], I[1]) =
               (DataType)I[1] / (DataType)(std::get<2>(length) + 1);
           boundary_domain(std::get<0>(length) + 1, I[0], I[1]) =
               (DataType)I[1] / (DataType)(std::get<2>(length) + 1);
           boundary_domain(I[0], 0, I[1]) =
               (DataType)I[1] / (DataType)(std::get<2>(length) + 1);
           boundary_domain(I[0], std::get<1>(length) + 1, I[1]) =
               (DataType)I[1] / (DataType)(std::get<2>(length) + 1);
         })
        .wait();
  }

  std::cout << "The length is: " << std::get<0>(length) << std::endl;

  return 0;
}
