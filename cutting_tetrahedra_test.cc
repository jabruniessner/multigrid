#include "Domain.h"
#include "hipSYCL/sycl/queue.hpp"
#include "tetraeda_type.h"
#include <iostream>
#include <sycl/sycl.hpp>

int main(int argc, char *argv[]) {

#ifdef DEBUGMODE
  sycl::cpu_selector selector;
#else
  sycl::gpu_selector selector;
#endif

  sycl::queue q{selector,
                sycl::property_list{sycl::property::queue::in_order{}}};

  tetraeda_line_points<3> tet_example;

  domain::Grid<tetraeda_line_points<3>, 3, 10, 10, 10> tet_grid(
      Paddings::PERIODIC, q, 1);

  std::cout << "Hello World!" << std::endl;

  using vector3d = vector<DataType, 3>;

  vector3d footpoint(0.25f, 0.f, 0.f), center(0.f, 0.f, 0.f),
      direction(1.f, 0.f, 0.f);

  DataType isec_p =
      find_intersection_point_sphere<3>(footpoint, direction, center, 1.f);

  std::cout << "The intersection distance is " << isec_p << std::endl;

  return 0;
}
