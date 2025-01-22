#include "Domain.h"
#include "hipSYCL/sycl/device_selector.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include <fstream>
#include <iostream>
#include <sycl/sycl.hpp>

int main(int argc, char *argv[]) {

  std::cout << "Hello World!" << std::endl;
  sycl::cpu_selector selector;
  sycl::queue q{selector,
                sycl::property_list{sycl::property::queue::in_order{}}};
  std::ofstream dx_file{"dx_trial.dx"};
  domain::Domain<3, 10, 10, 10> domain(Paddings::PERIODIC, q, 1);

  if (dx_file.is_open()) {
    domain.print_dx_to_stream(dx_file, -1, -1, -1, 11);
  }
  return 0;
}
