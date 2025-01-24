#include "Domain.h"
#include "concepts.h"
#include "create_charge_distribution.h"
#include "hipSYCL/sycl/device_selector.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>

int main() {

  sycl::cpu_selector selector;
  sycl::queue q{selector,
                sycl::property_list{sycl::property::queue::in_order{}}};

  std::cout << "Hello World!" << std::endl;
  constexpr std::size_t Dim = 3;
  constexpr std::size_t stride = 2;

  domain::Domain<Dim, stride, stride, stride> domain(Paddings::PERIODIC, q, 1);
  std::array<DataType, Dim> a{1.5, 1, 1};

  std::cout << "Before adding charges: " << std::endl;
  domain.print_domain();

  add_charges_to_distribution(domain, a, spacing<DataType, 1.f>{});

  std::cout << "After adding charges:" << std::endl;
  domain.print_domain();

  return 0;
}
