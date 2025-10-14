#include "Domain.h"
#include "concepts.h"
#include "create_charge_distribution.h"
#include <cstdlib>
#include <iostream>

int main() {

  std::cout << "Hello World!" << std::endl;
  constexpr std::size_t Dim = 3;
  constexpr std::size_t stride = 2;

  domain::Domain<Dim, stride, stride, stride> domain(Paddings::PERIODIC, 1);
  std::array<DataType, Dim> a{1.5, 1, 1};

  std::cout << "Before adding charges: " << std::endl;
  domain.print_domain();

  add_charges_to_distribution(domain, a, static_cast<DataType>(1.),
                              spacing<DataType, static_cast<DataType>(1.)>{});

  std::cout << "After adding charges:" << std::endl;
  domain.print_domain();

  return 0;
}
