#include "Domain.h"
#include "dot_finder.h"
#include "hipSYCL/sycl/queue.hpp"
#include "predefinitions.h"
#include "utils.h"
#include <array>
#include <cstddef>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
// #include <utility>

int main() {

  sycl::cpu_selector selector;
  sycl::queue q{selector,
                sycl::property_list{sycl::property::queue::in_order{}}};

  constexpr std::size_t Dim = 3;

  std::cout << "Before algorithm" << std::endl;
  domain::Domain<Dim, 351, 351, 351> domain(Paddings::PERIODIC, q, 1);
  // domain.print_domain();

  // std::cout << "Hello World!" << std::endl;
  std::random_device rd;

  std::mt19937 gen(rd());
  double lower_bound = 0.0;
  double upper_bound = 1.0;

  auto *Spheres = sycl::malloc_device<Sphere<DataType, 3u>>(1, q);

  q.submit([=](sycl::handler &h) {
     h.single_task([=]() {
       Spheres->Position[0] = 115.192108;
       Spheres->Position[1] = 324.394073;
       Spheres->Position[2] = 168.093201;
       Spheres->radius = 3.0;
     });
   }).wait();

  // q.submit([=](sycl::handler &h) {
  //    h.single_task([=]() { find_dots_in_sphere(Spheres[0], domain, 1.f); });
  // }).wait();

  // domain(0) = 1.;
  std::cout << "After algo" << std::endl;
  find_dots_in_sphere(Spheres[0], domain, static_cast<DataType>(1.));
  // domain.print_domain();

  // std::cout << "The sphere position is: " << Spheres->Position[0] <<
  // std::endl; std::cout << "The sphere radius is: " << Spheres->radius <<
  // std::endl;

  // domain.print_domain();

  // print_index_sequence(std::make_index_sequence<0u>{});
}
