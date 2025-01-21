#include "Domain.h"
#include "dot_finder.h"
#include "predefinitions.h"
#include "utils.h"
#include <array>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
// #include <utility>

int main() {

  sycl::gpu_selector selector;
  sycl::queue q(selector,
                sycl::property_list{sycl::property::queue::in_order{}});

  constexpr std::size_t Dim = 3;
  constexpr std::size_t num_spheres = 400000;

  std::cout << "Before algorithm" << std::endl;
  domain::Domain<Dim, 351, 351, 351> domain(Paddings::PERIODIC, q, 1);

  // std::cout << "Hello World!" << std::endl;
  std::random_device rd;

  std::mt19937 gen(rd());
  double lower_bound = 10.0;
  double upper_bound = 340.0;

  std::uniform_real_distribution<double> dist(lower_bound, upper_bound);

  std::array<DataType, num_spheres> x_pos, y_pos, z_pos;
  for (int i = 0; i < num_spheres; i++) {
    x_pos[i] = dist(gen);
    y_pos[i] = dist(gen);
    z_pos[i] = dist(gen);
  }

  auto *x_pos_d = sycl::malloc_device<DataType>(num_spheres, q);
  auto *y_pos_d = sycl::malloc_device<DataType>(num_spheres, q);
  auto *z_pos_d = sycl::malloc_device<DataType>(num_spheres, q);

  q.memcpy(x_pos_d, x_pos.data(), sizeof(DataType) * num_spheres).wait();
  q.memcpy(y_pos_d, y_pos.data(), sizeof(DataType) * num_spheres).wait();
  q.memcpy(z_pos_d, z_pos.data(), sizeof(DataType) * num_spheres).wait();

  auto *Spheres = sycl::malloc_device<Sphere<DataType, 3u>>(num_spheres, q);

  q.parallel_for(sycl::range<1>(num_spheres), [=](sycl::id<1> I) {
     Spheres[I].Position[0] = x_pos_d[I];
     Spheres[I].Position[1] = y_pos_d[I];
     Spheres[I].Position[2] = z_pos_d[I];
     Spheres[I].radius = 3.03;
   }).wait();

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 1000; i++) {
    q.parallel_for(sycl::range<1>(num_spheres), [=](sycl::id<1> I) {
      find_dots_in_sphere(Spheres[I], domain, 1.f);
    });
  }

  q.wait();

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;

  std::cout << "The required time was: " << duration.count() << " seconds"
            << std::endl;

  // for (int i = 0; i < num_spheres; i++) {
  //   find_dots_in_sphere(Spheres[i], domain, 1.f);
  // }

  //  q.submit([=](sycl::handler &h) {
  //     h.single_task([=]() {
  //       Spheres->Position[0] = 3. / 2.;
  //       Spheres->Position[1] = 3. / 2.;
  //       Spheres->Position[2] = 3. / 2.;
  //       Spheres->radius = 1.0;
  //     });
  //   }).wait();

  // q.submit([=](sycl::handler &h) {
  //    h.single_task([=]() { find_dots_in_sphere(Spheres[0], domain, 1.f); });
  // }).wait();

  // domain(0) = 1.;
  // std::cout << "After algo" << std::endl;
  // find_dots_in_sphere(Spheres[0], domain, 0.5f);

  // std::cout << "The sphere position is: " << Spheres->Position[0] <<
  // std::endl; std::cout << "The sphere radius is: " << Spheres->radius <<
  // std::endl;

  // domain.print_domain();

  // print_index_sequence(std::make_index_sequence<0u>{});
}
