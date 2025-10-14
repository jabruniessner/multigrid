#include "Domain.h"
#include "dot_finder.h"
#include "predefinitions.h"
#include "utils.h"
#include <array>
#include <boost/iterator/counting_iterator.hpp>
#include <chrono>
#include <cstddef>
#include <execution>
#include <iostream>
#include <random>

int main() {

  constexpr std::size_t Dim = 3;
  constexpr std::size_t num_spheres = 400000;

  std::cout << "Before algorithm" << std::endl;
  domain::Domain<Dim, 351, 351, 351> domain(Paddings::PERIODIC, 1);

  // std::cout << "Hello World!" << std::endl;
  std::random_device rd;

  std::mt19937 gen(rd());
  double lower_bound = 10.0;
  double upper_bound = 340.0;

  std::uniform_real_distribution<double> dist(lower_bound, upper_bound);

  std::vector<DataType> x_pos(num_spheres), y_pos(num_spheres),
      z_pos(num_spheres);
  for (int i = 0; i < num_spheres; i++) {
    x_pos[i] = dist(gen);
    y_pos[i] = dist(gen);
    z_pos[i] = dist(gen);
  }

  DataType *x_pos_d = x_pos.data();
  DataType *y_pos_d = y_pos.data();
  DataType *z_pos_d = z_pos.data();

  auto *Spheres = new Sphere<DataType, 3u>[num_spheres];

  boost::iterators::counting_iterator<int> start(0);
  boost::iterators::counting_iterator<int> end(1);

  std::for_each(std::execution::par_unseq, start, end, [=](int I) {
    Spheres[I].Position[0] = x_pos_d[I];
    Spheres[I].Position[1] = y_pos_d[I];
    Spheres[I].Position[2] = z_pos_d[I];
    Spheres[I].radius = 3.03;
  });

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 1000; i++) {

    std::for_each(std::execution::par_unseq, start, end, [=](int I) {
      find_dots_in_sphere(Spheres[I], domain, (DataType)1.f);
    });
  }

  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end_time - start_time;

  std::cout << "The required time was: " << duration.count() << " seconds"
            << std::endl;
}
