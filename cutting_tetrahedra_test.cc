#include "Atom_types.h"
#include "Domain.h"
#include "dot_finder.h"
#include "hipSYCL/sycl/device_selector.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include "tetraeda_type.h"
#include <iostream>
#include <sycl/sycl.hpp>

constexpr std::size_t Dim = 3;

void find_polygon_cuts(std::array<int, Dim> point) {
  // Iteration over all cubes
  for (std::uint8_t i = 0; i < 8; i++) {

    // Iteration over all lines
    for (std::uint8_t i = 1; i < 8; i++) {
    }
    // Iteration over all tetrahedra
    for (std::uint8_t i = 1; i < 6; i++) {
    }
  }
}

int main(int argc, char *argv[]) {

  // #ifdef DEBUGMODE
  //   sycl::cpu_selector selector;
  // #else
  //   sycl::gpu_selector selector;
  // #endif
  //
  // Using host device for prove of concept.

  sycl::queue q{sycl::host_selector{},
                sycl::property_list{sycl::property::queue::in_order{}}};

  tetraeda_line_points<3> tet_example;

  Atom<DataType> atom_host;
  atom_host.Position[0] = 6;
  atom_host.Position[1] = 6;
  atom_host.Position[2] = 6;
  atom_host.radius = 5;

  // Atom<DataType> *atom_device = sycl::malloc_host<Atom<DataType>>(1, q);
  // q.memcpy(atom_device, &atom_host, sizeof(Atom<DataType>)).wait();
  domain::Grid<tetraeda_line_points<3>, 3, 100, 100, 100> tet_grid(
      Paddings::PERIODIC, q, 1);
  // std::cout << "Hello World!" << std::endl;
  // using vector3d = vector<DataType, 3>;
  // vector3d footpoint(0.25f, 0.f, 0.f), center(0.f, 0.f, 0.f),
  //    direction(1.f, 0.f, 0.f);
  // DataType isec_p =
  //    find_intersection_point_sphere<3>(footpoint, direction, center, 1.f);
  // std::cout << "The intersection distance is " << isec_p << std::endl;
  //

  constexpr DataType grid_step = 1;

  auto edge_points = line_sphere_intersection(atom_host, grid_step);
  for (auto edge_point : edge_points) {
    for (auto i : edge_point) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
