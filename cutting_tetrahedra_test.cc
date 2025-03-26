#include "Atom_types.h"
#include "Domain.h"
#include "blas.h"
#include "dot_finder.h"
#include "hipSYCL/sycl/device_selector.hpp"
#include "hipSYCL/sycl/libkernel/marray.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include "scientific_quantities.h"
#include "tetraeda_type.h"
#include <bitset>
#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>

constexpr std::size_t Dim = 3;

constexpr DataType sqrt3inv = 1 / const_sqrt(3.);
constexpr DataType sqrt2inv = 1 / const_sqrt(2.);

using vector3d = sycl::marray<DataType, Dim>;
inline constexpr DataType norm(vector3d vec) {
  DataType n = 0.f;
  for (auto i : vec)
    n += i * i;

  return std::sqrt(n);
}
// std::countr_zero
// std::countl_zero

void find_polygon_cuts(vector3d point, vector3d center, DataType radius) {
  // Iteration over all cubes
  for (std::uint8_t i = 0; i < 8; i++) {

    if (i == 0 || i == 7) {
      int prefact = (1 - 2 * (i == 7));
      DataType prefactf = static_cast<DataType>(prefact);
      // Iteration over all lines
      std::uint8_t cube = (1 << i);

      for (std::uint8_t j = 1; i < 8; i++) {
        auto line = vector3d(((j >> 2) & 1) * prefactf,
                             ((j >> 1) & 1) * prefactf, (j & 1) * prefactf);

        auto neighbour_point = point + line;
        // If the point is outside
        if (norm(neighbour_point - center) < radius) {
          cube |= (1 << (i + j * prefact));
        }
      }
      // Now we know which points are outside and inside

      // Iterating over all edges in the cube
      // All connections to 0
      for (std::uint8_t j = 1; j < 7; j++) {
      }
      // All connections to 7
      for (std::int8_t j = -1; j > -7; j++) {
      }
      // connection between 7-0
      {
      }
      // All connection between other points
      for (std::uint8_t j = 1; j < 8; j++) {
      }

      for (std::uint8_t j = 1; j < 8; j++) {
      }

      // Iteration over all tetrahedra
      for (std::uint8_t j = 1; i < 6; i++) {
      }
    } else if (std::bitset<8>(i).count() == 1) {
      std::array<std::uint8_t, 2> other_points{
          static_cast<std::uint8_t>(((i << 1 | i >> (3 - 1)) & ~248) | i),
          static_cast<std::uint8_t>(((i >> 1 | i << (3 - 1)) & ~248) | i)};

      // Iteration over all lines
      for (std::uint8_t j = 1; j <= 3; j++) {
      }
      // Iteration over all tetrahedra
      for (std::uint8_t j = 1; j <= 2; j++) {
      }
    } else {
      std::uint8_t i_inv = ~i;
      std::array<std::uint8_t, 2> other_points{
          static_cast<std::uint8_t>(((i_inv << 1 | i_inv >> (3 - 1)) & ~248)),
          static_cast<std::uint8_t>(((i_inv >> 1 | i_inv << (3 - 1)) & ~248))};

      // Iteration over all lines
      for (std::uint8_t j = 1; j <= 2; j++) {
      }
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
  atom_host.radius = .5;

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

  auto start = std::chrono::high_resolution_clock::now();
  auto edge_cubes = finding_edge_cubes(atom_host, grid_step);
  int j = 0;
  for (auto edge_cube : edge_cubes) {
    for (auto i : edge_cube) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;

  // std::cout << "The required time was: " << duration.count() << std::endl;

  // std::cout << "The number of points considered is: " << j << std::endl;

  return 0;
}
