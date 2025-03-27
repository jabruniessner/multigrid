#include "Atom_types.h"
#include "Domain.h"
#include "blas.h"
#include "dot_finder.h"
#include "hipSYCL/sycl/device_selector.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include "scientific_quantities.h"
#include "tetraeda_type.h"
#include <iostream>
#include <sycl/sycl.hpp>

constexpr std::size_t Dim = 3;

constexpr DataType sqrt3inv = 1 / const_sqrt(3.);
constexpr DataType sqrt2inv = 1 / const_sqrt(2.);

using vector3d = blas::vector<DataType, Dim>;
using Position3D = std::array<int, Dim>;
inline constexpr DataType norm(vector3d vec) {
  DataType n = 0.f;
  for (auto i : vec)
    n += i * i;

  return std::sqrt(n);
}

template <Dimension Dim, typename DataType>
constexpr inline DataType compute_interesect_for_no_princ(
    std::uint8_t i, DataType &grid_step, blas::vector<DataType, Dim> &point,
    blas::vector<DataType, Dim> &center, DataType &radius) {

  std::uint8_t distance_num =
      static_cast<std::uint8_t>(((i >> 1 | i << (3 - 1)) & 7));

  vector3d distance{(distance_num & 1) * grid_step,
                    ((distance_num >> 1) & 1) * grid_step,
                    ((distance_num >> 2) & 1) * grid_step};

  vector3d point_setoff =
      point + vector3d{(i & 1) * grid_step, ((i >> 1) & 1) * grid_step,
                       ((i >> 2) & 1) * grid_step};

  return find_intersection_point_sphere<Dim>(point_setoff, distance, center,
                                             radius);
}

// std::countr_zero
// std::countl_zero

// example of cyclic bit shift
//   static_cast<std::uint8_t>(((i << 1 | i >> (3 - 1)) & ~248) | i),
//   static_cast<std::uint8_t>(((i >> 1 | i << (3 - 1)) & ~248) | i)};

inline constexpr bool in_sphere(const vector3d &point, const vector3d &center,
                                DataType &radius) {
  return norm(point - center) < radius;
}

void find_polygon_cuts(vector3d &point, vector3d &center, DataType &radius,
                       DataType grid_step) {
  // Iteration over all cubes

  std::array<DataType, 19> lengths{};
  std::uint8_t points = 0;
  for (std::uint8_t i = 0; i < 8; i++) {
    const vector3d neighbour_point =
        point + vector3d{(i & 1) * grid_step, ((i >> 1) & 1) * grid_step,
                         ((i >> 2) & 1) * grid_step};

    vector3d a{(i & 1) * grid_step, ((i >> 1) & 1) * grid_step,
               ((i >> 2) & 1) * grid_step};

    if (in_sphere(neighbour_point, center, radius)) {
      points |= (1 << i);
    }
  }

  // Now we know which of the points are in inside the sphere
  // Now iterating over all

  // Iterating over all edges connected to 0;
  {
    bool zero_in_sphere = (bool)(points & 1);
    for (std::uint8_t i = 1; i < 7; i++) {
      if (static_cast<bool>((points >> i) & 1) == zero_in_sphere)
        continue;

      vector3d distance{(i & 1) * grid_step, ((i >> 1) & 1) * grid_step,
                        ((i >> 2) & 1) * grid_step};
      distance /= norm(distance);

      DataType isec_p =
          find_intersection_point_sphere<Dim>(point, distance, center, radius);

      lengths[i] = isec_p;
    };
  }
  // Iterating over all edges connected to 7;
  {
    bool seven_in_sphere = (bool)((points >> 7) & 1);
    for (std::int8_t i = -1; i > -7; i--) {
      std::uint8_t point_num = 7 + i;
      // Checking whether the other point is on the other side of the surface
      if (static_cast<bool>((points >> point_num) & 1) == seven_in_sphere)
        continue;

      vector3d distance{(i & 1) * (-grid_step), ((i >> 1) & 1) * (-grid_step),
                        ((i >> 2) & 1) * (-grid_step)};

      distance /= norm(distance);

      DataType isec_p =
          find_intersection_point_sphere<Dim>(point, distance, center, radius);

      lengths[7 - i - 1] = isec_p;
    };
  }

  // Iterating over all edges that are connected to each other
  {
    int edge_number = 14;
    for (std::uint8_t i = 1; i <= 4; i *= 2) {

      bool this_in_sphere = (bool)((points >> i) & 1);
      std::uint8_t other_point_1 =
          static_cast<std::uint8_t>(((i >> 1 | i << (3 - 1)) & 7) | i);

      bool other_in_sphere = (bool)((points >> other_point_1) & 1);

      if (other_in_sphere != this_in_sphere) {
        lengths[edge_number++] = compute_interesect_for_no_princ(
            i, grid_step, point, center, radius);
      }

      std::uint8_t other_point_2 =
          static_cast<std::uint8_t>(((i << 1 | i >> (3 - 1)) & 7) | i);

      other_in_sphere = (bool)((points >> other_point_2) & 1);

      if (other_in_sphere != this_in_sphere) {
        lengths[edge_number++] = compute_interesect_for_no_princ(
            i, grid_step, point, center, radius);
      }
    }
  }

  for (DataType num : lengths)
    std::cout << num << " ";

  std::cout << std::endl;

  // Now iterating over all the tetrahedra and computing the
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
  atom_host.Position[0] = 50;
  atom_host.Position[1] = 50;
  atom_host.Position[2] = 50;
  atom_host.radius = 45;

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

  // constexpr DataType grid_step = 1;

  // auto start = std::chrono::high_resolution_clock::now();
  // auto edge_cubes = finding_edge_cubes(atom_host, grid_step);
  // int j = 0;
  // for (auto edge_cube : edge_cubes) {
  //   for (auto i : edge_cube) {
  //     std::cout << i << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // auto end = std::chrono::high_resolution_clock::now();

  // std::chrono::duration<double> duration = end - start;

  // std::cout << "The required time was: " << duration.count() << std::endl;

  // std::cout << "The number of points considered is: " << j << std::endl;

  DataType grid_step = 1.f;

  vector3d center{0.1f, 0.1f, 0.1f};
  DataType Radius = std::sqrt(3) * 10.f;
  vector3d point{1.f, 0.f, 0.f};
  point = point * (Radius - 0.2f);

  find_polygon_cuts(point, center, Radius, grid_step);

  return 0;
}
