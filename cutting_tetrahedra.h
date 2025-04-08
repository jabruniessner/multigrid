#include "Atom_types.h"
#include "Domain.h"
#include "blas.h"
#include "cubes_cutter.h"
#include "hipSYCL/sycl/device_selector.hpp"
#include "hipSYCL/sycl/libkernel/atomic_ref.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include "scientific_quantities.h"
#include "tetraeda_type.h"
#include <array>
#include <boost/container/static_vector.hpp>
// #include <cblas.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <sycl/sycl.hpp>
#include <utility>

#ifndef CUTTING_TETERAHEDRA_H
#define CUTTING_TETERAHEDRA_H

namespace Cutter_utils {

constexpr std::size_t Dim = 3;

constexpr DataType sqrt3inv = 1 / const_sqrt(3.);
constexpr DataType sqrt2inv = 1 / const_sqrt(2.);

constexpr DataType Upper_limit =
    static_cast<DataType>(std::numeric_limits<std::uint32_t>::max());

using vector3d = blas::vector<DataType, Dim>;
using Position3D = std::array<int, Dim>;

using Face = boost::container::static_vector<vector3d, 3>;

inline constexpr DataType norm(vector3d vec) {
  DataType n = static_cast<DataType>(0);
  for (auto i : vec)
    n += i * i;

  return std::sqrt(n);
}

template <Dimension Dim, typename DataType>
constexpr inline DataType compute_interesect_for_no_princ(
    std::uint8_t i, std::uint8_t distance_num, DataType &grid_step,
    blas::vector<DataType, Dim> &point, blas::vector<DataType, Dim> &center,
    DataType &radius) {

  // std::uint8_t distance_num =
  //     static_cast<std::uint8_t>(((i >> 1 | i << (3 - 1)) & 7));

  vector3d distance{(distance_num & 1) * grid_step,
                    ((distance_num >> 1) & 1) * grid_step,
                    ((distance_num >> 2) & 1) * grid_step};

  vector3d point_setoff =
      point + vector3d{(i & 1) * grid_step, ((i >> 1) & 1) * grid_step,
                       ((i >> 2) & 1) * grid_step};

  return find_intersection_point_sphere<Dim>(point_setoff, distance, center,
                                             radius);
}

inline constexpr bool in_sphere(const vector3d &point, const vector3d &center,
                                DataType &radius) {
  return norm(point - center) <= radius;
}

template <typename DataType>
void find_polygon_cuts(vector3d &point, vector3d &center, DataType &radius,
                       DataType grid_step, std::uint32_t *lengths) {
  // Iteration over all cubes
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

  if (points == 0 || points == 255)
    return;

  {
    bool zero_in_sphere = (bool)(points & 1);
    for (std::uint8_t i = 1; i <= 7; i++) {

      bool other = (static_cast<bool>((points >> i) & 1));
      if (other == zero_in_sphere)
        continue;

      vector3d distance{(i & 1) * grid_step, ((i >> 1) & 1) * grid_step,
                        ((i >> 2) & 1) * grid_step};

      DataType dist_norm = norm(distance);
      distance /= dist_norm;

      DataType isec_p = i;
      // find_intersection_point_sphere<Dim>(point, distance, center, radius);

      std::uint32_t isec_p_int =
          static_cast<std::uint32_t>(isec_p / dist_norm * Upper_limit);

      sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                       sycl::memory_scope::device>
          edge(lengths[i % 7]);

      // zero_in_sphere && (!other) ? edge = edge.fetch_min(isec_p_int)
      //                            : edge = edge.fetch_max(isec_p_int);

      edge = isec_p_int;
    };
  }
  // Iterating over all edges connected to 7;
  {
    bool seven_in_sphere = (bool)((points >> 7) & 1);
    for (std::uint8_t i = 1; i < 7; i++) {
      std::uint8_t point_num = 7 - i;
      // Checking whether the other point is on the other side of the surface
      bool other = (static_cast<bool>((points >> point_num) & 1));
      if (other == seven_in_sphere)
        continue;

      vector3d distance{(i & 1) * (-grid_step), ((i >> 1) & 1) * (-grid_step),
                        ((i >> 2) & 1) * (-grid_step)};

      DataType dist_norm = norm(distance);

      distance /= dist_norm;

      auto point7 = point + vector3d{grid_step, grid_step, grid_step};

      DataType isec_p = i;
      // find_intersection_point_sphere<Dim>(point7, distance, center, radius);

      std::uint32_t isec_p_int =
          static_cast<std::uint32_t>(isec_p / dist_norm * Upper_limit);

      sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                       sycl::memory_scope::device>
          edge(lengths[7 + i - 1]);

      // seven_in_sphere && (!other) ? edge = edge.fetch_min(isec_p_int)
      //                             : edge = edge.fetch_max(isec_p_int);

      edge = isec_p_int;
    };
  }

  // Iterating over all edges that are connected to each other
  {
    int edge_number = 13;
    for (std::uint8_t i = 1; i <= 4; i *= 2) {

      bool this_in_sphere = (bool)((points >> i) & 1);
      std::uint8_t other_point_1 =
          static_cast<std::uint8_t>(((i >> 1 | i << (3 - 1)) & 7) | i);

      bool other_in_sphere = (bool)((points >> other_point_1) & 1);

      std::uint8_t dir = other_point_1 - i;

      if (other_in_sphere != this_in_sphere) {
        auto isec_p = i;

        // compute_interesect_for_no_princ(i, dir, grid_step, point, center,
        //                                 radius);

        std::uint32_t isec_p_int =
            static_cast<std::uint32_t>(isec_p * Upper_limit);

        sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>
            edge(lengths[edge_number]);

        // this_in_sphere ? edge.fetch_max(isec_p_int)
        //                : edge.fetch_min(isec_p_int);

        edge = isec_p_int;
      }

      edge_number++;
      std::uint8_t other_point_2 =
          static_cast<std::uint8_t>(((i << 1 | i >> (3 - 1)) & 7) | i);

      other_in_sphere = (bool)((points >> other_point_2) & 1);

      dir = other_point_2 - i;

      if (other_in_sphere != this_in_sphere) {

        auto isec_p = i;

        // compute_interesect_for_no_princ(i, dir, grid_step, point, center,
        //                                 radius);
        std::uint32_t isec_p_int =
            static_cast<std::uint32_t>(isec_p * Upper_limit);
        sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>
            edge(lengths[edge_number]);
        // this_in_sphere ? edge.fetch_max(isec_p_int)
        //                : edge.fetch_min(isec_p_int);

        edge = isec_p_int;
      }

      edge_number++;
    }
  }
}

template <template <typename> typename container>
void print_faces_to_ply(std::ostream &stream, container<Face> &faces) {
  stream << "ply\n";
  stream << "format ascii 1.0\n";
  stream << "element vertex " << faces.size() * 3 << "\n";
  stream << "property float x\n";
  stream << "property float y\n";
  stream << "property float z\n";
  stream << "element face " << faces.size() << "\n";
  stream << "property list uchar int vertex_index\n";
  stream << "end_header\n";

  for (auto face : faces) {
    for (auto point : face) {
      stream << point[0] << " " << point[1] << " " << point[2] << "\n";
    }
  }

  int index = 0;
  for (int i = 0; i < faces.size(); i++) {
    stream << "3 " << index << " " << index + 1 << " " << index + 2 << "\n";
    index += 3;
  }

  stream << std::endl;
}

} // namespace Cutter_utils

#endif // CUTTING_TETERAHEDRA_H
