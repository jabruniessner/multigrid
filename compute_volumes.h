#include "Atom_types.h"
#include "Domain.h"
#include "blas.h"
#include "concepts.h"
#include "cutting_tetrahedra.h"
#include "predefinitions.h"
#include "scientific_quantities.h"
#include "tetraeda_type.h"
#include "tprint.hpp"
#include "utils.h"
#include <array>
#include <bits/elements_of.h>
#include <cstddef>
#include <cstdint>
#include <generator>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <sycl/sycl.hpp>
#include <tuple>
#include <utility>

#ifndef COMPUTE_VOLUMES_H
#define COMPUTE_VOLUMES_H

namespace volume_computer {

using vector3d = blas::vector<DataType, 3>;

inline DataType two_point_helper(std::array<DataType, 4> &lengths,
                                 const vector3d &a, const vector3d &b,
                                 const vector3d &c, const vector3d &first,
                                 const vector3d &second) {

  DataType volume_data = 0;

  volume_data += lengths[0] * lengths[1];
  volume_data +=
      std::abs(determinant<DataType, 3>(a - first, b - first, c - first));
  volume_data +=
      std::abs(determinant<DataType, 3>(second - first, b - first, c - first));
  // volume_data +=
  //     std::abs(determinant<DataType, 3>(a - second, b - second, c - second));
  //      I think the line above is incorrect
  return volume_data;
}

inline DataType conv32(std::uint32_t value) {
  return static_cast<DataType>(value) /
         static_cast<DataType>(std::numeric_limits<std::uint32_t>::max());
}

struct Volume_comp {

  void operator()(std::span<std::uint32_t, 19> tet_grid_span,
                  const std::uint8_t point,
                  std::span<DataType, 6> tetrahedra_span,
                  std::span<DataType, 3> sphere_position,
                  const DataType sphere_radius, const DataType grid_step,
                  std::size_t position_0, std::size_t position_1,
                  std::size_t position_2) {

    std::array<std::uint8_t, 6> tetrahedra_array{0, 0, 0, 0, 0, 0};
    if (point & 1)
      for (int i = 0; i < 6; ++i)
        tetrahedra_array[i] += 1;

    if ((point >> 7) & 1)
      for (int i = 0; i < 6; ++i)
        tetrahedra_array[i] += 1;

    for (int i = 0, j = 1; i < 3; ++i, j *= 2) {
      if ((point >> j) & 1) {
        tetrahedra_array[2 * i] += 1;
        tetrahedra_array[2 * i + 1] += 1;
      }

      std::uint8_t other_point_1 =
          static_cast<std::uint8_t>(((j >> 1 | j << (3 - 1)) & 7) | j);

      std::uint8_t other_point_2 =
          static_cast<std::uint8_t>(((j << 1 | j >> (3 - 1)) & 7) | j);

      std::uint8_t other_point = j | other_point_1;

      if ((point >> other_point) & 1)
        tetrahedra_array[2 * i] += 1;

      other_point = j | other_point_2;
      if ((point >> other_point) & 1)
        tetrahedra_array[2 * i + 1] += 1;
    }

    for (int i = 0, j = 1; i < 3; ++i, j *= 2)
      for (int k = 0; k < 2; ++k) {

        std::uint8_t other_point = static_cast<std::uint8_t>(
            ((j >> (1 + k) | j << (3 - (1 + k))) & 7) | j);

        // other_point = other_point + j; //This is deleted because addition
        // with j is already in the line above

        if (tetrahedra_array[2 * i + k] == 0)
          tetrahedra_span[2 * i + k] = 0;
        else if (tetrahedra_array[2 * i + k] == 4)
          tetrahedra_span[2 * i + k] = 1.f;
        else if (tetrahedra_array[2 * i + k] == 1 ||
                 tetrahedra_array[2 * i + k] == 3) {
          std::uint8_t one_point =
              tetrahedra_array[2 * i + k] == 1 ? point : ~point;
          auto volume =
              ((one_point >> 7) & 1)
                  ? (1 - conv32(tet_grid_span[0])) *
                        conv32(tet_grid_span[13 - j]) *
                        conv32(tet_grid_span[13 - other_point])
              : (one_point & 1)
                  ? conv32(tet_grid_span[0]) * conv32(tet_grid_span[j]) *
                        conv32(tet_grid_span[other_point])
              : (one_point >> j) & 1
                  ? (1 - conv32(tet_grid_span[j])) *
                        (1 - conv32(tet_grid_span[13 - j])) *
                        conv32(tet_grid_span[13 + 2 * i + k])
                  : (1 - conv32(tet_grid_span[other_point])) *
                        (1 - conv32(tet_grid_span[13 - other_point])) *
                        (1 - conv32(tet_grid_span[13 + 2 * i + k]));

          tetrahedra_span[2 * i + k] =
              tetrahedra_array[2 * i + k] == 1 ? volume : 1 - volume;

        } else // if(tetrahedra_array[2*i+k]==2)
        {
          // This is the place where I will have to resume tomorrow
          //(Only the case for two points left)

          std::array<DataType, 4> length{};
          vector3d first{};
          vector3d second{};
          vector3d a{};
          vector3d b{};
          vector3d c{};

          auto negative = false;

          if (((point & 1) &&
               ((point >> j) & 1)) || // For points (0,1) or (2, 3)
              (((point >> 7) & 1) && ((point >> other_point) & 1))) {

            length = {conv32(tet_grid_span[other_point]),
                      conv32(tet_grid_span[0]),
                      conv32(tet_grid_span[13 + 2 * i + k]),
                      1 - conv32(tet_grid_span[13 - j])};

            first = {0.f, 0.f, 1.f};
            second = first + vector3d{1.f, 1.f, 0.f} * length.back();

            a = {0.f, length[0], length[0]};
            b = {length[1], length[1], length[1]};
            c = {0.f, length[2], 1.f};

            negative = (point >> 7) & 1;

          } else if ((((point >> j) & 1) && ((point >> other_point) & 1)) ||
                     ((point & 1) &&
                      ((point >> 7) & 1))) // for points (1,2) or (0,3)
          {

            length = {1 - conv32(tet_grid_span[j]),
                      1 - conv32(tet_grid_span[13 - j]),
                      1 - conv32(tet_grid_span[other_point]),
                      1 - conv32(tet_grid_span[13 - other_point])};

            first = {0.f, 1.f, 1.f};
            second = first + vector3d{1.f, 0.f, 0.f} * length.back();

            a = {0.f, 0.f, 1 - length[0]};
            b = {length[1], length[1], 1.f};
            c = {0.f, 1 - length[2], 1 - length[2]};

            negative = point & 1;

          } else if (((point & 1) && ((point >> other_point) & 1)) ||
                     (((point >> 7) & 1) &&
                      ((point >> j) & 1))) // for points (0,2) or (1, 3)
          {
            length = {conv32(tet_grid_span[j]), conv32(tet_grid_span[0]),
                      1 - conv32(tet_grid_span[13 + 2 * i + k]),
                      1 - conv32(tet_grid_span[13 - other_point])};

            first = {0.f, 1.f, 1.f};
            second = first + vector3d{1.f, 0.f, 0.f} * length.back();

            a = {0.f, 0.f, length[0]};
            b = {length[1], length[1], length[1]};
            c = {0.f, 1 - length[2], 1.f};

            negative = (point >> 7) & 1;
          }

          auto volume = two_point_helper(length, a, b, c, first, second);
          tetrahedra_span[2 * i + k] = !negative ? volume : 1 - volume;
        }
      }

    // Implementation of volume computation
  }

  template <std::size_t Dim, std::size_t... side_lengths>
  void operator()(
      const domain::Grid<std::uint32_t, Dim + 1, side_lengths..., 19> &tet_grid,
      const domain::Grid<std::uint32_t, Dim, side_lengths...> &inside_outside,
      const domain::Grid<DataType, Dim + 1, side_lengths..., 6> &tetrahedra,
      const DataType *sphere_position, const DataType sphere_radius,
      const DataType grid_step, const std::size_t position_0,
      const std::size_t position_1, const std::size_t position_2) const {

    auto &point = inside_outside(position_0, position_1, position_2);
    std::array<std::uint8_t, 6> tetrahedra_array{};
    std::span<DataType> tetrahedra_span(
        &tetrahedra(position_0, position_1, position_2, 0), 6);

    std::span<std::uint32_t> tet_grid_span(
        &tet_grid(position_0, position_1, position_2, 0), 19);

    this->operator()(tet_grid_span, point, tetrahedra_span, sphere_position,
                     sphere_radius, grid_step, position_0, position_1,
                     position_2);
  }
}; // Struct Volume_comp

} // namespace volume_computer

#endif // COMPUTE_VOLUMES_H
