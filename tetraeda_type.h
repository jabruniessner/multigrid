#include "blas.h"
#include "predefinitions.h"
#include "utils.h"
#include <array>
#include <cstdint>
#include <iostream>

#ifndef TETRAEDA_TYPE_H
#define TERTAEDA_TYPE_H

template <Dimension Dim>
struct tetraeda_line_points
    : std::array<std::uint8_t, utils::power_off(2, Dim)> {

  using array = typename std::array<std::uint8_t, utils::power_off(2, Dim)>;

  std::uint8_t &operator[](std::size_t index) {
    return this->array::operator[](index - 1);
  }
};

// This function takes in a foot point and the direction vector for a line and
// both the center point and the radius of a sphere and returns the distance at
// which the line intersects the sphere. It is the programmers responsibility to
// ensure that the norm of the direction vector is 1.
template <Dimension Dim, typename VType>
inline constexpr DataType
find_intersection_point_sphere(VType footpoint, VType direction, VType center,
                               DataType radius) {

  VType distance_vec = footpoint - center;
  DataType p_half = distance_vec * direction;
  DataType p_half_squared = blas::square(p_half);
  DataType q = distance_vec * distance_vec - blas::square(radius);

  assert(p_half_squared - q >= 0);
  return std::sqrt(p_half_squared - q) - p_half;

  // return 0;
}

#endif
