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
inline DataType find_intersection_point_sphere(VType footpoint, VType direction,
                                               VType center, DataType radius) {

  VType distance_vec = footpoint - center;
  DataType p_half = distance_vec * direction;
  DataType p_half_squared = blas::square(p_half);
  DataType distance_vec_squared = distance_vec * distance_vec;
  DataType radius_squared = blas::square(radius);
  DataType q = distance_vec_squared - radius_squared;

  // assert(p_half_squared - q >= 0);

  // std::cout << "p_half: " << p_half << std::endl;
  // std::cout << "p_half_squared: " << p_half_squared << ", q: " << q
  //           << std::endl;

  if (p_half_squared - q < 0) {
    return -p_half;
  }

  DataType result =
      (1 - 2 * (distance_vec_squared >= radius_squared && p_half < 0)) *
          std::sqrt(p_half_squared - q) -
      p_half;

  // std::cout << "result: " << result << std::endl;

  return result;
  // return 0;
}

#endif
