#include "Domain.h"
#include "bitshift_lib.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>

#ifndef GET_EPSILON_VALUES_H
#define GET_EPSILON_VALUES_H

template <UnsignedIntegral num_type, Dimension Dim, std::size_t... strides_all,
          typename... Positions>
std::array<num_type, 2 * Dim> get_tet_vals_dim(
    domain::Grid<num_type, Dim + 1, strides_all..., utils::factorial(Dim) - 2>
        epsilon_grid,
    Positions... positions) {
  static_assert(
      sizeof...(positions) == Dim,
      "You gave more position indices than there are number of Dimensions");

  std::array<num_type, 2 * Dim> vals{};

  auto ups_and_down = [&](auto a, auto j, num_type dir) {
    num_type b[Dim];
    for (int i = 0; i < Dim; i++)
      b[i] = a[i + 1] - a[i];
    auto it = std::find(b, &b[Dim], dir);
    auto c = std::distance(b, it);
    auto tet_pos_up =
        blas::vector<int, Dim>{positions...} -
        bitshift::convert_byte_to_vec<num_type, num_type, Dim>(a[c]);

    auto tet_pos_down =
        blas::vector<int, Dim>{positions...} -
        bitshift::convert_byte_to_vec<num_type, num_type, Dim>(a[c + 1]);

    auto index = std::countr_zero(dir);

    vals[index] += std::apply(
        [&](auto... positions) { return epsilon_grid(positions..., j); },
        (std::array<int, Dim>)tet_pos_up);

    vals[3 + index] += std::apply(
        [&](auto... positions) { return epsilon_grid(positions..., j); },
        (std::array<int, Dim>)tet_pos_down);
  };

  for (std::uint8_t dir = 1; dir < utils::power_off(2, Dim); dir *= 2)
    iterate_over_tets<Dim, num_type>(
        [&](auto a, auto j) { ups_and_down(a, j, dir); });

  return vals;
}

#endif
