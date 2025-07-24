
#include "bitshift_lib.h"
#include <array>

#ifndef ITERATE_TETS_H
#define ITERATE_TETS_H

template <Dimension Dim, Dimension helper_dim, UnsignedIntegral num_type>
inline void iterate_over_tets_helper(std::array<num_type, Dim + 1> &a,
                                     num_type indices) {
  if constexpr (helper_dim == Dim) {
    a[helper_dim] = utils::Power<2, Dim>::value - 1;
    for (num_type i : a) {
      std::cout << static_cast<int>(i) << " ";
    }
    std::cout << std::endl;

  } else {

    constexpr num_type mask = utils::Power<2, Dim>::value - 1;

    for (; indices != 0; indices &= indices - 1) {
      auto least_significant_bit = indices & (~indices + 1);
      a[helper_dim] = a[helper_dim - 1] + least_significant_bit;
      num_type indices_new = (~a[helper_dim]) & mask;
      iterate_over_tets_helper<Dim, helper_dim + 1>(a, indices_new);
    }
  }
}

template <Dimension Dim, UnsignedIntegral num_type>
inline void iterate_over_tets() {
  constexpr auto upper_limit = utils::Power<2, Dim>::value;
  // number of corners in tetrahedra: Dim+1
  std::array<num_type, Dim + 1> a{};
  // Last corner will always be 2^Dim-1
  constexpr num_type mask = utils::Power<2, Dim>::value - 1;

  for (std::size_t i = 1; i < upper_limit; i *= 2) {
    a[1] = i;
    num_type indices = (~i) & mask;
    iterate_over_tets_helper<Dim, 2>(a, indices);
  }
}

#endif
