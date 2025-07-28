
#include "bitshift_lib.h"
#include <array>
#include <boost/intrusive/pack_options.hpp>

#ifndef ITERATE_TETS_H
#define ITERATE_TETS_H

template <Dimension Dim, Dimension helper_dim, UnsignedIntegral num_type,
          typename Callable>
inline void iterate_over_tets_helper(blas::vector<num_type, Dim + 1> &a,
                                     num_type indices, std::size_t &j,
                                     Callable func) {
  if constexpr (helper_dim == Dim) {
    a[helper_dim] = utils::Power<2, Dim>::value - 1;
    func(a, j);
    j++;

  } else {

    constexpr num_type mask = utils::Power<2, Dim>::value - 1;

    for (; indices != 0; indices &= indices - 1) {
      auto least_significant_bit = indices & (~indices + 1);
      a[helper_dim] = a[helper_dim - 1] + least_significant_bit;
      num_type indices_new = (~a[helper_dim]) & mask;
      iterate_over_tets_helper<Dim, helper_dim + 1>(a, indices_new, j, func);
    }
  }
}

template <Dimension Dim, UnsignedIntegral num_type, typename Callable>
inline void iterate_over_tets(Callable func) {
  constexpr auto upper_limit = utils::Power<2, Dim>::value;
  // number of corners in tetrahedra: Dim+1
  blas::vector<num_type, Dim + 1> a{};
  // Last corner will always be 2^Dim-1
  constexpr num_type mask = utils::Power<2, Dim>::value - 1;

  for (std::size_t i = 1, j = 0; i < upper_limit; i *= 2) {
    a[1] = i;
    num_type indices = (~i) & mask;
    iterate_over_tets_helper<Dim, 2>(a, indices, j, func);
  }
}

#endif
