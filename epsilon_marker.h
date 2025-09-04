#include "Domain.h"
#include "concepts.h"
#include <iostream>
#include <sycl/sycl.hpp>

#ifndef EPSILON_MARKER_H
#define EPSILON_MARKER_H

template <std::size_t Dim, UnsignedIntegral num_type,
          UnsignedIntegral num_type_2, UnsignedIntegral num_type_3,
          Length... strides_all, typename... Positions>
void mark_epsilons_helper(
    std::array<num_type, Dim + 1> a, std::size_t j,
    domain::Grid<num_type_2, Dim, strides_all...> io_domain,
    domain::Grid<num_type_3, Dim + 1, strides_all..., utils::factorial(Dim) - 2>
        values,
    Positions... pos) {

  static_assert(sizeof...(strides_all) == Dim);

  num_type_3 cell_inside = 0;
  for (auto i : a) {
    auto vector =
        bitshift::convert_byte_to_vec<std::size_t, decltype(i), Dim>(i);
    vector = vector + decltype(vector){pos...};
    using vec_base = std::array<std::size_t, Dim>;

    if (std::apply(io_domain, (vec_base)vector) == 1) {
      values(pos..., j) = static_cast<num_type_3>(1);
      return;
    }
  }
}

template <std::size_t Dim, UnsignedIntegral num_type_2,
          UnsignedIntegral num_type_3, Length... strides_all,
          std::size_t... dirs>
void mark_epsilons_helper(
    domain::Grid<num_type_2, Dim, strides_all...> io_domain,
    domain::Grid<num_type_3, Dim + 1, strides_all..., utils::factorial(Dim) - 2>
        values,
    std::index_sequence<dirs...>) {

  auto &length = io_domain.length;
  io_domain.q.parallel_for(
      sycl::range<Dim>((std::get<dirs>(length) + 1)...), [=](sycl::id<Dim> I) {
        auto f = [&](std::array<std::uint8_t, Dim + 1> a, std::size_t j) {
          mark_epsilons_helper<Dim, std::uint8_t, num_type_2, num_type_3>(
              a, j, io_domain, values, I[dirs]...);
        };

        iterate_over_tets<Dim, std::uint8_t>(f);
      });
}

template <std::size_t Dim, UnsignedIntegral num_type_2,
          UnsignedIntegral num_type_3, Length... strides_all>
void mark_epsilons(
    domain::Grid<num_type_2, Dim, strides_all...> io_domain,
    domain::Grid<num_type_3, Dim + 1, strides_all..., utils::factorial(Dim) - 2>
        values) {
  mark_epsilons_helper<Dim, num_type_2, num_type_3>(
      io_domain, values, std::make_index_sequence<Dim>{});
}

#endif
