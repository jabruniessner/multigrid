#include "Atom_types.h"
#include "Domain.h"
#include "blas.h"
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
#include <random>
#include <sycl/sycl.hpp>
#include <tuple>
#include <utility>

#ifndef CUBES_CUTTER_H
#define CUBES_CUTTER_H

namespace cubes_cutter {

struct Cutter {
  template <std::size_t Dim, std::size_t... side_lengths>
  void operator()(
      const DataType *sphere_position, const DataType sphere_radius,
      const DataType grid_step, const std::size_t position_0,
      const std::size_t position_1, const std::size_t position_2,
      const domain::Grid<std::uint32_t, Dim, side_lengths...> &inside_outside,
      const domain::Grid<std::uint32_t, Dim + 1, side_lengths..., 19> &tet_grid

  ) const {

    Cutter_utils::vector3d point{static_cast<DataType>(position_0),
                                 static_cast<DataType>(position_1),
                                 static_cast<DataType>(position_2)};
    Cutter_utils::vector3d center{sphere_position[0], sphere_position[1],
                                  sphere_position[2]};
    DataType radius = sphere_radius;
    std::uint32_t points = Cutter_utils::find_polygon_cuts(
        point, center, radius, grid_step,
        &tet_grid(position_0, position_1, position_2, 0));

    // tet_grid(position_0, position_1, position_2, 0)++;
    // syclx::printf("Hello World form inside lambda!"); // NOLINT
    // const std::size_t i = 342;
    // syclx::printf("%d %d %d", position_0, position_1, position_2); // NOLINT

    //  std::cout << position_0 << " " << position_1 << " " << position_2
    //            << std::endl; // NOLINT

    // tet_grid(position_0, position_1, position_2, 0)++; //);

    sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>
        atomic_inside_outside(
            inside_outside(position_0, position_1, position_2));

    atomic_inside_outside.fetch_or(points);
  }
};

template <Length length, typename Cutter, typename DataType,
          typename... Domains, std::size_t... DomainSize, Dimension Dim,
          typename... Positions, Length... directions>
void cutting_cubes_helper(
    const Cutter &cutter,
    std::tuple<
        Sphere<DataType, Dim> &, const DataType &,
        const domain::Grid<std::uint32_t, Dim, DomainSize...> &,
        const domain::Grid<std::uint32_t, Dim + 1, DomainSize..., length> &,
        Domains &...>
        arg_tuple,
    const std::index_sequence<directions...>, Positions... positions) {

  constexpr DataType sqrt_Dim = const_sqrt(static_cast<DataType>(Dim)) * 0.5;
  using vectorDimd = blas::vector<DataType, Dim>;

  static_assert(sizeof...(directions) <= Dim,
                "Break condition never satisfied");

  if constexpr (sizeof...(directions) == Dim - 1) {
    const auto &sphere = std::get<0>(arg_tuple);
    const auto &grid_step = std::get<1>(arg_tuple);
    const auto &Position = sphere.Position;
    const auto &current_value = Position[sizeof...(directions)];
    const auto radius_outer = sphere.radius + sqrt_Dim * grid_step;
    const auto radius_inner = std::max(sphere.radius - sqrt_Dim * grid_step,
                                       static_cast<DataType>(0.));

    auto sqrt_squared_outer =
        (blas::square(radius_outer) -
         (blas::square(Position[directions] - grid_step * (positions + .5)) +
          ...));

    if (sqrt_squared_outer <= 0) {
      sqrt_squared_outer = 0;
    }

    DataType sqrt_outer = std::sqrt(sqrt_squared_outer);

    const int lower_bound =
        std::ceil((current_value - sqrt_outer) / grid_step - .5);
    const int upper_bound =
        std::floor((current_value + sqrt_outer) / grid_step - .5);

    //  auto sqrt_squared_inner =
    //      (blas::square(radius_inner) -
    //       (blas::square(Position[directions] - grid_step * (positions + .5))
    //       +
    //        ...));

    //  if (sqrt_squared_inner <= 0) {
    for (int i = lower_bound; i <= upper_bound; i++) {

      // co_yield std::array<int, Dim>{static_cast<int>(positions)..., i};
      // Here come the necessary routines to cut the tetrahedra
      //
      //
      // constexpr auto i_seq = std::make_index_sequence<2 +
      // sizeof...(Domains)>{};
      constexpr auto i_seq_os =
          utils::make_index_sequence_with_offset<2, 2 + sizeof...(Domains)>();

      auto domains_tuple = utils::extract_tuple(arg_tuple, i_seq_os);

      auto sphere_position_pointer = sphere.Position.data();

      auto arg_tuple_inner1 = std::forward_as_tuple(
          sphere_position_pointer, sphere.radius, grid_step, positions..., i);

      auto arg_tuple_inner = std::tuple_cat(arg_tuple_inner1, domains_tuple);

      std::apply(cutter, arg_tuple_inner);
    }
    //    } else if (sqrt_squared_inner > 0) {
    //
    //      const auto sqrt_inner = std::sqrt(sqrt_squared_inner);
    //
    //      const int lower_bound_inner =
    //          std::floor((current_value - sqrt_inner) / grid_step - .5);
    //
    //      for (int i = lower_bound; i <= lower_bound_inner; i++) {
    //        // co_yield std::array<int, Dim>{static_cast<int>(positions)...,
    //        i};
    //        // Here come the necessary routines to cut the tetrahedra
    //        cutter(tet_grid, inside_outside,
    //               static_cast<const DataType *>(sphere.Position.data()),
    //               sphere.radius, grid_step, positions..., i);
    //      }
    //
    //      const int upper_bound_inner =
    //          std::ceil((current_value + sqrt_inner) / grid_step - .5);
    //
    //      for (int i = upper_bound_inner; i <= upper_bound; i++) {
    //        // co_yield std::array<int, Dim>{static_cast<int>(positions)...,
    //        i};
    //        // Here come the necessary routines to cut the tetrahedra
    //        cutter(tet_grid, inside_outside,
    //               static_cast<const DataType *>(sphere.Position.data()),
    //               sphere.radius, grid_step, positions..., i);
    //      }
    //    }

  } else if constexpr (sizeof...(directions) == 0) {

    auto &sphere = std::get<0>(arg_tuple);
    auto &grid_step = std::get<1>(arg_tuple);

    const int lower_bound = std::ceil(
        (sphere.Position[0] - (sphere.radius + sqrt_Dim * grid_step)) /
            grid_step -
        .5);
    const int upper_bound = std::floor(
        (sphere.Position[0] + (sphere.radius + sqrt_Dim * grid_step)) /
            grid_step -
        .5);

    for (int i = lower_bound; i <= upper_bound; ++i) {
      cutting_cubes_helper<length>(
          cutter, arg_tuple,
          std::make_index_sequence<sizeof...(directions) + 1>{}, i);
    }
  } else {

    auto &sphere = std::get<0>(arg_tuple);
    auto &grid_step = std::get<1>(arg_tuple);
    const auto &Position = sphere.Position;
    const auto &current_value = Position[sizeof...(directions)];
    const auto radius = sphere.radius + sqrt_Dim * grid_step;

    auto sqrt_squared =
        (blas::square(radius) -
         (blas::square(Position[directions] - grid_step * (positions + .5)) +
          ...));

    if (sqrt_squared < 0) {
      sqrt_squared = 0;
    }

    const auto sqrt = std::sqrt(sqrt_squared);

    const int lower_bound = std::ceil((current_value - sqrt) / grid_step - .5);
    const int upper_bound = std::floor((current_value + sqrt) / grid_step - .5);

    for (int i = lower_bound; i <= upper_bound; i++) {
      cutting_cubes_helper<length>(
          cutter, arg_tuple, // NOLINT
          std::make_index_sequence<sizeof...(directions) + 1>{}, positions...,
          i);
    }
  }
}

template <Length length, typename Cutter, typename DataType, Dimension Dim,
          typename... Domains, Length... DomainSize>
void cutting_cubes(
    const Cutter &cutter,
    std::tuple<
        Sphere<DataType, Dim> &, const DataType &,
        const domain::Grid<std::uint32_t, Dim, DomainSize...> &,
        const domain::Grid<std::uint32_t, Dim + 1, DomainSize..., length> &,
        Domains &...>
        arg_tuple) {
  cutting_cubes_helper<length>(cutter, arg_tuple,
                               std::make_index_sequence<0u>{});
}

} // namespace cubes_cutter

#endif
