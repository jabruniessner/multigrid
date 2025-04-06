#include "Atom_types.h"
#include "Domain.h"
#include "blas.h"
#include "predefinitions.h"
#include "scientific_quantities.h"
#include "utils.h"
#include <array>
#include <bits/elements_of.h>
#include <cstddef>
#include <generator>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <tuple>
#include <utility>

#ifndef DOT_FINDER_H
#define DOT_FINDER_H

// template <typename T> inline T square(T x) { return x * x; }

template <typename DataType, Dimension Dim, Length... strides,
          typename... Positions, Length... directions>
void find_dots_in_sphere_helper(Sphere<DataType, Dim> &sphere,
                                domain::Domain<Dim, strides...> &domain,
                                DataType grid_step,
                                std::index_sequence<directions...>,
                                Positions... positions) {

  static_assert(sizeof...(directions) <= Dim,
                "Break condition never satisfied");

  if constexpr (sizeof...(directions) == Dim) {
    auto a = std::make_tuple(positions...);
    domain(positions...) = 1;

  } else if constexpr (sizeof...(directions) == 0) {

    const int lower_bound = static_cast<int>(
        std::ceil((sphere.Position[0] - sphere.radius) / grid_step));
    const int upper_bound =
        static_cast<int>((sphere.Position[0] + sphere.radius) / grid_step);

    for (int i = lower_bound; i <= upper_bound; ++i) {
      find_dots_in_sphere_helper(
          sphere, domain, grid_step,
          std::make_index_sequence<sizeof...(directions) + 1>{}, i);
    }

  } else {
    const auto &Position = sphere.Position;
    const auto &current_value = Position[sizeof...(directions)];
    const auto &radius = sphere.radius;

    auto sqrt_squared =
        (blas::square(radius) -
         (blas::square(Position[directions] - grid_step * positions) + ...));

    if (sqrt_squared < 0) {
      sqrt_squared = 0;
    }

    const auto sqrt = std::sqrt(sqrt_squared);

    const int lower_bound =
        static_cast<int>(std::ceil((current_value - sqrt) / grid_step));

    const int upper_bound =
        static_cast<int>((current_value + sqrt) / grid_step);

    for (int i = lower_bound; i <= upper_bound; i++) {
      find_dots_in_sphere_helper(
          sphere, domain, grid_step,
          std::make_index_sequence<sizeof...(directions) + 1>{}, positions...,
          i);
    }
  }
}

template <typename DataType, Dimension Dim, Length... strides>
void find_dots_in_sphere(Sphere<DataType, Dim> &sphere,
                         domain::Domain<Dim, strides...> domain,
                         const DataType grid_step) {
  find_dots_in_sphere_helper(sphere, domain, grid_step,
                             std::make_index_sequence<0u>{});
}

template <std::size_t... numbers>
void print_index_sequence(std::index_sequence<numbers...>) {
  ((std::cout << numbers << " "), ...);
  std::cout << std::endl;
}

template <typename DataType, Dimension Dim, typename... Positions,
          Length... directions>
std::generator<std::array<int, Dim>>
finding_edge_points_helper(Sphere<DataType, Dim> &sphere, DataType grid_step,
                           std::index_sequence<directions...>,
                           Positions... positions) {

  static_assert(sizeof...(directions) <= Dim,
                "Break condition never satisfied");

  if constexpr (sizeof...(directions) == Dim - 1) {

    const auto &Position = sphere.Position;
    const auto &current_value = Position[sizeof...(directions)];
    const auto &radius = sphere.radius;

    auto sqrt_squared =
        (blas::square(radius) -
         (blas::square(Position[directions] - grid_step * positions) + ...));

    if (sqrt_squared < 0) {
      sqrt_squared = 0;
    }

    const auto sqrt = std::sqrt(sqrt_squared);

    const int lower_bound =
        static_cast<int>(std::ceil((current_value - sqrt) / grid_step));

    co_yield std::array<int, Dim>{{(positions)..., lower_bound}};

    const int upper_bound =
        static_cast<int>((current_value + sqrt) / grid_step);

    co_yield std::array<int, Dim>{static_cast<int>(positions)..., upper_bound};

  } else if constexpr (sizeof...(directions) == 0) {

    const int lower_bound = static_cast<int>(
        std::ceil((sphere.Position[0] - sphere.radius) / grid_step));
    const int upper_bound =
        static_cast<int>((sphere.Position[0] + sphere.radius) / grid_step);

    for (int i = lower_bound; i <= upper_bound; ++i) {
      co_yield std::ranges::elements_of(finding_edge_points_helper(
          sphere, grid_step,
          std::make_index_sequence<sizeof...(directions) + 1>{}, i));
    }

  } else {
    const auto &Position = sphere.Position;
    const auto &current_value = Position[sizeof...(directions)];
    const auto &radius = sphere.radius;

    auto sqrt_squared =
        (blas::square(radius) -
         (blas::square(Position[directions] - grid_step * positions) + ...));

    if (sqrt_squared < 0) {
      sqrt_squared = 0;
    }

    const auto sqrt = std::sqrt(sqrt_squared);

    const int lower_bound =
        static_cast<int>(std::ceil((current_value - sqrt) / grid_step));

    const int upper_bound =
        static_cast<int>((current_value + sqrt) / grid_step);

    for (int i = lower_bound; i <= upper_bound; i++) {
      co_yield std::ranges::elements_of(finding_edge_points_helper(
          sphere, grid_step,
          std::make_index_sequence<sizeof...(directions) + 1>{}, positions...,
          i));
    }
  }
}

template <typename DataType, Dimension Dim>
std::generator<std::array<int, Dim>>
finding_edge_points(Sphere<DataType, Dim> &sphere, const DataType grid_step) {
  co_yield std::ranges::elements_of(finding_edge_points_helper(
      sphere, grid_step, std::make_index_sequence<0u>{}));
}

template <typename DataType, Dimension Dim, typename... Positions,
          Length... directions>
std::generator<std::array<int, Dim>>
finding_edge_cubes_helper(Sphere<DataType, Dim> &sphere, DataType grid_step,
                          std::index_sequence<directions...>,
                          Positions... positions) {

  constexpr DataType sqrt_Dim = const_sqrt(static_cast<DataType>(Dim)) * 0.5;

  static_assert(sizeof...(directions) <= Dim,
                "Break condition never satisfied");

  if constexpr (sizeof...(directions) == Dim - 1) {

    const auto &Position = sphere.Position;
    const auto &current_value = Position[sizeof...(directions)];
    const auto radius_outer = sphere.radius + sqrt_Dim * grid_step;
    const auto radius_inner =
        std::max(sphere.radius - sqrt_Dim * grid_step, 0.f);

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

    auto sqrt_squared_inner =
        (blas::square(radius_inner) -
         (blas::square(Position[directions] - grid_step * (positions + .5)) +
          ...));

    if (sqrt_squared_inner <= 0) {
      for (int i = lower_bound; i <= upper_bound; i++) {
        co_yield std::array<int, Dim>{static_cast<int>(positions)..., i};
      }
    } else if (sqrt_squared_inner > 0) {

      const auto sqrt_inner = std::sqrt(sqrt_squared_inner);

      const int lower_bound_inner =
          std::floor((current_value - sqrt_inner) / grid_step - .5);

      for (int i = lower_bound; i <= lower_bound_inner; i++) {
        co_yield std::array<int, Dim>{static_cast<int>(positions)..., i};
      }

      const int upper_bound_inner =
          std::ceil((current_value + sqrt_inner) / grid_step - .5);

      for (int i = upper_bound_inner; i <= upper_bound; i++) {
        co_yield std::array<int, Dim>{static_cast<int>(positions)..., i};
      }
    }

  } else if constexpr (sizeof...(directions) == 0) {

    const int lower_bound = std::ceil(
        (sphere.Position[0] - (sphere.radius + sqrt_Dim * grid_step)) /
            grid_step -
        .5);
    const int upper_bound = std::floor(
        (sphere.Position[0] + (sphere.radius + sqrt_Dim * grid_step)) /
            grid_step -
        .5);

    for (int i = lower_bound; i <= upper_bound; ++i) {
      co_yield std::ranges::elements_of(finding_edge_cubes_helper(
          sphere, grid_step,
          std::make_index_sequence<sizeof...(directions) + 1>{}, i));
    }
  } else {
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
      co_yield std::ranges::elements_of(finding_edge_cubes_helper(
          sphere, grid_step,
          std::make_index_sequence<sizeof...(directions) + 1>{}, positions...,
          i));
    }
  }
}

template <typename DataType, Dimension Dim>
std::generator<std::array<int, Dim>>
finding_edge_cubes(Sphere<DataType, Dim> &sphere, const DataType grid_step) {
  co_yield std::ranges::elements_of(finding_edge_cubes_helper(
      sphere, grid_step, std::make_index_sequence<0u>{}));
}

template <typename DataType, Dimension Dim, typename... Positions,
          Length... directions>
void cutting_cubes_helper(Sphere<DataType, Dim> &sphere, DataType grid_step,
                          std::index_sequence<directions...>,
                          Positions... positions) {

  constexpr DataType sqrt_Dim = const_sqrt(static_cast<DataType>(Dim)) * 0.5;

  static_assert(sizeof...(directions) <= Dim,
                "Break condition never satisfied");

  if constexpr (sizeof...(directions) == Dim - 1) {

    const auto &Position = sphere.Position;
    const auto &current_value = Position[sizeof...(directions)];
    const auto radius_outer = sphere.radius + sqrt_Dim * grid_step;
    const auto radius_inner =
        std::max(sphere.radius - sqrt_Dim * grid_step, 0.f);

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

    auto sqrt_squared_inner =
        (blas::square(radius_inner) -
         (blas::square(Position[directions] - grid_step * (positions + .5)) +
          ...));

    if (sqrt_squared_inner <= 0) {
      for (int i = lower_bound; i <= upper_bound; i++) {

        // co_yield std::array<int, Dim>{static_cast<int>(positions)..., i};
        // Here come the necessary routines to cut the tetrahedra
      }
    } else if (sqrt_squared_inner > 0) {

      const auto sqrt_inner = std::sqrt(sqrt_squared_inner);

      const int lower_bound_inner =
          std::floor((current_value - sqrt_inner) / grid_step - .5);

      for (int i = lower_bound; i <= lower_bound_inner; i++) {
        // co_yield std::array<int, Dim>{static_cast<int>(positions)..., i};
        // Here come the necessary routines to cut the tetrahedra
      }

      const int upper_bound_inner =
          std::ceil((current_value + sqrt_inner) / grid_step - .5);

      for (int i = upper_bound_inner; i <= upper_bound; i++) {
        // co_yield std::array<int, Dim>{static_cast<int>(positions)..., i};
        // Here come the necessary routines to cut the tetrahedra
      }
    }

  } else if constexpr (sizeof...(directions) == 0) {

    const int lower_bound = std::ceil(
        (sphere.Position[0] - (sphere.radius + sqrt_Dim * grid_step)) /
            grid_step -
        .5);
    const int upper_bound = std::floor(
        (sphere.Position[0] + (sphere.radius + sqrt_Dim * grid_step)) /
            grid_step -
        .5);

    for (int i = lower_bound; i <= upper_bound; ++i) {
      co_yield std::ranges::elements_of(cutting_cubes_helper(
          sphere, grid_step,
          std::make_index_sequence<sizeof...(directions) + 1>{}, i));
    }
  } else {
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
      co_yield std::ranges::elements_of(cutting_cubes_helper(
          sphere, grid_step,
          std::make_index_sequence<sizeof...(directions) + 1>{}, positions...,
          i));
    }
  }
}

template <typename DataType, Dimension Dim>
std::generator<std::array<int, Dim>>
cutting_cubes(Sphere<DataType, Dim> &sphere, const DataType grid_step) {
  co_yield std::ranges::elements_of(
      cutting_cubes_helper(sphere, grid_step, std::make_index_sequence<0u>{}));
}

#endif
