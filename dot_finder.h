#include "Atom_types.h"
#include "Domain.h"
#include "blas.h"
#include "predefinitions.h"
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
        (square(radius) -
         (square(Position[directions] - grid_step * positions) + ...));

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
std::generator<std::array<int, Dim>> line_sphere_intersections_helper(
    Sphere<DataType, Dim> &sphere, DataType grid_step,
    std::index_sequence<directions...>, Positions... positions) {

  static_assert(sizeof...(directions) <= Dim,
                "Break condition never satisfied");

  if constexpr (sizeof...(directions) == Dim - 1) {

    const auto &Position = sphere.Position;
    const auto &current_value = Position[sizeof...(directions)];
    const auto &radius = sphere.radius;

    auto sqrt_squared =
        (square(radius) -
         (square(Position[directions] - grid_step * positions) + ...));

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
      co_yield std::ranges::elements_of(line_sphere_intersections_helper(
          sphere, grid_step,
          std::make_index_sequence<sizeof...(directions) + 1>{}, i));
    }

  } else {
    const auto &Position = sphere.Position;
    const auto &current_value = Position[sizeof...(directions)];
    const auto &radius = sphere.radius;

    auto sqrt_squared =
        (square(radius) -
         (square(Position[directions] - grid_step * positions) + ...));

    if (sqrt_squared < 0) {
      sqrt_squared = 0;
    }

    const auto sqrt = std::sqrt(sqrt_squared);

    const int lower_bound =
        static_cast<int>(std::ceil((current_value - sqrt) / grid_step));

    const int upper_bound =
        static_cast<int>((current_value + sqrt) / grid_step);

    for (int i = lower_bound; i <= upper_bound; i++) {
      co_yield std::ranges::elements_of(line_sphere_intersections_helper(
          sphere, grid_step,
          std::make_index_sequence<sizeof...(directions) + 1>{}, positions...,
          i));
    }
  }
}

template <typename DataType, Dimension Dim>
std::generator<std::array<int, Dim>>
line_sphere_intersection(Sphere<DataType, Dim> &sphere,
                         const DataType grid_step) {
  co_yield std::ranges::elements_of(line_sphere_intersections_helper(
      sphere, grid_step, std::make_index_sequence<0u>{}));
}

#endif
