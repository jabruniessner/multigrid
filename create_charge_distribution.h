#include "Domain.h"
#include "predefinitions.h"
#include <array>
#include <iostream>
#include <iterator>
#include <tuple>

#ifndef CREATE_CHARGE_DISTRIBUTION_H
#define CREATE_CHARGE_DISTRIBUTION_H

// template <Dimension Dim, typename DataType> void convert_number_to_binary()
// {}

template <typename DataType, DataType value> struct spacing {};

template <Dimension Dim, typename DataType, DataType grid_spacing,
          std::size_t... Strides>
void add_charges_to_distribution(domain::Domain<Dim, Strides...> domain,
                                 std::array<DataType, Dim> &Point,
                                 spacing<DataType, grid_spacing>) {

  constexpr auto num_points = utils::Power<2, Dim>::value;
  constexpr auto Box_volume = utils::power_off(grid_spacing, Dim);
  std::array<DataType, 2 * Dim> distance_weight;
  std::array<std::size_t, 2 * Dim> Grid_Points;
  std::array<DataType, utils::power_off(2, Dim)> factors;

  for (int i = 0; i < Dim; i++) {
    const DataType Position_grid_spacing = Point[i] / grid_spacing;
    const std::size_t lower = static_cast<std::size_t>(Position_grid_spacing);

    Grid_Points[2 * i] = lower;
    Grid_Points[2 * i + 1] = lower + 1;

    const DataType distance_weight_lower = (lower + 1 - Position_grid_spacing);
    const DataType distance_weight_upper = (Position_grid_spacing - lower);

    distance_weight[2 * i] = distance_weight_lower;
    distance_weight[2 * i + 1] = distance_weight_upper;
  }

  std::array<std::size_t, Dim> Coordinate;
  for (char i = 0; i < num_points; i++) {

    DataType weight = 1;
    char value = 1;
    for (int j = 0; j < Dim; j++) {
      const char index = ((value & i) == value);
      weight *= distance_weight[2 * j + index];
      Coordinate[j] = Grid_Points[2 * j + index];
      value *= 2;
    }

    std::apply(
        [&, weight](auto... elems) { domain(elems...) = weight / Box_volume; },
        Coordinate);
  }
};

// template <Dimension Dim, typename DataType, std::size_t... Strides>
// void add_charge_to_charge_distribution(domain::Domain<Dim, Strides...>
// &domain,
//                                        std::array<DataType, Dim>
//                                        Point_Charge, DataType grid_spacing) {
//
//   std::array<std::size_t, 2 * Dim> Grid_Points;
//   std::array<DataType, utils::Power<2, Dim>::value> factors;
//
//   get_cube_arround_point_and_distances(Grid_Points, factors, Point_Charge,
//                                        grid_spacing);
// }

#endif // !CREATE_CHARGE_DISTRIBUTION_H
