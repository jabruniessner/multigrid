#include "compute_volumes.h"
#include "cubes_cutter.h"
#include "cutting_tetrahedra.h"
#include "predefinitions.h"
#include "scientific_quantities.h"
#include <cstddef>
#include <experimental/mdspan>
#include <iostream>
#include <ostream>

template <typename DataType> class TD;

constexpr int num_points = 34;
constexpr int num_stages = 10;

int main(int argc, char *argv[]) {

  cubes_cutter::Cutter cutter{};
  volume_computer::Volume_comp v_comp{};

  std::array<std::uint32_t, 19> cube_edges_values{};
  // std::mdspan cube_edges(cube_edges_values.data(), num_stages, num_points,
  // 19);

  std::array<DataType, 6> volume_tetrahedra_values{};
  // std::mdspan volume_tetrahedra(volume_tetrahedra_values.data(), num_stages,
  //                               num_points, 6);

  // TD<DataType> helllo;

  DataType radius = 10000;
  Cutter_utils::vector3d center1{(DataType)1., (DataType)1., (DataType)1.};
  center1 = center1 * ((10000 + std::sqrt(3.) / (6.)) / std::sqrt(3.));
  Cutter_utils::vector3d center2 = center1 * (-1);

  Cutter_utils::vector3d point_org{(DataType)-0.5, (DataType)-0.5,
                                   (DataType)-0.5};

  std::printf("The computed volumes are: \n");

  std::uint8_t point1 = Cutter_utils::find_polygon_cuts(
      point_org, center1, radius, (DataType)1.0, cube_edges_values.data());

  std::uint8_t point2 = Cutter_utils::find_polygon_cuts(
      point_org, center2, radius, (DataType)1.0, cube_edges_values.data());

  std::uint8_t point = point2 | point1;

  std::printf("The points are: %b\n", point);

  std::span<std::uint32_t, 19> tet_grid_span(cube_edges_values.data(), 19);
  std::span<DataType, 6> tetrahedra_span(volume_tetrahedra_values.data(), 6);
  std::span<DataType, 3> sphere_position(center1.data(), 3);
  DataType grid_step = 1.f;

  // They are not being used in this example
  std::size_t position_0 = static_cast<std::size_t>(point),
              position_1 = static_cast<std::size_t>(point),
              position_2 = static_cast<std::size_t>(point);

  std::printf("The Positions are: %8.3f %8.3f %8.3f\n", point_org[0],
              point_org[1], point_org[2]);

  v_comp(tet_grid_span, point, tetrahedra_span, sphere_position, radius,
         grid_step, position_0, position_1, position_2);

  std::printf("The intersection points are: \n");
  for (int k = 0; k < 19; k++)
    std::printf("%5.2f ", volume_computer::conv32(cube_edges_values[k]));
  std::printf("\n");

  for (int j = 0; j < 19; j++)
    std::printf("%5d ", j);
  std::printf("\n");

  DataType volume_all = 0;
  // std::printf("Volumes of single tet: ");
  for (auto vol : tetrahedra_span) {
    volume_all += vol;
    std::printf("%5.2f ", vol);
  }

  std::printf("\n");

  std::printf("%7.3f\n", volume_all);
}
