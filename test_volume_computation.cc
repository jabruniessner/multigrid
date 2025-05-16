#include "compute_volumes.h"
#include "cubes_cutter.h"
#include "cutting_tetrahedra.h"
#include "predefinitions.h"
#include "scientific_quantities.h"
#include <cstddef>
#include <experimental/mdspan>
#include <iostream>
#include <ostream>

int main(int argc, char *argv[]) {

  cubes_cutter::Cutter cutter{};
  volume_computer::Volume_comp v_comp{};

  std::array<std::uint32_t, 14 * 19> cube_edges_values{};
  std::mdspan cube_edges(cube_edges_values.data(), 14, 19);

  std::array<DataType, 14 * 6> volume_tetrahedra_values{};
  std::mdspan volume_tetrahedra(volume_tetrahedra_values.data(), 14, 6);

  DataType radius = 1000;
  Cutter_utils::vector3d center{2000.f, 2000.f, 2000.f};
  std::array<Cutter_utils::vector3d, 14> points{};

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      points[2 * i + j][i] = 1.f - 2 * j;

  int num = 6;
  for (DataType i = -1; i <= 1; i += 2)
    for (DataType j = -1; j <= 1; j += 2)
      for (DataType k = -1; k <= 1; k += 2)
        points[num++] = {i, j, k};

  for (auto point : points) {
    std::printf("%2.0f %2.0f %2.0f\n", point[0], point[1], point[2]);
  }

  for (auto &point : points)
    point = point * (radius / norm(point) - norm(point) / 2.f) + center -
            volume_computer::vector3d{
                static_cast<DataType>(std::signbit(point[0])),
                static_cast<DataType>(std::signbit(point[1])),
                static_cast<DataType>(std::signbit(point[2]))};

  std::cout << "The points after scaling are: " << std::endl;
  for (auto point : points) {
    std::printf("%7.3f %7.3f %7.3f\n", point[0], point[1], point[2]);
  }

  std::printf("The computed volumes are: \n");
  for (int i = 0; i < 14; i++) {
    // for (int i = 1; i < 2; i++) {
    std::uint8_t point = Cutter_utils::find_polygon_cuts(
        points[i], center, radius, 1.0f, &cube_edges[i, 0]);

    std::printf("The points are: %b\n", point);

    std::span<std::uint32_t, 19> tet_grid_span(&cube_edges[i, 0], 19);
    std::span<DataType, 6> tetrahedra_span(&volume_tetrahedra[i, 0], 6);
    std::span<DataType, 3> sphere_position(center.data(), 3);
    DataType grid_step = 1.f;

    // They are not being used in this example
    std::size_t position_0 = static_cast<std::size_t>(points[i][0]),
                position_1 = static_cast<std::size_t>(points[i][1]),
                position_2 = static_cast<std::size_t>(points[i][2]);

    std::printf("The Positions are: %3lu %3lu %3lu\n", position_0, position_1,
                position_2);

    v_comp(tet_grid_span, point, tetrahedra_span, sphere_position, radius,
           grid_step, position_0, position_1, position_2);

    std::printf("The intersection points are: \n");
    for (int j = 0; j < 19; j++)
      std::printf("%5.2f ", volume_computer::conv32(cube_edges[i, j]));
    std::printf("\n");

    for (int j = 0; j < 19; j++)
      std::printf("%5d ", j);
    std::printf("\n");

    DataType volume_all = 0;
    std::printf("Volumes of single tet: ");
    for (auto vol : tetrahedra_span) {
      volume_all += vol;
      std::printf("%5.2f ", vol);
    }

    std::printf("\n");

    std::printf("%2d: %7.3f\n", i, volume_all);
  }

  // std::cout << "Hello World!" << std::endl;
}
