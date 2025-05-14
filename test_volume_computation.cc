#include "compute_volumes.h"
#include "cubes_cutter.h"
#include "cutting_tetrahedra.h"
#include "predefinitions.h"
#include "scientific_quantities.h"
#include <experimental/mdspan>
#include <iostream>
#include <stdio.h>

int main(int argc, char *argv[]) {

  cubes_cutter::Cutter cutter{};
  volume_computer::Volume_comp v_comp{};

  std::array<std::uint32_t, 14 * 19> cube_edges_values{};
  std::mdspan cube_edges(cube_edges_values.data(), 14, 19);

  std::array<DataType, 14 * 6> volume_tetrahedra_values{};
  std::mdspan volume_tetrahedra(volume_tetrahedra_values.data(), 14, 6);

  DataType radius = 20;
  Cutter_utils::vector3d center{0.f, 0.f, 0.f};
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
    printf("%2.0f %2.0f %2.0f\n", point[0], point[1], point[2]);
  }

  for (auto &point : points)
    point = point * (radius - norm(point) / 2.f);

  std::cout << "The points after scaling are: " << std::endl;
  for (auto point : points) {
    printf("%7.3f %7.3f %7.3f\n", point[0], point[1], point[2]);
  }

  // std::cout << "Hello World!" << std::endl;
}
