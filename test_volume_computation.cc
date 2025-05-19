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

  std::array<std::uint32_t, num_stages * num_points * 19> cube_edges_values{};
  std::mdspan cube_edges(cube_edges_values.data(), num_stages, num_points, 19);

  std::array<DataType, num_stages * num_points * 6> volume_tetrahedra_values{};
  std::mdspan volume_tetrahedra(volume_tetrahedra_values.data(), num_stages,
                                num_points, 6);

  // TD<DataType> helllo;

  DataType radius = 10000;
  Cutter_utils::vector3d center{20000., 20000., 20000.};
  std::array<Cutter_utils::vector3d, num_stages * num_points> points{};
  std::mdspan points_span(points.data(), num_stages, num_points);

  for (int stage = 0; stage < num_stages; stage++)
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 2; j++)
        points_span[stage, 2 * i + j][i] = 1.f - 2 * j;

  int num = 6;
  for (int stage = 0; stage < num_stages; stage++)
    for (DataType i = -1; i <= 1; i += 2)
      for (DataType j = -1; j <= 1; j += 2)
        for (DataType k = -1; k <= 1; k += 2)
          points_span[stage, num++] = {i, j, k};

  for (int stage = 0; stage < num_stages; stage++)
    for (DataType i = -1; i <= 1; i += 2)
      for (DataType j = -1; j <= 1; j += 2)
        for (DataType k = -1; k <= 1; k += 2)
          points_span[stage, num++] = {i, j, k};

  //  for (auto point : points) {
  //    std::printf("%2.0f %2.0f %2.0f\n", point[0], point[1], point[2]);
  //  }
  for (int stage = 0; stage < num_stages; stage++)
    for (int i = 0; i < 6; i++) {
      auto &point = points_span[stage, i];
      point = point * ((radius + stage / (num_stages * 2.0)) / norm(point)) +
              center - volume_computer::vector3d{0.5, 0.5, 0.5};
    }

  for (int stage = 0; stage < num_stages; stage++)
    for (int i = 6; i < 14; i++) {
      auto &point = points_span[stage, i];
      point = point * (-1) * ((radius - 2 / std::sqrt(3)) / norm(point)) +
              center -
              volume_computer::vector3d{
                  static_cast<DataType>(std::signbit(-point[0])),
                  static_cast<DataType>(std::signbit(-point[1])),
                  static_cast<DataType>(std::signbit(-point[2]))};
    }

  for (int i = 14; i < 22; i++) {
    auto &point = points[i];
    point = point * (-1) * ((radius - 1 / std::sqrt(3)) / norm(point)) +
            center -
            volume_computer::vector3d{
                static_cast<DataType>(std::signbit(-point[0])),
                static_cast<DataType>(std::signbit(-point[1])),
                static_cast<DataType>(std::signbit(-point[2]))};
  }

  int cube_number = 22;
  for (int i = 0; i < 3; i++)
    for (int j = -1; j < 2; j += 2)
      for (int k = -1; k < 2; k += 2) {
        auto &point = points[cube_number++];
        point[i] = 0;
        point[(i + 1) % 3] = j;
        point[(i + 2) % 3] = k;

        point = point * (radius + norm(point)) -
                volume_computer::vector3d{0.5, 0.5, 0.5} + center;
      }

  //  for (auto &point : points)
  //    //  point = point * ((radius + 1 / std::sqrt(3)) / norm(point)) +
  //    center; point = point * (radius / norm(point) - norm(point) / 2.f) +
  //    center -
  //            volume_computer::vector3d{
  //                static_cast<DataType>(std::signbit(point[0])),
  //                static_cast<DataType>(std::signbit(point[1])),
  //                static_cast<DataType>(std::signbit(point[2]))}; //*
  //  // norm(point);

  std::cout << "The points after scaling are: " << std::endl;
  for (auto point : points) {
    std::printf("%7.3f %7.3f %7.3f\n", point[0], point[1], point[2]);
  }

  std::printf("The computed volumes are: \n");
  for (int i = 0; i < 34; i++)
    for (int j = 0; j < 10; j++) {
      // for (int i = 1; i < 2; i++) {
      std::uint8_t point = Cutter_utils::find_polygon_cuts(
          points[i], center, radius, 1.0, &cube_edges[j, i, 0]);

      std::printf("The points are: %b\n", point);

      std::span<std::uint32_t, 19> tet_grid_span(&cube_edges[j, i, 0], 19);
      std::span<DataType, 6> tetrahedra_span(&volume_tetrahedra[j, i, 0], 6);
      std::span<DataType, 3> sphere_position(center.data(), 3);
      DataType grid_step = 1.f;

      // They are not being used in this example
      std::size_t position_0 = static_cast<std::size_t>(points[i][0]),
                  position_1 = static_cast<std::size_t>(points[i][1]),
                  position_2 = static_cast<std::size_t>(points[i][2]);

      std::printf("The Positions are: %8.3f %8.3f %8.3f\n", points[i][0],
                  points[i][1], points[i][2]);

      v_comp(tet_grid_span, point, tetrahedra_span, sphere_position, radius,
             grid_step, position_0, position_1, position_2);

      std::printf("The intersection points are: \n");
      for (int k = 0; k < 19; k++)
        std::printf("%5.2f ", volume_computer::conv32(cube_edges[j, i, k]));
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
