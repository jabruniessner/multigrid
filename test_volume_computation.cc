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

int main(int argc, char *argv[]) {

  cubes_cutter::Cutter cutter{};
  volume_computer::Volume_comp v_comp{};



  // TD<DataType> helllo;

  DataType radius = 10000;
  Cutter_utils::vector3d center{20000., 20000., 20000.};


  int num = 6;
        for (DataType k = -1; k <= 1; k += 2)
          points_span[stage, num++] = {i, j, k};

  for (int stage = 0; stage < num_stages; stage++)
    for (DataType i = -1; i <= 1; i += 2)
      for (DataType j = -1; j <= 1; j += 2)
        for (DataType k = -1; k <= 1; k += 2)
          points_span[stage, num++] = {i, j, k};


  //  }

  for (int i = 6; i < 14; i++) {
    auto &point = points[i];
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



  // std::cout << "Hello World!" << std::endl;
}
