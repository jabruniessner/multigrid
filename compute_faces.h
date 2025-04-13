#pragma once

#include "blas.h"
#include "predefinitions.h"
#include "scientific_quantities.h"
#include <array>
#include <boost/container/static_vector.hpp>
#include <cstdint>
#include <span>

using vector3d = blas::vector<DataType, 3>;
using Face = boost::container::static_vector<vector3d, 3>;

constexpr DataType sqrt2inv = 1 / const_sqrt(2);
constexpr DataType sqrt3inv = 1 / const_sqrt(3);

template <template <typename T> typename Container>
void compute_faces(const std::uint8_t points, const DataType grid_step,
                   const vector3d &point, std::span<DataType, 19> lengths,
                   Container<Face> &faces) {

  std::uint8_t h = 0;
  for (std::uint8_t i = 1; i <= 4; i *= 2) {
    // Finding the other two points in the cube
    std::uint8_t other_point_left = (i >> 1 | i << (3 - 1) | i) & 7;
    std::uint8_t other_point_right = (i << 1 | i >> (3 - 1) | i) & 7;
    std::array<std::uint8_t, 2> second_points{other_point_left,
                                              other_point_right};

    // Iteration over the two other points (The two tetrahedra the point belongs
    // to)
    for (int l = 0; l < 2; l++) {

      boost::container::static_vector<vector3d, 4> tetrahedra_points;

      std::array<std::uint8_t, 4> points_tet{0, 7, i, second_points[l]};

      for (std::uint8_t j = 0; j < 3; j++) {
        bool inside_first = (points >> points_tet[j] & 1);

        for (std::uint8_t k = j + 1; k < 4; k++) {
          bool inside_second = (points >> points_tet[k] & 1);

          if (inside_second == inside_first)
            continue;

          std::uint8_t direction = std::abs(points_tet[k] - points_tet[j]);

          // Computing the prefactor in order to normlaize the direction vector
          auto inverse_dir = (~direction) & 7;

          auto prefac =
              (1 + (sqrt2inv - 1) * ((inverse_dir & (inverse_dir - 1)) == 0) +
               (sqrt3inv - sqrt2inv) * (direction == 7));

          auto direction_vec = vector3d{(direction & 1) * grid_step,
                                        ((direction >> 1) & 1) * grid_step,
                                        ((direction >> 2) & 1) * grid_step} *
                               (prefac * (1 - 2 * (points_tet[j] == 7)));

          auto point_setoff =
              point + vector3d{(points_tet[j] & 1) * grid_step,
                               ((points_tet[j] >> 1) & 1) * grid_step,
                               ((points_tet[j] >> 2) & 1) * grid_step};

          std::uint8_t length_index;

          // if (j == 0) {
          //   length_index = k;
          // } else if (j == 1) {
          //   length_index = 7 + k - 1;
          // } else {
          //   length_index = 13 + 2 * (h) + l;
          // }
          if (j == 0) {
            length_index = points_tet[k] % 7;
          } else if (j == 1) {
            length_index = 13 - points_tet[k];
          } else {
            length_index = 13 + 2 * (h) + l;
          }

          // std::uint8_t point_index = points_tet[length_index];

          direction_vec *= lengths[length_index];
          tetrahedra_points.push_back(point_setoff + direction_vec);
        }
      }

      if (tetrahedra_points.size() == 0)
        continue;

      // std::cout << "The number of points in the tetrahedra is: "
      //           << tetrahedra_points.size() << std::endl;

      // std::cout << "The points are: " << static_cast<int>(points) <<
      // std::endl;

      assert(tetrahedra_points.size() >= 3);

      if (tetrahedra_points.size() == 3) {
        Face face_points;
        face_points.assign(tetrahedra_points.begin(), tetrahedra_points.end());

        faces.push_back(face_points);

      } else {

        Face face_points1;
        face_points1.assign(tetrahedra_points.begin(),
                            tetrahedra_points.end() - 1);

        faces.push_back(face_points1);
        Face face_points2;
        face_points2.assign(tetrahedra_points.begin() + 1,
                            tetrahedra_points.end());
        faces.push_back(face_points2);
      }
    }

    h++;
  }
}
