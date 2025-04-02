#include "Atom_types.h"
#include "Domain.h"
#include "blas.h"
#include "dot_finder.h"
#include "hipSYCL/sycl/device_selector.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include "scientific_quantities.h"
#include "tetraeda_type.h"
#include <array>
#include <boost/container/static_vector.hpp>
#include <cblas.h>
#include <fstream>
#include <iostream>
#include <list>
#include <sycl/sycl.hpp>
#include <utility>

constexpr std::size_t Dim = 3;

constexpr DataType sqrt3inv = 1 / const_sqrt(3.);
constexpr DataType sqrt2inv = 1 / const_sqrt(2.);

using vector3d = blas::vector<DataType, Dim>;
using Position3D = std::array<int, Dim>;

using Face = boost::container::static_vector<vector3d, 3>;

inline constexpr DataType norm(vector3d vec) {
  DataType n = static_cast<DataType>(0);
  for (auto i : vec)
    n += i * i;

  return std::sqrt(n);
}

template <Dimension Dim, typename DataType>
constexpr inline DataType compute_interesect_for_no_princ(
    std::uint8_t i, std::uint8_t distance_num, DataType &grid_step,
    blas::vector<DataType, Dim> &point, blas::vector<DataType, Dim> &center,
    DataType &radius) {

  // std::uint8_t distance_num =
  //     static_cast<std::uint8_t>(((i >> 1 | i << (3 - 1)) & 7));

  vector3d distance{(distance_num & 1) * grid_step,
                    ((distance_num >> 1) & 1) * grid_step,
                    ((distance_num >> 2) & 1) * grid_step};

  vector3d point_setoff =
      point + vector3d{(i & 1) * grid_step, ((i >> 1) & 1) * grid_step,
                       ((i >> 2) & 1) * grid_step};

  return find_intersection_point_sphere<Dim>(point_setoff, distance, center,
                                             radius);
}

// std::countr_zero
// std::countl_zero

// example of cyclic bit shift
//   static_cast<std::uint8_t>(((i << 1 | i >> (3 - 1)) & ~248) | i),
//   static_cast<std::uint8_t>(((i >> 1 | i << (3 - 1)) & ~248) | i)};

inline constexpr bool in_sphere(const vector3d &point, const vector3d &center,
                                DataType &radius) {
  return norm(point - center) <= radius;
}

void find_polygon_cuts(vector3d &point, vector3d &center, DataType &radius,
                       DataType grid_step, std::list<Face> &faces) {
  // Iteration over all cubes

  std::array<DataType, 19> lengths{};
  std::uint8_t points = 0;
  for (std::uint8_t i = 0; i < 8; i++) {
    const vector3d neighbour_point =
        point + vector3d{(i & 1) * grid_step, ((i >> 1) & 1) * grid_step,
                         ((i >> 2) & 1) * grid_step};

    vector3d a{(i & 1) * grid_step, ((i >> 1) & 1) * grid_step,
               ((i >> 2) & 1) * grid_step};

    if (in_sphere(neighbour_point, center, radius)) {
      points |= (1 << i);
    }
  }

  // std::cout << "The points inside are: " << static_cast<int>(points)
  //           << std::endl;

  // std::cout << "The points inside are: " << static_cast<int>(points)
  //           << std::endl;

  if (points == 0 || points == 255)
    return;

  // Now we know which of the points are in inside the sphere
  // Now iterating over all

  // Iterating over all edges connected to 0;
  {
    bool zero_in_sphere = (bool)(points & 1);
    for (std::uint8_t i = 1; i <= 7; i++) {
      if (static_cast<bool>((points >> i) & 1) == zero_in_sphere)
        continue;

      vector3d distance{(i & 1) * grid_step, ((i >> 1) & 1) * grid_step,
                        ((i >> 2) & 1) * grid_step};
      distance /= norm(distance);

      DataType isec_p =
          find_intersection_point_sphere<Dim>(point, distance, center, radius);

      lengths[i % 7] = isec_p;
    };
  }
  // Iterating over all edges connected to 7;
  {
    bool seven_in_sphere = (bool)((points >> 7) & 1);
    for (std::uint8_t i = 1; i < 7; i++) {
      std::uint8_t point_num = 7 - i;
      // Checking whether the other point is on the other side of the surface
      if (static_cast<bool>((points >> point_num) & 1) == seven_in_sphere)
        continue;

      vector3d distance{(i & 1) * (-grid_step), ((i >> 1) & 1) * (-grid_step),
                        ((i >> 2) & 1) * (-grid_step)};

      distance /= norm(distance);

      auto point7 = point + vector3d{grid_step, grid_step, grid_step};

      DataType isec_p =
          find_intersection_point_sphere<Dim>(point7, distance, center, radius);

      lengths[7 + i - 1] = isec_p;
    };
  }

  // Iterating over all edges that are connected to each other
  {
    int edge_number = 13;
    for (std::uint8_t i = 1; i <= 4; i *= 2) {

      bool this_in_sphere = (bool)((points >> i) & 1);
      std::uint8_t other_point_1 =
          static_cast<std::uint8_t>(((i >> 1 | i << (3 - 1)) & 7) | i);

      bool other_in_sphere = (bool)((points >> other_point_1) & 1);

      std::uint8_t dir = other_point_1 - i;

      if (other_in_sphere != this_in_sphere) {
        lengths[edge_number] = compute_interesect_for_no_princ(
            i, dir, grid_step, point, center, radius);
      }

      edge_number++;

      std::uint8_t other_point_2 =
          static_cast<std::uint8_t>(((i << 1 | i >> (3 - 1)) & 7) | i);

      other_in_sphere = (bool)((points >> other_point_2) & 1);

      dir = other_point_2 - i;

      if (other_in_sphere != this_in_sphere) {
        lengths[edge_number] = compute_interesect_for_no_princ(
            i, dir, grid_step, point, center, radius);
      }

      edge_number++;
    }
  }

  //  for (DataType num : lengths)
  //    std::cout << std::format("{:<1.1f} ", num) << " ";
  //
  //  std::cout << std::endl;
  //
  //  for (int i = 0; i < 19; i++)
  //    std::cout << std::format("{:<3} ", i) << " ";
  //
  //  std::cout << std::endl;

  // Now I need to iterate over all tetrahedra in order to find the right
  // surface

  // Iterating over the tetrahedra

  // Here we are iterating over all lines in the cube that are connected to
  // point 0
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
  // Now iterating over all the tetrahedra and computing the faces

  // std::cout << "The number of faces is: " << faces.size() << std::endl;
}

void print_faces_to_ply(std::ostream &stream, std::list<Face> &faces) {
  stream << "ply\n";
  stream << "format ascii 1.0\n";
  stream << "element vertex " << faces.size() * 3 << "\n";
  stream << "property float x\n";
  stream << "property float y\n";
  stream << "property float z\n";
  stream << "element face " << faces.size() << "\n";
  stream << "property list uchar int vertex_index\n";
  stream << "end_header\n";

  for (auto face : faces) {
    for (auto point : face) {
      stream << point[0] << " " << point[1] << " " << point[2] << "\n";
    }
  }

  int index = 0;
  for (int i = 0; i < faces.size(); i++) {
    stream << "3 " << index << " " << index + 1 << " " << index + 2 << "\n";
    index += 3;
  }

  stream << std::endl;
}

int main(int argc, char *argv[]) {

  // #ifdef DEBUGMODE
  //   sycl::cpu_selector selector;
  // #else
  //   sycl::gpu_selector selector;
  // #endif
  //
  // Using host device for prove of concept.

  sycl::queue q{sycl::host_selector{},
                sycl::property_list{sycl::property::queue::in_order{}}};

  tetraeda_line_points<3> tet_example;

  Atom<DataType> atom_host;
  atom_host.Position[0] = 50;
  atom_host.Position[1] = 50;
  atom_host.Position[2] = 50;
  atom_host.radius = 1.6f;

  // Atom<DataType> *atom_device = sycl::malloc_host<Atom<DataType>>(1, q);
  // q.memcpy(atom_device, &atom_ost, sizeof(Atom<DataType>)).wait();
  domain::Grid<tetraeda_line_points<3>, 3, 100, 100, 100> tet_grid(
      Paddings::PERIODIC, q, 1);
  // std::cout << "Hello World!" << std::endl;
  // using vector3d = vector<DataType, 3>;
  // vector3d footpoint(0.25f, 0.f, 0.f), center(0.f, 0.f, 0.f),
  //    direction(1.f, 0.f, 0.f);
  // DataType isec_p =
  //    find_intersection_point_sphere<3>(footpoint, direction, center, 1.f);
  // std::cout << "The intersection distance is " << isec_p << std::endl;
  //

  std::list<Face> faces;

  vector3d center{atom_host.Position[0], atom_host.Position[1],
                  atom_host.Position[2]};

  // constexpr DataType grid_step = 1;

  DataType grid_step = 1.f;
  auto start = std::chrono::high_resolution_clock::now();
  auto edge_cubes = finding_edge_cubes(atom_host, grid_step);
  auto edge_cubes2 = finding_edge_cubes_helper(
      atom_host, grid_step, std::make_index_sequence<2>{}, 50, 50);

  int i = 0;
  for (auto edge_cube : edge_cubes) {

    i++;

    // for (auto j : edge_cube) {
    //   std::cout << j << " ";
    // }

    // std::cout << std::endl;

    vector3d point{static_cast<DataType>(edge_cube[0]),
                   static_cast<DataType>(edge_cube[1]),
                   static_cast<DataType>(edge_cube[2])};
    find_polygon_cuts(point, center, atom_host.radius, grid_step, faces);
  }

  // std::cout << "The number of edge cubes is: " << i << std::endl;
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;

  // std::cout << "The required time was: " << duration.count() << std::endl;

  // std::cout << "The number of faces is: " << faces.size() << std::endl;

  std::ofstream file("output.ply");

  print_faces_to_ply(file, faces);

  file.close();

  // std::cout << "The number of points considered is: " << j << std::endl;

  // vector3d center{0.1f, 0.1f, 0.1f};
  // DataType Radius = std::sqrt(3) * 10.f;
  // vector3d point{1.f, 0.f, 0.f};
  // point = point * (Radius - 0.2f);

  // std::list<Face> faces;

  // find_polygon_cuts(point, center, Radius, grid_step, faces);

  return 0;
}
