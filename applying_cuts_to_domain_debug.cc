#include "Atom_types.h"
#include "Domain.h"
#include "compute_faces.h"
#include "cutting_tetrahedra.h"
#include "fileio.h"
#include "hipSYCL/sycl/libkernel/half.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"
#include "ply_file_writer.h"
#include <array>
#include <boost/container/static_vector.hpp>
#include <cstddef>
#include <cstdint>
#include <experimental/mdspan>
#include <iostream>
#include <span>
#include <sycl/sycl.hpp>
#include <sys/types.h>

constexpr int Dim = 3;
constexpr int side_length = 353;

using DataType = float;

template <typename T> struct TD;

using cube_edges = std::array<sycl::half, 19>;
using cube_edges_refs =
    std::array<sycl::atomic_ref<DataType, sycl::memory_order::relaxed,
                                sycl::memory_scope::device>,
               19>;

using edge_ref = sycl::atomic_ref<DataType, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device>;

using vector3d = blas::vector<DataType, Dim>;
using Face = boost::container::static_vector<vector3d, 3>;

using cube_tetrahedrons = std::array<sycl::half, 6>;
using cube_tetrahedrons_refs = std::array<sycl::atomic<sycl::half>, 6>;

struct Cutter {
  template <std::size_t... side_lengths>
  void operator()(
      const domain::Grid<std::uint32_t, Dim + 1, side_lengths..., 19> &tet_grid,
      const domain::Grid<std::uint8_t, Dim, side_lengths...> &inside_outside,
      const DataType *sphere_position, const DataType sphere_radius,
      const DataType grid_step, const std::size_t position_0,
      const std::size_t position_1, const std::size_t position_2) const {

    Cutter_utils::vector3d point{static_cast<DataType>(position_0),
                                 static_cast<DataType>(position_1),
                                 static_cast<DataType>(position_2)};
    Cutter_utils::vector3d center{sphere_position[0], sphere_position[1],
                                  sphere_position[2]};
    DataType radius = sphere_radius;
    std::uint8_t points = Cutter_utils::find_polygon_cuts(
        point, center, radius, grid_step,
        &tet_grid(position_0, position_1, position_2, 0));

    inside_outside(position_0, position_1, position_2) = points;
    //   Implement the cutting logic here
    // for (int i = 0; i < 19; ++i) {
    //   tet_grid(position_0, position_1, position_2, i) = 0;
    // }
  }
};

int main(int argc, char *argv[]) {

  //  if (argc != 5) {
  //    std::cerr << "Usage: " << argv[0]
  //              << " <input_pqr_file> origin_x origin_y origin_z" <<
  //              std::endl;
  //    return 1;
  //  }

  // std::cout << "The size of an std::array<sycl::half, 19> is "
  //           << sizeof(cube_edges) << std::endl;

  // std::cout << "The size of a edge_ref is " << sizeof(edge_ref) << std::endl;

  sycl::cpu_selector selector;
  sycl::queue q(selector,
                sycl::property_list{sycl::property::queue::in_order{}});

  // sycl::queue q(selector);

  domain::Grid<std::uint32_t, Dim + 1, 100, 100, 100, 19> grid_edges(
      Paddings::PERIODIC, q, 1);

  domain::Grid<std::uint8_t, Dim, 100, 100, 100> inside_outside(
      Paddings::PERIODIC, q, 1);

  q.memset(grid_edges.values_buff, 0,
           sizeof(std::uint32_t) * grid_edges.num_values);
  q.memset(inside_outside.values_buff, 0,
           sizeof(std::uint8_t) * inside_outside.num_values);
  q.wait();

  auto start2 = std::chrono::high_resolution_clock::now();
  q.parallel_for(sycl::range<1>(grid_edges.num_values), [=](sycl::id<1> i) {
     grid_edges.values_buff[i] = 0;
   }).wait();
  auto end2 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed_seconds2 = end2 - start2;
  std::cout << "Elapsed time for parallel_for: " << elapsed_seconds2.count()
            << "s\n";

  const DataType grid_step = 1.f;

  std::list<Atom<DataType>> atoms;

  // std::array<DataType, 3> origin = {-50, -50, -50};

  atoms.push_back({.charge = 1.0f});
  atoms.back().Position = {50.f, 50.f, 50.f};
  atoms.back().radius = 45.f;

  std::vector<Atom<DataType>> atoms_vector;
  atoms_vector.reserve(atoms.size());

  for (auto &atom : atoms) {
    atoms_vector.push_back(atom);
  }

  Cutter cutter{};

  // Copy atoms to device
  Atom<DataType> *atoms_device =
      sycl::malloc_device<Atom<DataType>>(atoms_vector.size(), q);

  q.memcpy(atoms_device, atoms_vector.data(),
           sizeof(Atom<DataType>) * atoms_vector.size())
      .wait();

  // // TD<decltype(grid_edges)> grid_edges_t;

  auto start = std::chrono::high_resolution_clock::now();
  // for (int i = 0; i < 100; i++)
  q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> i) {
    Atom<DataType> atom = atoms_device[i];
    cutting_cubes(cutter, grid_edges, inside_outside, atom, grid_step);
  });

  q.wait();

  auto end = std::chrono::high_resolution_clock::now();

  std::vector<std::uint8_t> inside_outside_host;
  inside_outside_host.reserve(inside_outside.num_values);

  q.memcpy(inside_outside_host.data(), inside_outside.values_buff,
           sizeof(std::uint8_t) * inside_outside.num_values)
      .wait();

  auto inside_outside_span =
      std::mdspan(inside_outside_host.data(), 100, 100, 100);

  // Copy the grid_edges back to the host
  std::vector<std::uint32_t> grid_edges_host;
  grid_edges_host.reserve(grid_edges.num_values);
  q.memcpy(grid_edges_host.data(), grid_edges.values_buff,
           sizeof(std::uint32_t) * grid_edges.num_values)
      .wait();

  auto grid_edges_span = std::mdspan(grid_edges_host.data(), 100, 100, 100, 19);

  int cut_cells = 0;

  std::list<Face> faces;

  for (int i = 0; i < inside_outside_span.extent(0); i++) {
    for (int j = 0; j < inside_outside_span.extent(1); j++) {
      for (int k = 0; k < inside_outside_span.extent(2); k++) {

        std::uint8_t points = inside_outside_span[i, j, k];
        if (points == 0 || points == 255)
          continue;

        std::uint32_t *grid_value = &(grid_edges_span[i, j, k, 0]);
        std::span<std::uint32_t> grid_value_span(grid_value, 19);

        vector3d point{static_cast<DataType>(i * grid_step),
                       static_cast<DataType>(j * grid_step),
                       static_cast<DataType>(k * grid_step)};

        compute_faces(points, grid_step, point, grid_value_span, faces);
      }
    }
  }

  std::ofstream outfile("cut_faces_simple.ply");

  ply::print_faces_to_ply(outfile, faces);

  std::cout << "Number of cut cells: " << cut_cells << std::endl;

  std::chrono::duration<double> elapsed_seconds = end - start;

  std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

  // std::cout << "Hello, World!" << std::endl;
  return 0;
}
