#include "Atom_types.h"
#include "Domain.h"
#include "compute_faces.h"
#include "compute_volumes.h"
#include "cubes_cutter.h"
#include "cutting_tetrahedra.h"
#include "fileio.h"
#include "hipSYCL/sycl/libkernel/half.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "ply_file_writer.h"
#include <array>
#include <boost/container/static_vector.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <experimental/mdspan>
#include <iostream>
#include <limits>
#include <span>
#include <sycl/sycl.hpp>
#include <sys/types.h>
#include <tuple>
#include <utility>

constexpr int Dim = 3;
constexpr int side_length = 353;

constexpr int side_length_x = 510;
constexpr int side_length_y = 510;
constexpr int side_length_z = 60;
constexpr int num_edges = 19;

// using DataType = float;

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

int main(int argc, char *argv[]) {

  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <input_pqr_file> origin_x origin_y origin_z" << std::endl;
    return 1;
  }

  // std::cout << "The size of an std::array<sycl::half, 19> is "
  //           << sizeof(cube_edges) << std::endl;

  // std::cout << "The size of a edge_ref is " << sizeof(edge_ref) << std::endl;

  sycl::gpu_selector selector;
  sycl::queue q(selector,
                sycl::property_list{sycl::property::queue::in_order{}});

  // sycl::queue q(selector);

  domain::Grid<std::uint32_t, Dim + 1, side_length_x, side_length_y,
               side_length_z, num_edges>
      grid_edges(Paddings::PERIODIC, q, 1);

  domain::Grid<std::uint32_t, Dim, side_length_x, side_length_y, side_length_z>
      inside_outside(Paddings::PERIODIC, q, 1);

  domain::Grid<DataType, Dim + 1, side_length_x, side_length_y, side_length_z,
               6>
      Volumes_tetrahedra(Paddings::PERIODIC, q, 1);

  q.memset(grid_edges.values_buff, 0,
           sizeof(std::uint32_t) * grid_edges.num_values);
  q.memset(inside_outside.values_buff, 0,
           sizeof(std::uint32_t) * inside_outside.num_values);
  q.memset(Volumes_tetrahedra.values_buff, 0,
           sizeof(DataType) * Volumes_tetrahedra.num_values);
  q.wait();

  //  auto start2 = std::chrono::high_resolution_clock::now();
  //  q.parallel_for(sycl::range<1>(grid_edges.num_values), [=](sycl::id<1> i) {
  //     grid_edges.values_buff[i] = 0;
  //   }).wait();
  //  auto end2 = std::chrono::high_resolution_clock::now();

  // std::chrono::duration<double> elapsed_seconds2 = end2 - start2;
  // std::cout << "Elapsed time for parallel_for: " << elapsed_seconds2.count()
  //           << "s\n";

  const DataType grid_step = 1.f;

  std::list<Atom<DataType>> atoms;

  // std::array<DataType, 3> origin = {-50, -50, -50};

  // atoms.push_back({.charge = -1.0f});
  // atoms.back().Position = {60.f, 50.f, 50.f};
  // atoms.back().radius = 20.f;
  // atoms.push_back({.charge = 1.0f});
  // atoms.back().Position = {40.f, 50.f, 50.f};
  // atoms.back().radius = 20.f;

  std::string filename = argv[1];
  DataType origin_x = std::atof(argv[2]);
  DataType origin_y = std::atof(argv[3]);
  DataType origin_z = std::atof(argv[4]);

  read_pqr_file(filename, atoms);

  std::vector<Atom<DataType>> atoms_vector;
  atoms_vector.reserve(atoms.size());

  for (auto &atom : atoms) {
    atom.Position[0] -= origin_x;
    atom.Position[1] -= origin_y;
    atom.Position[2] -= origin_z;
    // atom.radius += 1.5f;
    atoms_vector.push_back(atom);
  }

  cubes_cutter::Cutter cutter{};
  volume_computer::Volume_comp v_comp{};

  // Copy atoms to device
  Atom<DataType> *atoms_device =
      sycl::malloc_device<Atom<DataType>>(atoms_vector.size(), q);

  q.memcpy(atoms_device, atoms_vector.data(),
           sizeof(Atom<DataType>) * atoms_vector.size())
      .wait();

  // // TD<decltype(grid_edges)> grid_edges_t;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 100; i++) {
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> i) {
      Sphere<DataType, Dim> &atom = atoms_device[i];
      auto arg_tuple =
          std::forward_as_tuple(atom, grid_step, grid_edges, inside_outside);
      cubes_cutter::cutting_cubes(cutter, arg_tuple);
    });

    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> i) {
      Sphere<DataType, Dim> &atom = atoms_device[i];
      auto arg_tuple = std::forward_as_tuple(
          atom, grid_step, grid_edges, inside_outside, Volumes_tetrahedra);
      cubes_cutter::cutting_cubes(v_comp, arg_tuple);
    });
  }

  q.wait();

  auto end = std::chrono::high_resolution_clock::now();

  auto time = std::chrono::duration<double>(end - start);

  std::cout << "The required time was: " << time.count() << std::endl;

  return 0;
}
