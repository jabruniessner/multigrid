#include "Atom_types.h"
#include "Domain.h"
#include "cutting_tetrahedra.h"
#include "fileio.h"
#include "hipSYCL/sycl/libkernel/half.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"
// #include "tetraeda_type.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
// #include <mdspan>
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

using cube_tetrahedrons = std::array<sycl::half, 6>;
using cube_tetrahedrons_refs = std::array<sycl::atomic<sycl::half>, 6>;

struct Cutter {
  void operator()(const domain::Grid<std::uint32_t, Dim + 1, side_length,
                                     side_length, side_length, 19> &tet_grid,
                  const DataType *sphere_position, const DataType sphere_radius,
                  const DataType grid_step, const std::size_t position_0,
                  const std::size_t position_1,
                  const std::size_t position_2) const {

    Cutter_utils::vector3d point{static_cast<DataType>(position_0),
                                 static_cast<DataType>(position_1),
                                 static_cast<DataType>(position_2)};
    Cutter_utils::vector3d center{sphere_position[0], sphere_position[1],
                                  sphere_position[2]};
    DataType radius = sphere_radius;
    Cutter_utils::find_polygon_cuts(
        point, center, radius, grid_step,
        &tet_grid(position_0, position_1, position_2, 0));
    //   Implement the cutting logic here
    // for (int i = 0; i < 19; ++i) {
    //   tet_grid(position_0, position_1, position_2, i) = 0;
    // }
  }
};

int main(int argc, char *argv[]) {

  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <input_pqr_file> origin_x origin_y origin_z" << std::endl;
    return 1;
  }

  std::cout << "The size of an std::array<sycl::half, 19> is "
            << sizeof(cube_edges) << std::endl;

  std::cout << "The size of a edge_ref is " << sizeof(edge_ref) << std::endl;

  sycl::gpu_selector selector;
  //  sycl::queue q(selector,
  //                sycl::property_list{sycl::property::queue::out_of_order{}});

  sycl::queue q(selector);

  domain::Grid<std::uint32_t, Dim + 1, side_length, side_length, side_length,
               19>
      grid_edges(Paddings::PERIODIC, q, 1);
  {
    auto start = std::chrono::high_resolution_clock::now();

    q.memset(grid_edges.values_buff, 0,
             sizeof(DataType) * grid_edges.num_values);
    q.wait();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds1 = end - start;

    std::cout << "Elapsed time for memset: " << elapsed_seconds1.count()
              << "s\n";
  }

  {
    auto start = std::chrono::high_resolution_clock::now();

    q.memset(grid_edges.values_buff, 0,
             sizeof(DataType) * grid_edges.num_values);
    q.wait();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds1 = end - start;

    std::cout << "Elapsed time for memset: " << elapsed_seconds1.count()
              << "s\n";
  }

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

  std::array<DataType, 3> origin = {std::stof(argv[2]), std::stof(argv[3]),
                                    std::stof(argv[4])};

  std::string filename = {argv[1]};
  read_pqr_file(filename, atoms);

  std::vector<Atom<DataType>> atoms_vector;
  atoms_vector.reserve(atoms.size());

  for (auto &atom : atoms) {
    atom.Position[0] -= origin[0];
    atom.Position[1] -= origin[1];
    atom.Position[2] -= origin[2];
    atoms_vector.push_back(atom);
  }

  Cutter cutter{};

  // Copy atoms to device
  Atom<DataType> *atoms_device =
      sycl::malloc_device<Atom<DataType>>(atoms_vector.size(), q);

  auto start = std::chrono::high_resolution_clock::now();

  q.memcpy(atoms_device, atoms_vector.data(),
           sizeof(Atom<DataType>) * atoms_vector.size())
      .wait();

  // TD<decltype(grid_edges)> grid_edges_t;

  for (int i = 0; i < 400; i++)
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> i) {
      Atom<DataType> atom = atoms_device[i];
      cutting_cubes(cutter, grid_edges, atom, grid_step);
    });

  q.wait();

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;

  std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

  std::cout << "Hello, World!" << std::endl;
  return 0;
}
