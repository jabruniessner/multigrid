#include "Atom_types.h"
#include "Domain.h"
#include "cutting_tetrahedra.h"
#include "fileio.h"
#include "hipSYCL/sycl/libkernel/half.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"
// #include "tetraeda_type.h"
#include <array>
#include <cstddef>
#include <iostream>
#include <sycl/sycl.hpp>
#include <sys/types.h>

constexpr int Dim = 3;
constexpr int side_length = 353;

using DataType = float;

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
  void operator()(const domain::Grid<cube_edges_refs, Dim, side_length,
                                     side_length, side_length> &tet_grid,
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
    // Cutter_utils::find_polygon_cuts(
    //     point, center, radius, grid_step,
    //     tet_grid(position_0, position_1, position_2));
    //  Implement the cutting logic here
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
  sycl::queue q(selector,
                sycl::property_list{sycl::property::queue::in_order{}});

  domain::Grid<cube_edges, Dim, side_length, side_length, side_length>
      grid_edges(Paddings::PERIODIC, q, 1);

  domain::Grid<cube_edges_refs, Dim, side_length, side_length, side_length>
      grid_edges_refs(Paddings::PERIODIC, q, 1);

  domain::Grid<cube_tetrahedrons, Dim, side_length, side_length, side_length>
      grid_tetrahedrons(Paddings::PERIODIC, q, 1);

  domain::Grid<cube_tetrahedrons_refs, Dim, side_length, side_length,
               side_length>
      grid_tetrahedrons_refs(Paddings::PERIODIC, q, 1);

  domain::Domain<Dim, side_length, side_length, side_length> values(
      Paddings::PERIODIC, q, 1);

  // std::this_thread::sleep_for(std::chrono::seconds(20));

  DataType grid_step = 1.f;

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

  q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> i) {
     Atom<DataType> atom = atoms_device[i];
     cutting_cubes(cutter, grid_edges_refs, atom, grid_step);
   }).wait();

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;

  std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

  std::cout << "Hello, World!" << std::endl;
  return 0;
}
