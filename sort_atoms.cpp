#include "Atom_types.h"
#include "Domain.h"
#include "blas.h"
#include "cubes_cutter.h"
#include "fileio.h"
#include "hipSYCL/sycl/usm.hpp"
#include <algorithm>
#include <boost/container/static_vector.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

constexpr int Dim = 3;
constexpr int side_length = 353;

constexpr int side_length_x = 520;
constexpr int side_length_y = 520;
constexpr int side_length_z = 80;
constexpr int num_edges = 19;

using DataType = float;

constexpr DataType grid_step = 1.f;

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

bool compare_atoms(std::vector<Atom<float>> &a, std::vector<Atom<float>> &b) {
  return a.front().radius < b.front().radius;
}

struct hash_atoms {
  std::size_t operator()(float radius) const { return std::round(radius * 10); }
};

// std::size_t hash_atoms(float radius) { return std::round(radius * 10); }

int main(int argc, char *argv[]) {

  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <input_pqr_file> <origin_x> <origin_y> <origin_z>"
              << std::endl;
    return 1;
  }

  DataType origin_x = std::atof(argv[2]), origin_y = std::atof(argv[3]),
           origin_z = std::atof(argv[4]);

  sycl::cpu_selector selector;
  sycl::queue q(selector);

  domain::Grid<std::uint32_t, Dim + 1, side_length_x, side_length_y,
               side_length_z, num_edges>
      grid_edges(Paddings::PERIODIC, q, 1);

  domain::Grid<std::uint32_t, Dim, side_length_x, side_length_y, side_length_z>
      inside_outside(Paddings::PERIODIC, q, 1);

  q.memset(grid_edges.values_buff, 0,
           sizeof(std::uint32_t) * grid_edges.num_values);
  q.memset(inside_outside.values_buff, 0,
           sizeof(std::uint32_t) * inside_outside.num_values);
  q.wait();

  std::string filename = argv[1];
  std::list<Atom<float>> atoms;

  read_pqr_file(filename, atoms);

  std::unordered_set<std::size_t> radius_set;
  std::unordered_multimap<std::size_t, Atom<float>, hash_atoms> atom_map;

  hash_atoms hash_fn;

  for (auto &atom : atoms) {
    atom.Position[0] -= origin_x;
    atom.Position[1] -= origin_y;
    atom.Position[2] -= origin_z;
    radius_set.insert(hash_fn(atom.radius));
    atom_map.insert({hash_fn(atom.radius), atom});
  }

  std::cout << "Unique radii: " << radius_set.size() << std::endl;
  std::cout << "Unique atoms: " << atom_map.size() << std::endl;

  std::vector<std::vector<Atom<float>>> sorted_atoms;
  sorted_atoms.reserve(radius_set.size());

  for (const auto &radius : radius_set) {

    if (!radius)
      continue;

    auto range = atom_map.equal_range(radius);
    std::vector<Atom<float>> atom_vector;
    atom_vector.reserve(std::distance(range.first, range.second));

    for (auto it = range.first; it != range.second; ++it) {
      atom_vector.push_back(it->second);
    }
    sorted_atoms.push_back(atom_vector);
  }

  std::sort(sorted_atoms.begin(), sorted_atoms.end(), compare_atoms);

  for (const auto &atom_vector : sorted_atoms) {
    std::cout << "Radius: " << atom_vector.front().radius
              << " number of atoms: " << atom_vector.size() << std::endl;
  }

  std::vector<Atom<float> *> atoms_device(sorted_atoms.size());
  for (size_t i = 0; i < sorted_atoms.size(); ++i) {

    atoms_device[i] =
        sycl::malloc_device<Atom<DataType>>(sorted_atoms[i].size(), q);

    q.wait();
    q.memcpy(atoms_device[i], sorted_atoms[i].data(),
             sizeof(Atom<DataType>) * sorted_atoms[i].size());
  }
  q.wait();

  cubes_cutter::Cutter cutter{};

  auto atoms_device_data = atoms_device.data();

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 100; i++)
    for (int j = 0; j < sorted_atoms.size(); j++)
      q.parallel_for(sycl::range<1>(sorted_atoms[j].size()),
                     [=](sycl::id<1> i) {
                       Atom<DataType> atom = atoms_device_data[j][i];
                       cubes_cutter::cutting_cubes(
                           cutter, grid_edges, inside_outside, atom, grid_step);
                     });

  auto end_iterations = std::chrono::high_resolution_clock::now();

  q.wait_and_throw();

  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "Elapsed time for iterations: "
            << std::chrono::duration<double>(end_iterations - start).count()
            << "s\n";

  std::cout << "Elapsed time: "
            << std::chrono::duration<double>(end - start).count() << "s\n";

  return 0;
}
