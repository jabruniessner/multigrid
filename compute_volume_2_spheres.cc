#include "Atom_types.h"
#include "Domain.h"
#include "blas.h"
#include "compute_volumes.h"
#include "cubes_cutter.h"
#include "hipSYCL/sycl/device_selector.hpp"
#include "predefinitions.h"
#include <iostream>

constexpr std::size_t side_length = 50;
constexpr std::size_t num_edges = 19;
constexpr std::size_t Dim = 3;

using vector3d = blas::vector<DataType, 3>;

int main(int argc, char *argv[]) {

  sycl::cpu_selector selector;
  sycl::queue q(selector,
                sycl::property_list{sycl::property::queue::in_order{}});

  domain::Grid<std::uint32_t, Dim + 1, side_length, side_length, side_length,
               num_edges>
      grid_edges(Paddings::PERIODIC, q, 1);

  domain::Grid<std::uint32_t, Dim, side_length, side_length, side_length>
      inside_outside(Paddings::PERIODIC, q, 1);

  domain::Grid<DataType, Dim + 1, side_length, side_length, side_length, 6>
      Volumes_tetrahedra(Paddings::PERIODIC, q, 1);

  q.memset(grid_edges.values_buff, 0,
           sizeof(std::uint32_t) * grid_edges.num_values);
  q.memset(inside_outside.values_buff, 0,
           sizeof(std::uint32_t) * inside_outside.num_values);

  q.wait();

  const DataType grid_step = 1.f;

  std::vector<Atom<DataType>> atoms;
  atoms.push_back(
      Atom<DataType>{std::array<DataType, 3>{25.f, 25.f, 25.f}, 10.f, 1.f});
  atoms.push_back(
      Atom<DataType>{std::array<DataType, 3>{35.f, 35.f, 35.f}, 10.f, -1.f});

  cubes_cutter::Cutter cutter{};

  Atom<DataType> *atoms_device =
      sycl::malloc_device<Atom<DataType>>(atoms.size(), q);

  q.memcpy(atoms_device, atoms.data(), sizeof(Atom<DataType>) * atoms.size())
      .wait();

  q.parallel_for(sycl::range<1>(atoms.size()), [=](sycl::id<1> i) {
    Atom<DataType> atom = atoms_device[i];
    auto arg_tuple =
        std::forward_as_tuple(static_cast<Sphere<DataType, Dim> &>(atom),
                              grid_step, grid_edges, inside_outside);
    cubes_cutter::cutting_cubes(cutter, arg_tuple);
  });

  volume_computer::Volume_comp v_comp{};
  q.parallel_for(sycl::range<1>(atoms.size()), [=](sycl::id<1> i) {
    Atom<DataType> atom = atoms_device[i];
    auto arg_tuple = std::forward_as_tuple(
        static_cast<Sphere<DataType, Dim> &>(atom), grid_step, grid_edges,
        inside_outside, Volumes_tetrahedra);
    cubes_cutter::cutting_cubes(v_comp, arg_tuple);
  });

  q.wait();

  // std::cout << "Hello, World!" << std::endl;
  return 0;
}
