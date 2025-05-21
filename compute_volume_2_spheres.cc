#include "Atom_types.h"
#include "Domain.h"
#include "blas.h"
#include "compute_volumes.h"
#include "cubes_cutter.h"
#include "hipSYCL/sycl/device_selector.hpp"
#include "hipSYCL/sycl/usm.hpp"
#include "predefinitions.h"
#include <iostream>

constexpr std::size_t side_length = 50;
constexpr std::size_t num_edges = 19;
constexpr std::size_t Dim = 3;

using vector3d = blas::vector<DataType, 3>;

// struct TD;

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
      Atom<DataType>{std::array<DataType, 3>{30.f, 25.f, 25.f}, 3.f, 1.f});
  atoms.push_back(
      Atom<DataType>{std::array<DataType, 3>{33.f, 25.f, 25.f}, 3.f, -1.f});

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

  DataType *Volume_device = sycl::malloc_device<DataType>(1, q);
  q.single_task([=] { *Volume_device = 0; });

  q.parallel_for(sycl::range<Dim>(side_length, side_length, side_length),
                 sycl::reduction(Volume_device, sycl::plus<>()),
                 [=](sycl::id<Dim> I, auto &r) {
                   for (int i = 0; i < 6; i++)
                     r += Volumes_tetrahedra(I[0], I[1], I[2], i);
                 });

  std::size_t *num_cut_cells = sycl::malloc_device<std::size_t>(1, q);
  std::size_t *num_volumes_inside = sycl::malloc_device<std::size_t>(1, q);
  q.single_task([=] {
    *num_cut_cells = 0;
    *num_volumes_inside = 0;
  });
  q.parallel_for(sycl::range<Dim>(side_length, side_length, side_length),
                 sycl::reduction(num_cut_cells, sycl::plus<>()),
                 sycl::reduction(num_volumes_inside, sycl::plus<>()),
                 [=](sycl::id<Dim> I, auto &n, auto &i) {
                   auto point = inside_outside(I[0], I[1], I[2]);
                   if (point != 0) {
                     i += 1;
                     if (point != 0b11111111) {
                       n += 1;
                     }
                   }
                 });

  // TD<doubel>;
  DataType Volume_host{0};
  std::size_t num_cut_cells_host{};
  std::size_t num_volumes_inside_host{};
  q.memcpy(&Volume_host, Volume_device, sizeof(DataType));
  q.memcpy(&num_cut_cells_host, num_cut_cells, sizeof(std::size_t));
  q.memcpy(&num_volumes_inside_host, num_volumes_inside, sizeof(std::size_t));

  q.wait();

  std::cout << "Hello, World!" << std::endl;
  std::cout << "The overall computed Volume is: " << Volume_host / 6.
            << std::endl;

  std::cout << "The number of cells (partially) inside the Sphere are: "
            << num_volumes_inside_host << std::endl;

  std::cout << "The number of cut cells are: " << num_cut_cells_host
            << std::endl;

  return 0;
}
