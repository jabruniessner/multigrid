#include "Atom_types.h"
#include "Domain.h"
#include "MultigridDomain.h"
#include "bitshift_lib.h"
#include "compute_faces.h"
#include "compute_volumes.h"
#include "cubes_cutter.h"
#include "cutting_tetrahedra.h"
#include "dot_finder.h"
#include "fileio.h"
#include "hipSYCL/sycl/libkernel/half.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "iterate_tets.h"
#include "ply_file_writer.h"
#include <array>
#include <boost/container/static_vector.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <experimental/mdspan>
#include <iostream>
#include <locale>
#include <sycl/sycl.hpp>
#include <sys/types.h>
#include <tuple>
#include <utility>

constexpr DataType delta_epsilon = epsilon_p - epsilon_r;
constexpr DataType grid_step = 1.;
constexpr std::size_t nlev = 4;
constexpr std::size_t Dim = 3;
constexpr std::size_t side_length = 6;

template <typename D_Type, std::size_t Type_dim, std::size_t... type_dirs>
using MG_domain =
    multigrid_domain::Multigrid_domain_t<D_Type, Dim, Type_dim, nlev,
                                         side_length, side_length, side_length,
                                         type_dirs...>;

template <typename T> struct TD;

using cube_tetrahedrons = std::array<sycl::half, 6>;

template <std::size_t level = nlev>
void coarsen_domains(MG_domain<std::uint32_t, 0u> domain) {
  if constexpr (level <= 1) {
    return;
  } else {
    auto &dest = domain.template get_domain<level - 1>();
    auto &src = domain.template get_domain<level>();
    level_transition::coarsening_inject(dest, src);
    coarsen_domains<level - 1>(domain);
  }
}

template <std::size_t Dim, UnsignedIntegral num_type,
          UnsignedIntegral num_type_2, UnsignedIntegral num_type_3,
          Length... strides_all, typename... Positions>
void mark_epsilons_helper(
    std::array<num_type, Dim + 1> a, std::size_t j,
    domain::Grid<num_type_2, Dim, strides_all...> io_domain,
    domain::Grid<num_type_3, Dim + 1, strides_all..., utils::factorial(Dim) - 2>
        values,
    Positions... pos) {

  static_assert(sizeof...(strides_all) == Dim);

  num_type_3 cell_inside = 0;
  for (auto i : a) {
    auto vector =
        bitshift::convert_byte_to_vec<std::size_t, decltype(i), Dim>(i);
    vector = vector + decltype(vector){pos...};
    using vec_base = std::array<std::size_t, Dim>;

    if (std::apply(io_domain, (vec_base)vector) == 1) {
      values(pos..., j) = static_cast<num_type_3>(1);
      return;
    }
  }
}

template <std::size_t Dim, UnsignedIntegral num_type_2,
          UnsignedIntegral num_type_3, Length... strides_all,
          std::size_t... dirs>
void mark_epsilons_helper(
    domain::Grid<num_type_2, Dim, strides_all...> io_domain,
    domain::Grid<num_type_3, Dim + 1, strides_all..., utils::factorial(Dim) - 2>
        values,
    std::index_sequence<dirs...>) {

  auto &length = io_domain.length;
  io_domain.q.parallel_for(
      sycl::range<Dim>((std::get<dirs>(length) + 1)...), [=](sycl::id<Dim> I) {
        auto f = [&](std::array<std::uint8_t, Dim + 1> a, std::size_t j) {
          mark_epsilons_helper<Dim, std::uint8_t, num_type_2, num_type_3>(
              a, j, io_domain, values, I[dirs]...);
        };

        iterate_over_tets<Dim, std::uint8_t>(f);
      });
}

template <std::size_t Dim, UnsignedIntegral num_type_2,
          UnsignedIntegral num_type_3, Length... strides_all>
void mark_epsilons(
    domain::Grid<num_type_2, Dim, strides_all...> io_domain,
    domain::Grid<num_type_3, Dim + 1, strides_all..., utils::factorial(Dim) - 2>
        values) {
  mark_epsilons_helper<Dim, num_type_2, num_type_3>(
      io_domain, values, std::make_index_sequence<Dim>{});
}

template <UnsignedIntegral num_type1, UnsignedIntegral num_type2,
          std::size_t... level>
void mark_epsilon_MG(MG_domain<num_type1, 0u> inside_outside,
                     MG_domain<num_type2, 1u, utils::factorial(Dim) - 2> values,
                     std::index_sequence<level...>) {

  (mark_epsilons<Dim, num_type1, num_type2>(
       inside_outside.template get_domain<level>(),
       values.template get_domain<level>()),
   ...);
}

template <UnsignedIntegral num_type1, UnsignedIntegral num_type2>
void mark_epsilon_MG(
    MG_domain<num_type1, 0u> inside_outside,
    MG_domain<num_type2, 1u, utils::factorial(Dim) - 2> values) {

  mark_epsilon_MG(inside_outside, values,
                  utils::make_index_sequence_with_offset<1, nlev>());
}

int main(int argc, char *argv[]) {

  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <input_pqr_file> origin_x origin_y origin_z" << std::endl;
    return 1;
  }

  sycl::gpu_selector selector;
  sycl::queue q(selector,
                sycl::property_list{sycl::property::queue::in_order{}});

  // sycl::queue q(selector);

  MG_domain<std::uint32_t, 0u> inside_outside(q);
  MG_domain<std::uint8_t, 1u, utils::factorial(Dim) - 2> Volumes_tetrahedra(q);

  std::cout << "The number of values in the inside outside domain is: "
            << inside_outside.get_domain().num_values << std::endl;

  std::cout << "The number of values in the Volumes_tetrahedra domain is: "
            << Volumes_tetrahedra.get_domain().num_values << std::endl;

  using Domain_type = decltype(Volumes_tetrahedra.get_domain());
  using Domain_type_io = decltype(inside_outside.get_domain());

  //  TD<Domain_type> td;
  //  TD<Domain_type_io> td2;

  std::list<Atom<DataType>> atoms;

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

  // cubes_cutter::Cutter cutter{};
  // volume_computer::Volume_comp v_comp{};

  // Copy atoms to device
  {
    Atom<DataType> *atoms_device =
        sycl::malloc_device<Atom<DataType>>(atoms_vector.size(), q);

    q.memcpy(atoms_device, atoms_vector.data(),
             sizeof(Atom<DataType>) * atoms_vector.size());

    // Finding all the dots inside the protein
    auto &io_domain = inside_outside.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      find_dots_in_sphere(Atom, io_domain, static_cast<DataType>(1.));
    });

    // Coarsening the dots domain
    coarsen_domains(inside_outside);

    mark_epsilon_MG(inside_outside, Volumes_tetrahedra);
  }

  {
  }

  return 0;
}
