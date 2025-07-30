#include "Atom_types.h"
#include "Domain.h"
#include "MultigridDomain.h"
#include "bitshift_lib.h"
#include "compute_faces.h"
#include "compute_volumes.h"
#include "cubes_cutter.h"
#include "cutting_tetrahedra.h"
#include "dot_finder.h"
#include "epsilon_marker.h"
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
#include <fstream>
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

template <UnsignedIntegral num_type1, Dimension Dim, Length... strides_all,
          typename func_type>
void set_n_values_to_one_and_f(
    domain::Grid<num_type1, Dim, strides_all...> domain, std::size_t n,
    std::size_t current_val, func_type function) {
  if (n == 0) {
    function(domain);
  } else {
    for (std::size_t i = current_val; i < domain.num_values; i++) {
      domain.q.submit([=](sycl::handler &h) {
        h.single_task([=]() { domain.values_buff[i] = 1; });
      });
      set_n_values_to_one_and_f(domain, n - 1, i + 1, function);
      domain.q.memset(&domain.values_buff[i], 0,
                      sizeof(num_type1) * (domain.num_values - (i)));
    }
  }
}

// This functions is intended for use with Paddingwidth 1

int checker_count = 0;
void check_correctness(std::array<std::uint8_t, 8> epsilon,
                       std::array<std::uint8_t, 6> tets) {

  checker_count++;

  auto i = std::reduce(tets.begin(), tets.end());
  // Checking whether the number is correct first check for 0 and 7
  if (epsilon[0] == 1 || epsilon[7] == 1) {
    assert(i == 6);
    return;
  }

  for (int k = 1, l = 0; k <= 4; k *= 2) {
    if (epsilon[k]) {
      //  std::cout << "The value of k is: ";
      //  std::cout << k << std::endl;
      //  std::cout << "The value of l is: " << l << std::endl;
      assert(tets[l] == 1 && tets[l + 1] == 1);
    }
    l += 2;
  }

  for (std::uint8_t k : {3, 5, 6}) {
    if (epsilon[k]) {
      // getting_list_significant bit
      std::uint8_t lsb = k & (~k + 1);
      std::uint8_t indices = ~lsb & 7;
      bool passed = false;

      for (std::uint8_t j = 0; indices != 0; indices &= indices - 1, j++) {
        std::uint8_t lsb_i = indices & (~indices + 1);
        //  std::cout << "The value for k is: " << (int)k << std::endl;
        //  std::cout << "lsb_i: " << (int)lsb_i << " lsb: " << (int)lsb
        //            << std::endl;
        if (lsb_i + lsb == k) {
          auto index = __builtin_ctz(lsb);
          assert(tets[2 * index + j]);
          passed = true;
        }
      }

      assert(passed);
    }
  }
}

int main(int argc, char *argv[]) {

  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <input_pqr_file> origin_x origin_y origin_z" << std::endl;
    return 1;
  }

#ifdef DEBUGMODE
  sycl::cpu_selector selector;
#else
  sycl::gpu_selector selector;
#endif

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

    domain::Grid<std::uint8_t, 3, 0, 0, 0> trial_domain(Paddings::PERIODIC, q,
                                                        1);

    domain::Grid<std::uint8_t, 4, 0, 0, 0, 4> trial_tetraeda_domain(
        Paddings::PERIODIC, q, 1);

    std::ofstream outfile("outfile.inp");
    domain::print_grid_to_inp<std::uint8_t, 3, 0, 0, 0>(trial_tetraeda_domain,
                                                        1.f, outfile);

    int i = 0;
    auto func = [&](decltype(trial_domain)) mutable { i++; };
    auto func2 = [&](decltype(trial_domain) domain) {
      std::array<std::uint8_t, 8> a{};
      domain.q.memcpy(a.data(), domain.values_buff,
                      sizeof(std::uint8_t) * domain.num_values);
      domain.q.wait();
      for (auto i : a)
        std::cout << (int)i << " ";
      std::cout << std::endl;
    };

    auto func3 = [&](decltype(trial_domain) domain) {
      q.memset(trial_tetraeda_domain.values_buff, 0,
               sizeof(std::uint8_t) * trial_tetraeda_domain.num_values);
      mark_epsilons<3, std::uint8_t, std::uint8_t>(domain,
                                                   trial_tetraeda_domain);

      // getting the values of the tetraeda_array
      std::array<std::uint8_t, 6> tets{};
      q.memcpy(tets.data(), trial_tetraeda_domain.values_buff,
               sizeof(std::uint8_t) * 6);

      // getting the values from the domain
      std::array<std::uint8_t, 8> epsilons{};
      q.memcpy(epsilons.data(), trial_domain.values_buff,
               sizeof(std::uint8_t) * trial_domain.num_values);

      q.wait();
      // Finished getting the value from the device

      check_correctness(epsilons, tets);
    };

    std::cout << "The number of values of the domain is: "
              << trial_domain.num_values << std::endl;

    for (std::size_t i = 0; i <= 8; i++) {
      set_n_values_to_one_and_f(trial_domain, i, 0, func3);
    }

    std::cout << "The value of i is: " << i << std::endl;

    q.wait();

    trial_domain.print_domain();
  }

  std::cout << "The checker count is: " << checker_count << std::endl;

  return 0;
}
