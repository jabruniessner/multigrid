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
#include "get_epsilon_values.h"
#include "hipSYCL/sycl/libkernel/half.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/sycl/usm.hpp"
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
constexpr std::size_t side_length = 1;

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

  //  if (argc != 5) {
  //    std::cerr << "Usage: " << argv[0]
  //              << " <input_pqr_file> origin_x origin_y origin_z" <<
  //              std::endl;
  //    return 1;
  //  }

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

  auto &epsilon_domain_coarse = Volumes_tetrahedra.template get_domain<1>();
  auto &inside_outside_coarse = inside_outside.template get_domain<1>();
  auto values_device = sycl::malloc_device<std::uint8_t>(2 * Dim, q);
  std::array<std::uint8_t, 6> values{};
  // TD<decltype(inside_outside_coarse)> td1;
  // TD<decltype(epsilon_domain_coarse)> td;

  auto f = [&](auto &io_domain) {
    mark_epsilons<Dim, std::uint32_t, std::uint8_t>(io_domain,
                                                    epsilon_domain_coarse);
    q.single_task([=]() {
      auto vals = get_tet_vals_dim<std::uint8_t, Dim, 1, 1, 1>(
          epsilon_domain_coarse, 1, 1, 1);

      for (int i = 0; i < 2 * Dim; i++)
        values_device[i] = vals[i];
    });

    q.memcpy(values.data(), values_device,
             sizeof(std::uint8_t) * values.size());

    q.memset(epsilon_domain_coarse.values_buff, 0,
             epsilon_domain_coarse.num_values * sizeof(std::uint8_t));

    q.wait();

    std::cout << "The values of the array are: " << std::endl;
    for (auto i : values)
      std::cout << (int)i << " ";
    std::cout << std::endl;
  };

  set_n_values_to_one_and_f(inside_outside_coarse, 1, 0, f);

  return 0;
}
