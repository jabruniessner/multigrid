#include "Atom_types.h"
#include "Domain.h"
#include "MultigridDomain.h"
#include "bitshift_lib.h"
#include "compute_volumes.h"
#include "cubes_cutter.h"
#include "cutting_tetrahedra.h"
#include "cycles.h"
#include "dot_finder.h"
#include "epsilon_marker.h"
#include "fileio.h"
#include "get_epsilon_values.h"
#include "iterate_tets.h"
#include <array>
#include <boost/container/static_vector.hpp>
#include <cstddef>
#include <cstdint>
#include <experimental/mdspan>
#include <iostream>
#include <sycl/sycl.hpp>
#include <sys/types.h>
#include <type_traits>
#include <utility>

template <typename T> struct TD;

constexpr std::size_t nlev = 1;
constexpr std::size_t Dim = 3;
constexpr std::size_t side_length = 256;
constexpr DataType omega = 1.;

template <typename D_Type, std::size_t Type_dim, std::size_t... type_dirs>
using MG_domain =
    multigrid_domain::Multigrid_domain_t<D_Type, Dim, Type_dim, nlev,
                                         side_length, side_length, side_length,
                                         type_dirs...>;

constexpr auto &length = MG_domain<DataType, 0u>::length;
constexpr DataType box_length_x = 33.;
constexpr DataType grid_step = box_length_x / std::get<0>(length);

template <std::size_t N>
using MemFnPtrd_type =
    decltype(&MG_domain<DataType, 0u>::template get_domain<N>);

template <std::size_t N>
using d_type = std::remove_reference_t<
    std::invoke_result_t<MemFnPtrd_type<N>, MG_domain<DataType, 0u>>>;

template <std::size_t N>
using MemFnPtrd_type_eps =
    decltype(&MG_domain<std::uint8_t, 1u,
                        utils::factorial(Dim) - 2>::template get_domain<N>);

template <std::size_t N>
using d_type_eps = std::remove_reference_t<std::invoke_result_t<
    MemFnPtrd_type_eps<N>,
    MG_domain<std::uint8_t, 1u, utils::factorial(Dim) - 2>>>;

using domain::Domain;

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

template <Dimension Dim, UnsignedIntegral num_type, DataType grid_step,
          std::size_t... dims>
auto map(auto domain, sycl::id<Dim> I, auto epsilon_domain,
         std::index_sequence<dims...>) {

  constexpr DataType h_2_inv = 1 / (grid_step * grid_step);
  constexpr auto strides_array =
      std::remove_reference_t<decltype(domain)>::length;

  using d_type = std::remove_reference_t<decltype(domain)>;

  auto index_add_in_place = [=](int place, const d_type domain,
                                auto epsilon_values, auto... elems) {
    std::array<std::size_t, sizeof...(elems)> indices{elems...};
    indices[place] += 1;
    return std::apply(domain, indices) *
           get_local_epsilon(
               static_cast<num_type>(2 * Dim - epsilon_values[place]),
               epsilon_values[place]);
  };

  auto index_sub_in_place = [=](int place, const d_type domain,
                                auto epsilon_values, auto... elems) {
    std::array<std::size_t, sizeof...(elems)> indices{elems...};
    indices[place] -= 1;
    return std::apply(domain, indices) *
           get_local_epsilon(
               static_cast<num_type>(2 * Dim - epsilon_values[place + 3]),
               epsilon_values[place + 3]);
  };

  std::array<num_type, 2 * Dim> epsilon_values =
      get_tet_vals_dim<num_type, Dim, std::get<dims>(strides_array)...>(
          epsilon_domain, (int)I[dims]...);

  const auto subs =
      (index_sub_in_place(dims, domain, epsilon_values, I[dims]...) + ...);

  const auto adds =
      (index_add_in_place(dims, domain, epsilon_values, I[dims]...) + ...);

  DataType middle_value = 0;
  for (int i = 0; i < 2 * Dim; i++)
    middle_value += get_local_epsilon(
        static_cast<num_type>(2 * Dim - epsilon_values[i]), epsilon_values[i]);

  return (-adds - subs + middle_value * domain(I[dims]...)) * h_2_inv;
}

template <Dimension Dim, UnsignedIntegral num_type, DataType grid_step>
auto map(auto domain, sycl::id<Dim> id, auto epsilon_domain) {
  return map<Dim, num_type, grid_step>(domain, id, epsilon_domain,
                                       std::make_index_sequence<Dim>{});
}

template <Dimension Dim, std::size_t access_level> struct map_struct {

  map_struct() = delete;
  map_struct(d_type_eps<access_level> epsilon_domain)
      : epsilon_domain(epsilon_domain) {}

  map_struct(MG_domain<std::uint8_t, 1u, utils::factorial(Dim) - 2> eps_domain)
      : epsilon_domain(eps_domain.template get_domain<access_level>()) {}

  DataType operator()(d_type<access_level> domain, sycl::id<Dim> id) const {
    return map<Dim, std::uint8_t,
               (DataType)(grid_step *
                          utils::power_off(sqrt2, nlev - access_level))>(
        domain, id, epsilon_domain);
  }

  d_type_eps<access_level> epsilon_domain;
};

// This functions is intended for use with Paddingwidth 1

int main(int argc, char *argv[]) {

  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " <input_pqr_file> origin_x origin_y origin_z num_iters"
              << std::endl;
    return 1;
  }

  int num_iter = std::stoi(argv[5]);

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

  constexpr auto &length = MG_domain<DataType, 0u>::length;

  MG_domain<DataType, 0u> lhs_domain1(q);
  MG_domain<DataType, 0u> lhs_domain2(q);
  MG_domain<DataType, 0u> lhs_domain3(q);
  MG_domain<DataType, 0u> rhs_domain(q);
  MG_domain<DataType, 0u> boundary_values(q);
  MG_domain<DataType, 0u> defect_domain(q);

  Domain<3, std::get<0>(length), std::get<1>(length), std::get<2>(length)>
      u_domain(Paddings::PERIODIC, q, 1), convolved(Paddings::PERIODIC, q, 1),
      helper(Paddings::PERIODIC, q, 1);

  std::cout << "The number of values in the inside outside domain is: "
            << inside_outside.get_domain().num_values << std::endl;

  std::cout << "The number of values in the Volumes_tetrahedra domain is: "
            << Volumes_tetrahedra.get_domain().num_values << std::endl;

  using Domain_type = decltype(Volumes_tetrahedra.get_domain());
  using Domain_type_io = decltype(inside_outside.get_domain());

  {

    auto boundary_conditions = [=](DataType x, DataType y, DataType z) {
      return -3 * x * x - 4 * y * y + 7 * z * z;
    };

    auto &boundary_domain = lhs_domain1.template get_domain<nlev>();
    q.parallel_for(
         sycl::range<2>(std::get<1>(length) + 2, std::get<2>(length) + 2),
         [=](sycl::id<2> I) {
           boundary_domain(I[0], I[1], 0) = boundary_conditions(
               (DataType)I[0] / (std::get<1>(length) + 1),
               (DataType)I[1] / (std::get<2>(length) + 1), 0);

           boundary_domain(I[0], I[1], std::get<2>(length) + 1) =
               boundary_conditions((DataType)I[0] / (std::get<1>(length) + 1),
                                   (DataType)I[1] / (std::get<2>(length) + 1),
                                   1);

           boundary_domain(0, I[0], I[1]) = boundary_conditions(
               0, (DataType)I[0] / (std::get<1>(length) + 1),
               (DataType)I[1] / (std::get<2>(length) + 1));

           boundary_domain(std::get<0>(length) + 1, I[0], I[1]) =
               boundary_conditions(1,
                                   (DataType)I[0] / (std::get<1>(length) + 1),
                                   (DataType)I[1] / (std::get<2>(length) + 1));

           boundary_domain(I[0], 0, I[1]) = boundary_conditions(
               (DataType)I[0] / (std::get<1>(length) + 1), 0,
               (DataType)I[1] / (std::get<2>(length) + 1));

           boundary_domain(I[0], std::get<1>(length) + 1, I[1]) =
               boundary_conditions((DataType)I[0] / (std::get<1>(length) + 1),
                                   1,
                                   (DataType)I[1] / (std::get<2>(length) + 1));
         })
        .wait();

    //  std::cout << "The boundary domain is: " << std::endl;
    //  boundary_domain.print_domain();
  }

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
      find_dots_in_sphere(Atom, io_domain, grid_step);
    });

    // Coarsening the dots domain
    coarsen_domains(inside_outside);

    mark_epsilon_MG(inside_outside, Volumes_tetrahedra);
  }

  // Defining the necessary function and creating the CG and GS type
  {

    // Instantiation the linear map for the coarse grid solver
    // using d_type = decltype(lhs_domain1.template get_domain<1>());
    auto vol_domain = Volumes_tetrahedra.template get_domain<1>();

    auto domain = lhs_domain1.template get_domain<1>();

    auto solver =
        cg_solver::make_solver<DataType, 3>(Float<(DataType)1e-8>{}, domain);

    using num_type = decltype(Volumes_tetrahedra)::ValueType;

    auto smoother = cycles::make_GS_Smoother_epsilon<num_type>(
        lhs_domain1, Volumes_tetrahedra);

    multigrid_domain::Multi_Level_map<Dim, DataType, map_struct, nlev> map_type{
        Volumes_tetrahedra};

    cycles::V_Cycle_base v_cycle{smoother, smoother, solver, lhs_domain1};

    auto *current = &lhs_domain2;
    auto *next = &lhs_domain3;
    Domain<3, std::get<0>(length), std::get<1>(length), std::get<2>(length)>
        helper2(Paddings::PERIODIC, q, 1);

    std::stringstream filenames;

    filenames << "deviations" << side_length << ".txt";

    std::ofstream out_file_devation(filenames.str());

    auto start = std::chrono::high_resolution_clock::now();

    std::index_sequence<1> num_iters_level;
    std::index_sequence<2> smoother_sequence_pre;
    std::index_sequence<2> smoother_sequence_post;

    for (int num = 0; num < num_iter; num++) {

      // Computing the defect
      convolution::Subtract_Convolve_map(
          defect_domain.get_domain(), lhs_domain1.get_domain(),
          rhs_domain.get_domain(), map_type.get_map());

      v_cycle.iteration(*next, *current, defect_domain, map_type, grid_step,
                        omega, num_iters_level, smoother_sequence_pre,
                        smoother_sequence_post, true);

      add_domains(lhs_domain1.get_domain(), next->get_domain(),
                  lhs_domain1.get_domain());

      // Computing the defect after adding
      convolution::Subtract_Convolve_map(
          defect_domain.get_domain(), lhs_domain1.get_domain(),
          rhs_domain.get_domain(), map_type.get_map());

      DataType residual{};
      domain_compute_norm_squared(residual, defect_domain.get_domain());
      residual =
          std::sqrt(residual / rhs_domain.template get_domain<nlev>().num_dofs);
      std::cout << "The residual after " << num << " iterations is " << residual
                << std::endl;

      std::swap(current, next);

      //  std::cout << "After the iterations the current is: " <<
      // std::endl;
      //  current->get_domain().print_domain();
      //  std::cout << "After the iteration the next is: " << std::endl;
      //  next->get_domain().print_domain();
    }

    q.wait();

    // Now doing the actual solving
  }

  return 0;
}
