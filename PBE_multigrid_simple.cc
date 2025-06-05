#include "Convolution.h"
#include "Debye_Hueckel_functions.h"
#include "MultigridDomain.h"
#include "create_charge_distribution.h"
#include "cycles.h"
#include "dot_finder.h"
#include "fileio.h"
#include "hipSYCL/sycl/device_selector.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include "hipSYCL/sycl/usm.hpp"
#include "level_transition.h"
#include "scientific_quantities.h"
#include <array>
#include <chrono>
#include <cstddef>
#include <exception>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace cycles;
using namespace convolution;

constexpr Dimension Dim = 3;
constexpr std::size_t nlev = 2u;
constexpr std::size_t base_length = 8;
constexpr DataType omega = 1.;
constexpr DataType box_length = 16;
constexpr DataType ionic_strength = 0.15;
constexpr DataType kappa = KappaA(ionic_strength);
constexpr DataType kappa_2 = 0; // kappa * kappa; // kappa * kappa;
constexpr DataType ionradius = 1.5;
constexpr DataType grid_step = 0.5;

// constexpr DataType delta_epsilon = 0;
constexpr DataType delta_epsilon =
    epsilon_p - epsilon_r; // Difference in epsilon

using Domain_Type =
    Multigrid_domain<Dim, nlev, base_length, base_length, base_length>;

using Domain_Type_upper = decltype(Domain_Type::domain_t_v)::domain_t;

DataType sqr(double val) { return val * val; }

template <std::size_t level = nlev>
void Set_boundary_conditions(Atom<DataType> *atoms, std::size_t num_atoms,
                             const Domain_Type_upper &domain, std::size_t x,
                             std::size_t y, std::size_t z,
                             const DataType grid_step) {
  DataType buffer_value = 0;
  for (int i = 0; i < num_atoms; i++) {
    const DataType distance =
        std::sqrt(sqr(x * grid_step - atoms[i].Position[0]) +
                  sqr(y * grid_step - atoms[i].Position[1]) +
                  sqr(z * grid_step - atoms[i].Position[2]));

    buffer_value +=
        DH_Sphere(atoms[i].radius, atoms[i].charge, distance, kappa);
  }
  domain(x, y, z) = buffer_value;
}

template <std::size_t level = nlev> void coarsen_domains(Domain_Type domain) {
  if constexpr (level <= 1) {
    return;
  } else {
    using Offset_Type = std::array<DataType, Dim>;
    constexpr std::array<Offset_Type, 1> offsets{};
    constexpr std::array<DataType, 1> values{1};

    auto &dest = domain.template get_domain<level - 1>();
    auto &src = domain.template get_domain<level>();
    level_transition::coarsening(dest, src, values, offsets);
    coarsen_domains<level - 1>(domain);
  }
}

int main(int argc, char *argv[]) {

  // epsilon_r = 1.0;

  using OffsetType = std::array<int, Dim>;

  if (argc < 3) {
    std::cout
        << "Usage: ./this_program in_file out_file x_min y_min z_min num_iters"
        << std::endl;
    return 0;
  }
  // int num_iters = std::stoi(argv[3]);

#ifdef DEBUGMODE
  sycl::cpu_selector selector;
#else
  sycl::gpu_selector selector;
#endif

  sycl::queue q{selector,
                sycl::property_list{sycl::property::queue::in_order{}}};

  std::string filename_in{argv[1]};
  std::string filename_out{argv[2]};
  std::list<Atom<DataType>> atom_list;
  read_pqr_file(filename_in, atom_list);

  const auto num_atoms = atom_list.size();

  std::vector<Atom<DataType>> atoms_vector;
  atoms_vector.reserve(num_atoms);

  auto x_min = std::stod(argv[3]);
  auto y_min = std::stod(argv[4]);
  auto z_min = std::stod(argv[5]);
  int iter_num = std::stof(argv[6]);

  // std::printf("The threshold is %f\n", thresh);

  for (auto &atom : atom_list) {
    atom.Position[0] -= x_min;
    atom.Position[1] -= y_min;
    atom.Position[2] -= z_min;
    // atom.radius += ionradius;
    atoms_vector.push_back(atom);
  }

  Atom<DataType> *atoms_device =
      sycl::malloc_device<Atom<DataType>>(atom_list.size(), q);

  q.memcpy(atoms_device, atoms_vector.data(),
           atoms_vector.size() * sizeof(Atom<DataType>))
      .wait();

  Domain_Type sol(q), lhs_domain1(q), lhs_domain2(q), rhs_domain(q),
      boundary_values(q), epsilonx_map(q), epsilony_map(q), epsilonz_map(q),
      kappa_(q);

  constexpr auto length = Domain_Type::length;

  std::array<OffsetType, 7> offsets_op{{{-1, 0, 0},
                                        {1, 0, 0},
                                        {0, 0, 0},
                                        {0, -1, 0},
                                        {0, 1, 0},
                                        {0, 0, -1},
                                        {0, 0, 1}}};

  std::array<DataType, 7> values_op{
      -1., -1,  6,  -1.,
      -1., -1., -1.}; // Dividing the original operator by the Diagonal
                      // as it is only applied to the right hand side
                      // anyways

  //  Multi_Level_operator diff_operator(Integer<nlev>{}, values_op, offsets_op,
  //                                     box_length, Integer<base_length>{});

  std::array<OffsetType, 1u> offsets_coarse{{{0, 0, 0}}};
  std::array<DataType, 1u> values_coarse{1.};

  Multi_Level_operator coarser(Integer<nlev>{}, values_coarse, offsets_coarse,
                               Integer<base_length>{});

  coarser.print_operator();

  {

    // const DataType epsilon_r = 1;  //Dangerous parameter
    std::cout << "Warning epsilon_r is set to: " << epsilon_r << std::endl;
    auto &boundary_domain = boundary_values.template get_domain<nlev>();
    q.parallel_for(
         sycl::range<2>(std::get<1>(length) + 2, std::get<2>(length) + 2),
         [=](sycl::id<2> I) {
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, I[0], I[1], 0,
                                         grid_step);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, I[0], I[1],
                                         std::get<2>(length) + 1, grid_step);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, 0, I[0], I[1],
                                         grid_step);
           Set_boundary_conditions<nlev>(
               atoms_device, num_atoms, boundary_domain,
               std::get<0>(length) + 1, I[0], I[1], grid_step);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, I[0], 0, I[1],
                                         grid_step);
           Set_boundary_conditions<nlev>(
               atoms_device, num_atoms, boundary_domain, I[0],
               std::get<1>(length) + 1, I[1], grid_step);
         })
        .wait();

    auto &epsilonx_domain = epsilonx_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[0] -= grid_step / 2.;
      find_dots_in_sphere(Atom, epsilonx_domain, grid_step);
    });

    auto &epsilony_domain = epsilony_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[1] -= grid_step / 2.;
      find_dots_in_sphere(Atom, epsilony_domain, grid_step);
    });

    auto &epsilonz_domain = epsilonz_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[2] -= grid_step / 2.;
      find_dots_in_sphere(Atom, epsilonz_domain, grid_step);
    });

    // q.parallel_for(sycl::range<1>(epsilon_domain.num_values),
    //                [=](sycl::id<1> I) {
    //                  epsilon_domain.values_buff[I] != 0
    //                      ? epsilon_domain.values_buff[I] = 0
    //                      : epsilon_domain.values_buff[I] = 1;
    //                })
    //     .wait();

    std::cout << "The size if the atoms vector is: " << atoms_vector.size()
              << std::endl;
    auto &kappa_domain = kappa_.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      auto atom = atoms_device[I];
      atom.radius += 1.5;
      //                       diff_operator.get_offsets(), 1e-2);
      find_dots_in_sphere(atom, kappa_domain, grid_step);
    });

    // Inverting the kappa domain because the original functions marks the
    // points inside the protein with 1.
    q.parallel_for(sycl::range<1>(kappa_domain.num_values), [=](sycl::id<1> I) {
       kappa_domain.values_buff[I] != 0 ? kappa_domain.values_buff[I] = 0
                                        : kappa_domain.values_buff[I] = 1;
     }).wait();

    //  {
    //    std::ofstream outfile{"kappa_map_own.dx"};
    //    kappa_domain.print_dx_to_stream(outfile, x_min, y_min, z_min, 16);
    //  }

    // Now we need to coarsen the kappa map and the epsilon map
    coarsen_domains(kappa_);
    coarsen_domains(epsilonx_map);
    coarsen_domains(epsilony_map);
    coarsen_domains(epsilonz_map);

    q.wait();

    //  auto kappa_coarse = kappa_.template get_domain<1>();
    //  {
    //    std::ofstream outfile{"kappa_coarse_own.dx"};
    //    kappa_coarse.print_dx_to_stream(outfile, x_min, y_min, z_min, 16);
    //  }

    auto &rhs = rhs_domain.template get_domain<nlev>();
    auto &kappa_map = kappa_.template get_domain<nlev>();
    // auto &epsilon_map_ = epsilon_map.template get_domain<nlev>();
    std::array<Domain<Dim, std::get<0>(length), std::get<1>(length),
                      std::get<2>(length)>,
               Dim>
        epsilon_domains{epsilonx_domain, epsilony_domain, epsilonz_domain};

    convolution::PBE_Convolve(rhs, boundary_domain, kappa_map, epsilon_domains,
                              kappa_2, grid_step,
                              static_cast<DataType>(epsilon_r), delta_epsilon);

    // rhs.print_domain();

    //  //  q.wait();

    q.submit([=](sycl::handler &h) {
       h.single_task([=]() {
         for (int I = 0; I < num_atoms; I++) {

           add_charges_to_distribution(
               rhs, atoms_device[I].Position,
               static_cast<DataType>(atoms_device[I].charge / epsilon),
               spacing<DataType, grid_step>{});
         }
       });
     }).wait();

    //  {
    //    std::ofstream outfile{"charges_map_own.dx"};
    //    rhs.print_dx_to_stream(outfile, x_min, y_min, z_min, 16);
    //  }

    auto &defect_p = lhs_domain1.get_domain();
    auto &defect_r = lhs_domain2.get_domain();
    auto &init_guess = sol.get_domain();

    cg_solver::PBE_Solver_CG cg_solver(Float<(DataType)1e-8>{},
                                       sol.template get_domain<1>(), values_op,
                                       offsets_op);

    // Jacobi_Smoother_PBE j_smoother(rhs_domain);
    //
    Gauss_Seidel_PBE j_smoother(rhs_domain);

    V_Cycle_PBE v_cycle(j_smoother, j_smoother, cg_solver, rhs_domain, coarser);

    std::index_sequence<1> num_iters{};
    std::index_sequence<10> smoothing_steps;

    auto *a = &sol;
    auto *b = &lhs_domain1;

    for (int i = 0; i < iter_num; i++) {

      //  cg_solver(init_guess, rhs, kappa_map, epsilon_domains, kappa_2,
      //  grid_step,
      //            epsilon_r, delta_epsilon);

      DataType const residual =
          compute_residual_PBE(rhs_domain.template get_domain<nlev>(),
                               sol.template get_domain<nlev>(),
                               lhs_domain2.template get_domain<nlev>(),
                               kappa_.template get_domain<nlev>(),
                               epsilony_map.template get_domain<nlev>(),
                               epsilony_map.template get_domain<nlev>(),
                               epsilonz_map.template get_domain<nlev>(),
                               kappa_2, grid_step, epsilon_r, delta_epsilon);

      std::cout << "The residual after " << i << " iterations is " << residual
                << std::endl;

      //   std::index_sequence<1> iter_nums{};
      //   j_smoother(Integer<nlev>{}, iter_nums, *a, *b, rhs_domain, kappa_,
      //              epsilonx_map, epsilony_map, epsilonz_map, kappa_2,
      //              grid_step, epsilon_r, delta_epsilon, omega);

      //   std::swap(a, b);

      v_cycle.iteration(sol, lhs_domain1, rhs_domain, epsilonx_map,
                        epsilony_map, epsilonz_map, kappa_, kappa_2, grid_step,
                        epsilon_r, delta_epsilon, omega, num_iters, coarser,
                        smoothing_steps, smoothing_steps);
    }

    // cg_solver::CG_solver_PBE(
    //    //     init_guess, rhs, defect_r, defect_p, kappa_map,
    //    epsilon_domains,
    //    //     kappa_2, static_cast<DataType>(1.),
    //    static_cast<DataType>(epsilon_r),
    //    //     delta_epsilon, diff_operator.get_values(),
    //     diff_operator.get_offsets(), thresh);
    //
    //    // domain::subtract_domains(init_guess, boundary_domain, init_guess);
    //
    //    //  q.wait();
    //
    //    // Jacobi_Smoother_4BE j_smoother(rhs_domain);

    //  std::index_sequence<30> iter_nums{};
    //  j_smoother(Integer<1>{}, iter_nums, sol, lhs_domain1, rhs_domain,
    //  kappa_,
    //             epsilonx_map, epsilony_map, epsilonz_map, kappa_2, grid_step,
    //             epsilon_r, delta_epsilon, omega);

    // sol.get_domain().print_domain();
    //   //
    //   //  Smoothing operator
    //   std::array<DataType, 7u>
    //       values{-omega * 1. / 6., -omega * 1. / 6., -1. + omega,
    //              -omega * 1. / 6., -omega * 1. / 6., -omega * 1 / 6.,
    //              -omega * 1 / 6.}; // Formula S = 1 - D^(-1) L,

    // sol.get_domain().print_domain();
    //  Here I am checking out the previous smoother
    //  Jacobi_Smoother j_smoother(rhs_domain);
    // std::index_sequence<1> num_iters{};
    // j_smoother(Integer<1>{}, num_iters, sol, lhs_domain1, rhs_domain, values,
    //            offsets_op, box_length, omega);

    // sol.get_domain().print_domain();

    // rhs_domain.get_domain().print_domain();

    domain::add_domains(init_guess, boundary_domain, init_guess);

    // sol.get_domain().print_domain();

    std::ofstream outfile{filename_out};
    init_guess.print_dx_to_stream(outfile, x_min, y_min, z_min, box_length);

    //  init_guess.print_domain();
  }

  q.wait();

  // std::cout << "The length is: " << std::get<0>(length) << std::endl;

  return 0;
}
