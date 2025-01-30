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
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace cycles;
using namespace convolution;

constexpr Dimension Dim = 3;
constexpr std::size_t nlev = 4u;
constexpr std::size_t base_length = 6;
constexpr DataType omega = 4. / 5.;
constexpr DataType box_length = 96;
constexpr double ionic_strength = 0.005;
constexpr DataType kappa = KappaA(ionic_strength);
constexpr DataType kappa_2 = kappa * kappa;
constexpr DataType ionradius = 1.5;
constexpr DataType delta_epsilon =
    (epsilon_p - epsilon_r); // Difference in epsilon
                             //

using Domain_Type =
    Multigrid_domain<Dim, nlev, base_length, base_length, base_length>;

using Domain_Type_upper = decltype(Domain_Type::domain_t_v)::domain_t;

DataType sqr(double val) { return val * val; }

template <std::size_t level = nlev>
void Set_boundary_conditions(Atom<DataType> *atoms, std::size_t num_atoms,
                             const Domain_Type_upper &domain, std::size_t x,
                             std::size_t y, std::size_t z) {
  DataType buffer_value = 0;
  for (int i = 0; i < num_atoms; i++) {
    const DataType distance = std::sqrt(sqr(x - atoms[i].Position[0]) +
                                        sqr(y - atoms[i].Position[1]) +
                                        sqr(z - atoms[i].Position[2]));

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

  using OffsetType = std::array<int, Dim>;

  if (argc < 3) {
    std::cout
        << "Usage: ./this_program in_file out_file x_min y_min z_min num_iters"
        << std::endl;
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
  auto num_iters = std::stod(argv[6]);

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
      boundary_values(q), epsilon_map(q), kappa_(q);

  constexpr auto &length = Domain_Type::length;

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
                      // as it is only applied to the right hand side anyways

  Multi_Level_operator diff_operator(Integer<nlev>{}, values_op, offsets_op,
                                     box_length, Integer<base_length>{});

  {
    auto &boundary_domain = boundary_values.template get_domain<nlev>();
    q.parallel_for(
         sycl::range<2>(std::get<1>(length) + 2, std::get<2>(length) + 2),
         [=](sycl::id<2> I) {
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, I[0], I[1], 0);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, I[0], I[1],
                                         std::get<2>(length) + 1);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, 0, I[0], I[1]);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain,
                                         std::get<0>(length) + 1, I[0], I[1]);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, I[0], 0, I[1]);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, I[0],
                                         std::get<1>(length) + 1, I[1]);
         })
        .wait();

    auto &epsilon_domain = epsilon_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      find_dots_in_sphere(atoms_device[I], epsilon_domain,
                          static_cast<DataType>(1.));
    });

    // q.parallel_for(sycl::range<1>(epsilon_domain.num_values),
    //                [=](sycl::id<1> I) {
    //                  epsilon_domain.values_buff[I] != 0
    //                      ? epsilon_domain.values_buff[I] = 0
    //                      : epsilon_domain.values_buff[I] = 1;
    //                })
    //     .wait();
    auto &kappa_domain = kappa_.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      auto atom = atoms_device[I];
      atom.radius += 1.5;
      find_dots_in_sphere(atom, kappa_domain, static_cast<DataType>(1.));
    });

    // Inverting the kappa domain because the original functions marks the
    // points inside the protein with 1.
    q.parallel_for(sycl::range<1>(kappa_domain.num_values), [=](sycl::id<1> I) {
       kappa_domain.values_buff[I] != 0 ? kappa_domain.values_buff[I] = 0
                                        : kappa_domain.values_buff[I] = 1;
     }).wait();

    // Now we need to coarsen the kappa map and the epsilon map
    coarsen_domains(kappa_);
    coarsen_domains(epsilon_map);

    auto &rhs = rhs_domain.template get_domain<nlev>();
    auto &kappa_map = kappa_.template get_domain<nlev>();
    auto &epsilon_map_ = epsilon_map.template get_domain<nlev>();
    convolution::PBE_Convolve(
        rhs, boundary_domain, kappa_map, epsilon_map_, kappa_2, 1., epsilon_r,
        delta_epsilon, diff_operator.get_values(), diff_operator.get_offsets());

    q.wait();

    //   q.submit([=](sycl::handler &h) {
    //      h.single_task([=]() {
    //        for (int I = 0; I < num_atoms; I++)
    //          add_charges_to_distribution(rhs, atoms_device[I].Position,
    //                                      atoms_device[I].charge / epsilon,
    //                                      spacing<DataType, 1.>{});
    //      });
    //    }).wait();

    auto &defect_p = lhs_domain1.get_domain();
    auto &defect_r = lhs_domain2.get_domain();
    auto &init_guess = sol.get_domain();

    // cg_solver::CG_solver_PBE(
    //     init_guess, rhs, defect_r, defect_p, kappa_map, epsilon_map_,
    //     kappa_2, static_cast<DataType>(1.), epsilon_r, delta_epsilon,
    //     diff_operator.get_values(), diff_operator.get_offsets(), num_iters);
    cg_solver::CG_solver(init_guess, rhs, defect_r, defect_p,
                         diff_operator.get_values(),
                         diff_operator.get_offsets(), 1e-2);
  }

  std::cout << "The length is: " << std::get<0>(length) << std::endl;

  return 0;
}
