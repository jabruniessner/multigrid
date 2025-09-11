#include "Convolution.h"
#include "Debye_Hueckel_functions.h"
#include "MultigridDomain.h"
#include "bitshift_lib.h"
#include "create_charge_distribution.h"
#include "cycles.h"
#include "dot_finder.h"
#include "fileio.h"
#include "level_transition.h"
#include "pmgc/buildGd.h"
#include "pmgc/buildPd.h"
#include "pmgc/gsd.h"
#include "pmgc/matvecd.h"
#include "scientific_quantities.h"
#include <array>
#include <cstddef>
#include <string>
#include <vector>

using namespace cycles;
using namespace convolution;

constexpr Dimension Dim = 3;
constexpr std::size_t nlev = 2u;
constexpr std::size_t base_length = 2;
constexpr DataType omega = 1.;
constexpr DataType box_length = 16;
constexpr DataType ionic_strength = 0.15;
constexpr DataType kappa = KappaA(ionic_strength);
constexpr DataType kappa_2 = kappa * kappa;
constexpr DataType ionradius = 1.5;
constexpr DataType grid_step =
    (DataType)box_length / ((DataType)base_length * utils::power_off(2, nlev));
constexpr DataType delta_epsilon =
    (epsilon_p - epsilon_r); // Difference in epsilon
                             //

template <std::size_t nlev = nlev>
using Domain_Type =
    Multigrid_domain<Dim, nlev, base_length, base_length, base_length>;

using Domain_Type_upper = decltype(Domain_Type<>::domain_t_v)::domain_t;

DataType sqr(double val) { return val * val; }

template <std::size_t level = nlev>
void Set_boundary_conditions(Atom<DataType> *atoms, std::size_t num_atoms,
                             const Domain_Type_upper &domain, std::size_t x,
                             std::size_t y, std::size_t z) {
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

template <std::size_t level = nlev> void coarsen_domains(Domain_Type<> domain) {
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

  std::cout << "The value of grid_step is: " << grid_step << std::endl;

  using OffsetType = std::array<int, Dim>;

  if (argc < 3) {
    std::cout
        << "Usage: ./this_program in_file out_file x_min y_min z_min num_iters"
        << std::endl;
    return 0;
  }

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
  int num_iters = std::stoi(argv[6]);

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

  Domain_Type<> sol(q), sol2(q), lhs_domain1(q), lhs_domain2(q), rhs_domain(q),
      epsilon_oC_map(q), epsilon_oE_map(q), epsilon_oN_map(q),
      epsilon_uC_map(q), epsilon_oNE_map(q), epsilon_oNW_map(q),
      epsilon_uE_map(q), epsilon_uW_map(q), epsilon_uN_map(q),
      epsilon_uS_map(q), epsilon_uNE_map(q), epsilon_uNW_map(q),
      epsilon_uSE_map(q), epsilon_uSW_map(q),

      epsx_map(q), epsy_map(q), epsz_map(q), kappa_(q), kappa_second_map(q),
      oC(q), oE(q), oN(q), uC(q), oPC(q), oPN(q), oPS(q), oPE(q), oPW(q),
      oPNE(q), oPNW(q), oPSE(q), oPSW(q), uPC(q), uPN(q), uPS(q), uPE(q),
      uPW(q), uPNE(q), uPNW(q), uPSE(q), uPSW(q), dPC(q), dPN(q), dPS(q),
      dPE(q), dPW(q), dPNE(q), dPNW(q), dPSE(q), dPSW(q);

  q.fill(epsilon_uC_map.get_domain().values_buff,
         (DataType)epsilon_r / (grid_step * grid_step),
         epsilon_uC_map.get_domain().num_values);

  q.fill(epsilon_oN_map.get_domain().values_buff,
         (DataType)epsilon_r / (grid_step * grid_step),
         epsilon_oN_map.get_domain().num_values);

  q.fill(epsilon_oE_map.get_domain().values_buff,
         (DataType)epsilon_r / (grid_step * grid_step),
         epsilon_oE_map.get_domain().num_values);

  constexpr auto length = Domain_Type<>::length;
  constexpr auto length_coarse = Domain_Type<nlev - 1>::length;

  {
    auto &boundary_domain = sol.template get_domain<nlev>();
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

    auto &boundary_domain2 = sol2.template get_domain<nlev>();
    q.parallel_for(
         sycl::range<2>(std::get<1>(length) + 2, std::get<2>(length) + 2),
         [=](sycl::id<2> I) {
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain2, I[0], I[1], 0);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain2, I[0], I[1],
                                         std::get<2>(length) + 1);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain2, 0, I[0], I[1]);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain2,
                                         std::get<0>(length) + 1, I[0], I[1]);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain2, I[0], 0, I[1]);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain2, I[0],
                                         std::get<1>(length) + 1, I[1]);
         })
        .wait();

    auto &epsilonuC_domain = epsilon_uC_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[0] -= 0.5 * grid_step;
      find_dots_in_sphere(Atom, epsilonuC_domain,
                          static_cast<DataType>(grid_step),
                          (DataType)epsilon_p / (grid_step * grid_step));
    });

    auto &epsilonx2_domain = epsx_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[0] -= 0.5 * grid_step;
      find_dots_in_sphere(Atom, epsilonx2_domain,
                          static_cast<DataType>(grid_step));
    });

    // std::cout << "The x-domain is: " << std::endl;
    // epsilonx_domain.print_domain();

    auto &epsilonoN_domain = epsilon_oN_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[1] -= 0.5 * grid_step;
      find_dots_in_sphere(Atom, epsilonoN_domain,
                          static_cast<DataType>(grid_step),
                          (DataType)epsilon_p / (grid_step * grid_step));
    });

    auto &epsilony2_domain = epsy_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[1] -= 0.5 * grid_step;
      find_dots_in_sphere(Atom, epsilony2_domain,
                          static_cast<DataType>(grid_step));
    });

    // std::cout << "The y-domain is: " << std::endl;
    // epsilony_domain.print_domain();

    auto &epsilonoE_domain = epsilon_oE_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[2] -= 0.5 * grid_step;
      find_dots_in_sphere(Atom, epsilonoE_domain,
                          static_cast<DataType>(grid_step),
                          (DataType)epsilon_p / (grid_step * grid_step));
    });

    auto &epsilonz2_domain = epsz_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[2] -= 0.5 * grid_step;
      find_dots_in_sphere(Atom, epsilonz2_domain,
                          static_cast<DataType>(grid_step));
    });

    //  std::cout << "The z-domain is: " << std::endl;
    //  epsilonz_domain.print_domain();

    int nx = std::get<0>(length) + 2;
    int ny = std::get<1>(length) + 2;
    int nz = std::get<2>(length) + 2;

    int nxc = std::get<0>(length_coarse) + 2;
    int nyc = std::get<1>(length_coarse) + 2;
    int nzc = std::get<2>(length_coarse) + 2;

    auto &epsilonc_domain = epsilon_oC_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<3>(nx, ny, nz), [=](sycl::id<3> I) {
      epsilonc_domain(I[0], I[1], I[2]) =
          epsilonuC_domain(I[0], I[1], I[2]) +
          epsilonoN_domain(I[0], I[1], I[2]) +
          epsilonoE_domain(I[0], I[1], I[2]) +
          epsilonuC_domain(I[0] - 1, I[1], I[2]) +
          epsilonoN_domain(I[0], I[1] - 1, I[2]) +
          epsilonoE_domain(I[0], I[1], I[2] - 1);
    });

    q.wait();

    auto &kappa_domain = kappa_.template get_domain<nlev>();
    //  q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
    //    auto atom = atoms_device[I];
    //    atom.radius += 1.5;
    //    //                      diff_operator.get_offsets(), 1e-2);
    //    find_dots_in_sphere(atom, kappa_domain,
    //    static_cast<DataType>(grid_step));
    //  });

    auto &kappa_domain2 = kappa_second_map.template get_domain<nlev>();

    //  q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
    //    auto atom = atoms_device[I];
    //    atom.radius += 1.5;
    //    //                      diff_operator.get_offsets(), 1e-2);
    //    find_dots_in_sphere(atom, kappa_domain2,
    //                        static_cast<DataType>(grid_step));
    //  });

    // Inverting the kappa domain because the original functions marks the
    // points inside the protein with 1. And assigning the right value
    // q.parallel_for(sycl::range<1>(kappa_domain.num_values), [=](sycl::id<1>
    // I) {
    //    kappa_domain.values_buff[I] != 0
    //        ? kappa_domain.values_buff[I] = 0
    //        : kappa_domain.values_buff[I] = kappa_2 * epsilon_r;
    //  }).wait();

    //  std::cout << "The kappa domain is: " << std::endl;
    //  // kappa_domain.print_domain();

    //  q.parallel_for(sycl::range<1>(kappa_domain2.num_values),
    //                 [=](sycl::id<1> I) {
    //                   kappa_domain2.values_buff[I] != 0
    //                       ? kappa_domain2.values_buff[I] = 0
    //                       : kappa_domain2.values_buff[I] = 1;
    //                 })
    //      .wait();

    auto &rhs = rhs_domain.template get_domain<nlev>();

    q.submit([=](sycl::handler &h) {
       h.single_task([=]() {
         for (int I = 0; I < num_atoms; I++)
           add_charges_to_distribution(
               rhs, atoms_device[I].Position,
               static_cast<DataType>(atoms_device[I].charge / epsilon),
               spacing<DataType, static_cast<DataType>(grid_step)>{});
       });
     }).wait();

    pmgc::VbuildPb_op7(&nx, &ny, &nz, &nxc, &nyc, &nzc, (int *)nullptr,
                       (DataType *)nullptr,
                       epsilon_oC_map.get_domain().values_buff,
                       epsilon_oE_map.get_domain().values_buff,
                       epsilon_oN_map.get_domain().values_buff,
                       epsilon_uC_map.get_domain().values_buff,
                       oPC.template get_domain<nlev - 1>().values_buff,
                       oPN.template get_domain<nlev - 1>().values_buff,
                       oPS.template get_domain<nlev - 1>().values_buff,
                       oPE.template get_domain<nlev - 1>().values_buff,
                       oPW.template get_domain<nlev - 1>().values_buff,
                       oPNE.template get_domain<nlev - 1>().values_buff,
                       oPNW.template get_domain<nlev - 1>().values_buff,
                       oPSE.template get_domain<nlev - 1>().values_buff,
                       oPSW.template get_domain<nlev - 1>().values_buff,
                       uPC.template get_domain<nlev - 1>().values_buff,
                       uPN.template get_domain<nlev - 1>().values_buff,
                       uPS.template get_domain<nlev - 1>().values_buff,
                       uPE.template get_domain<nlev - 1>().values_buff,
                       uPW.template get_domain<nlev - 1>().values_buff,
                       uPNE.template get_domain<nlev - 1>().values_buff,
                       uPNW.template get_domain<nlev - 1>().values_buff,
                       uPSE.template get_domain<nlev - 1>().values_buff,
                       uPSW.template get_domain<nlev - 1>().values_buff,
                       dPC.template get_domain<nlev - 1>().values_buff,
                       dPN.template get_domain<nlev - 1>().values_buff,
                       dPS.template get_domain<nlev - 1>().values_buff,
                       dPE.template get_domain<nlev - 1>().values_buff,
                       dPW.template get_domain<nlev - 1>().values_buff,
                       dPNE.template get_domain<nlev - 1>().values_buff,
                       dPNW.template get_domain<nlev - 1>().values_buff,
                       dPSE.template get_domain<nlev - 1>().values_buff,
                       dPSW.template get_domain<nlev - 1>().values_buff, q);

    pmgc::VbuildG_7(&nx, &ny, &nz, &nxc, &nyc, &nzc,
                    oPC.template get_domain<nlev - 1>().values_buff,
                    oPN.template get_domain<nlev - 1>().values_buff,
                    oPS.template get_domain<nlev - 1>().values_buff,
                    oPE.template get_domain<nlev - 1>().values_buff,
                    oPW.template get_domain<nlev - 1>().values_buff,
                    oPNE.template get_domain<nlev - 1>().values_buff,
                    oPNW.template get_domain<nlev - 1>().values_buff,
                    oPSE.template get_domain<nlev - 1>().values_buff,
                    oPSW.template get_domain<nlev - 1>().values_buff,
                    uPC.template get_domain<nlev - 1>().values_buff,
                    uPN.template get_domain<nlev - 1>().values_buff,
                    uPS.template get_domain<nlev - 1>().values_buff,
                    uPE.template get_domain<nlev - 1>().values_buff,
                    uPW.template get_domain<nlev - 1>().values_buff,
                    uPNE.template get_domain<nlev - 1>().values_buff,
                    uPNW.template get_domain<nlev - 1>().values_buff,
                    uPSE.template get_domain<nlev - 1>().values_buff,
                    uPSW.template get_domain<nlev - 1>().values_buff,
                    dPC.template get_domain<nlev - 1>().values_buff,
                    dPN.template get_domain<nlev - 1>().values_buff,
                    dPS.template get_domain<nlev - 1>().values_buff,
                    dPE.template get_domain<nlev - 1>().values_buff,
                    dPW.template get_domain<nlev - 1>().values_buff,
                    dPNE.template get_domain<nlev - 1>().values_buff,
                    dPNW.template get_domain<nlev - 1>().values_buff,
                    dPSE.template get_domain<nlev - 1>().values_buff,
                    dPSW.template get_domain<nlev - 1>().values_buff,
                    epsilon_oC_map.template get_domain<nlev>().values_buff,
                    epsilon_oE_map.template get_domain<nlev>().values_buff,
                    epsilon_oN_map.template get_domain<nlev>().values_buff,
                    epsilon_uC_map.template get_domain<nlev>().values_buff,
                    epsilon_oC_map.template get_domain<nlev - 1>().values_buff,
                    epsilon_oE_map.template get_domain<nlev - 1>().values_buff,
                    epsilon_oN_map.template get_domain<nlev - 1>().values_buff,
                    epsilon_uC_map.template get_domain<nlev - 1>().values_buff,
                    epsilon_oNE_map.template get_domain<nlev - 1>().values_buff,
                    epsilon_oNW_map.template get_domain<nlev - 1>().values_buff,
                    epsilon_uE_map.template get_domain<nlev - 1>().values_buff,
                    epsilon_uW_map.template get_domain<nlev - 1>().values_buff,
                    epsilon_uN_map.template get_domain<nlev - 1>().values_buff,
                    epsilon_uS_map.template get_domain<nlev - 1>().values_buff,
                    epsilon_uNE_map.template get_domain<nlev - 1>().values_buff,
                    epsilon_uNW_map.template get_domain<nlev - 1>().values_buff,
                    epsilon_uSE_map.template get_domain<nlev - 1>().values_buff,
                    epsilon_uSW_map.template get_domain<nlev - 1>().values_buff,
                    q);
    // auto &defect_p = lhs_domain1.get_domain();
    // auto &defect_r = lhs_domain2.get_domain();

    DataType *DT_null = nullptr;

    //  int nx = std::get<0>(length) + 2;
    //  int ny = std::get<1>(length) + 2;
    //  int nz = std::get<2>(length) + 2;

    std::cout << "nx: " << nx << std::endl;
    std::cout << "ny: " << ny << std::endl;
    std::cout << "nz: " << nz << std::endl;

    int i = 1;

    cycles::Gauss_Seidel_PBE j_smoother{sol};
    //  auto *a = &sol;
    //  auto *b = &lhs_domain1;
    //  std::cout << "Before the iterations: " << std::endl;
    //  sol.get_domain().print_domain();

    std::array<Domain<Dim, std::get<0>(length), std::get<1>(length),
                      std::get<2>(length)>,
               Dim>
        epsilon_domains{epsilonuC_domain, epsilonoN_domain, epsilonoE_domain};

    for (int i = 0; i < num_iters + 1; i++) {

      int smoothing_iters = 2;

      // cg_solver(init_guess, rhs, kappa_map, epsilon_domains, kappa_2,
      // grid_step,
      //           epsilon_r, delta_epsilon);

      DataType const residual =
          compute_residual_PBE(rhs_domain.template get_domain<nlev>(),
                               sol.template get_domain<nlev>(),
                               lhs_domain2.template get_domain<nlev>(),
                               kappa_second_map.template get_domain<nlev>(),
                               epsx_map.template get_domain<nlev>(),
                               epsy_map.template get_domain<nlev>(),
                               epsz_map.template get_domain<nlev>(), kappa_2,
                               grid_step, epsilon_r, delta_epsilon);

      std::cout << "The residual after " << i << " iterations is: " << residual
                << std::endl;

      if (i >= num_iters)
        break;

      // Presmoothing
      pmgc::Vgsrb7x(&nx, &ny, &nz, (int *)nullptr, DT_null,
                    epsilon_oC_map.get_domain().values_buff,
                    kappa_domain.values_buff, rhs.values_buff,
                    epsilonoE_domain.values_buff, epsilonoN_domain.values_buff,
                    epsilonuC_domain.values_buff, sol.get_domain().values_buff,
                    &smoothing_iters, q);

      // Defect computation
      // This computes -A, in this case
      convolution::PBE_Convolve(sol2.get_domain(), sol.get_domain(),
                                kappa_.get_domain(), epsilon_domains, kappa_2,
                                grid_step, epsilon_r, delta_epsilon);

      // this needs to be add, because the convolve returns the negative
      add_domains(sol2.get_domain(), sol2.get_domain(),
                  rhs_domain.get_domain());

      // Restriction
      pmgc::Vrestrc2(&nx, &ny, &nz, &nxc, &nyc, &nzc,
                     sol2.get_domain().values_buff,
                     rhs_domain.template get_domain<nlev - 1>().values_buff,
                     oPC.template get_domain<nlev - 1>().values_buff,
                     oPN.template get_domain<nlev - 1>().values_buff,
                     oPS.template get_domain<nlev - 1>().values_buff,
                     oPE.template get_domain<nlev - 1>().values_buff,
                     oPW.template get_domain<nlev - 1>().values_buff,
                     oPNE.template get_domain<nlev - 1>().values_buff,
                     oPNW.template get_domain<nlev - 1>().values_buff,
                     oPSE.template get_domain<nlev - 1>().values_buff,
                     oPSW.template get_domain<nlev - 1>().values_buff,
                     uPC.template get_domain<nlev - 1>().values_buff,
                     uPN.template get_domain<nlev - 1>().values_buff,
                     uPS.template get_domain<nlev - 1>().values_buff,
                     uPE.template get_domain<nlev - 1>().values_buff,
                     uPW.template get_domain<nlev - 1>().values_buff,
                     uPNE.template get_domain<nlev - 1>().values_buff,
                     uPNW.template get_domain<nlev - 1>().values_buff,
                     uPSE.template get_domain<nlev - 1>().values_buff,
                     uPSW.template get_domain<nlev - 1>().values_buff,
                     dPC.template get_domain<nlev - 1>().values_buff,
                     dPN.template get_domain<nlev - 1>().values_buff,
                     dPS.template get_domain<nlev - 1>().values_buff,
                     dPE.template get_domain<nlev - 1>().values_buff,
                     dPW.template get_domain<nlev - 1>().values_buff,
                     dPNE.template get_domain<nlev - 1>().values_buff,
                     dPNW.template get_domain<nlev - 1>().values_buff,
                     dPSE.template get_domain<nlev - 1>().values_buff,
                     dPSW.template get_domain<nlev - 1>().values_buff, q);

      int itmax = 100;

      q.memset(sol2.template get_domain<nlev - 1>().values_buff, 0,
               sol2.template get_domain<nlev - 1>().num_values);

      // Coarse grid
      pmgc::Vgsrb27x(
          &nxc, &nyc, &nzc, (int *)nullptr, (DataType *)nullptr,
          epsilon_oC_map.template get_domain<nlev - 1>().values_buff,
          kappa_.template get_domain<nlev - 1>().values_buff,
          rhs_domain.template get_domain<nlev - 1>().values_buff,
          epsilon_oE_map.template get_domain<nlev - 1>().values_buff,
          epsilon_oN_map.template get_domain<nlev - 1>().values_buff,
          epsilon_uC_map.template get_domain<nlev - 1>().values_buff,
          epsilon_oNE_map.template get_domain<nlev - 1>().values_buff,
          epsilon_oNW_map.template get_domain<nlev - 1>().values_buff,
          epsilon_uE_map.template get_domain<nlev - 1>().values_buff,
          epsilon_uW_map.template get_domain<nlev - 1>().values_buff,
          epsilon_uN_map.template get_domain<nlev - 1>().values_buff,
          epsilon_uS_map.template get_domain<nlev - 1>().values_buff,
          epsilon_uNE_map.template get_domain<nlev - 1>().values_buff,
          epsilon_uNW_map.template get_domain<nlev - 1>().values_buff,
          epsilon_uSE_map.template get_domain<nlev - 1>().values_buff,
          epsilon_uSW_map.template get_domain<nlev - 1>().values_buff,
          sol2.template get_domain<nlev - 1>().values_buff, &itmax, q);

      pmgc::VinterpPMG2(&nxc, &nxc, &nxc, &nx, &ny, &nz,
                        sol2.get_domain<nlev - 1>().values_buff,
                        sol2.get_domain().values_buff,
                        oPC.template get_domain<nlev - 1>().values_buff,
                        oPN.template get_domain<nlev - 1>().values_buff,
                        oPS.template get_domain<nlev - 1>().values_buff,
                        oPE.template get_domain<nlev - 1>().values_buff,
                        oPW.template get_domain<nlev - 1>().values_buff,
                        oPNE.template get_domain<nlev - 1>().values_buff,
                        oPNW.template get_domain<nlev - 1>().values_buff,
                        oPSE.template get_domain<nlev - 1>().values_buff,
                        oPSW.template get_domain<nlev - 1>().values_buff,
                        uPC.template get_domain<nlev - 1>().values_buff,
                        uPN.template get_domain<nlev - 1>().values_buff,
                        uPS.template get_domain<nlev - 1>().values_buff,
                        uPE.template get_domain<nlev - 1>().values_buff,
                        uPW.template get_domain<nlev - 1>().values_buff,
                        uPNE.template get_domain<nlev - 1>().values_buff,
                        uPNW.template get_domain<nlev - 1>().values_buff,
                        uPSE.template get_domain<nlev - 1>().values_buff,
                        uPSW.template get_domain<nlev - 1>().values_buff,
                        dPC.template get_domain<nlev - 1>().values_buff,
                        dPN.template get_domain<nlev - 1>().values_buff,
                        dPS.template get_domain<nlev - 1>().values_buff,
                        dPE.template get_domain<nlev - 1>().values_buff,
                        dPW.template get_domain<nlev - 1>().values_buff,
                        dPNE.template get_domain<nlev - 1>().values_buff,
                        dPNW.template get_domain<nlev - 1>().values_buff,
                        dPSE.template get_domain<nlev - 1>().values_buff,
                        dPSW.template get_domain<nlev - 1>().values_buff, q);

      // Adding correction to the current guess
      add_domains(sol.get_domain(), sol2.get_domain(), sol.get_domain());

      // Postsmoothing
      pmgc::Vgsrb7x(&nx, &ny, &nz, (int *)nullptr, DT_null,
                    epsilon_oC_map.get_domain().values_buff,
                    kappa_domain.values_buff, rhs.values_buff,
                    epsilonoE_domain.values_buff, epsilonoN_domain.values_buff,
                    epsilonuC_domain.values_buff, sol.get_domain().values_buff,
                    &smoothing_iters, q);
    }
  }

  return 0;
}
