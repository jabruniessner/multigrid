#include "Convolution.h"
#include "Debye_Hueckel_functions.h"
#include "MultigridDomain.h"
#include "bitshift_lib.h"
#include "create_charge_distribution.h"
#include "cycles.h"
#include "dot_finder.h"
#include "fileio.h"
#include "hipSYCL/sycl/info/device.hpp"
#include "level_transition.h"
#include "pmgc/buildGd.h"
#include "pmgc/buildPd.h"
#include "pmgc/gsd.h"
#include "pmgc/matvecd.h"
#include "scientific_quantities.h"
#include <array>
#include <bit>
#include <cstddef>
#include <fstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace cycles;
using namespace convolution;

constexpr Dimension Dim = 3;
constexpr std::size_t nlev = 2u;
constexpr std::size_t base_length = 8;
constexpr DataType omega = 1.;
constexpr DataType box_length = 16;
constexpr DataType ionic_strength = 0;
constexpr DataType kappa = KappaA(ionic_strength);
constexpr DataType kappa_2 = kappa * kappa;
constexpr DataType ionradius = 1.5;
constexpr DataType grid_step =
    (DataType)box_length / ((DataType)base_length * utils::power_off(2, nlev));
constexpr DataType delta_epsilon =
    (epsilon_p - epsilon_r); // Difference in epsilon
                             //
                             //

constexpr std::size_t ilog2(std::size_t x) { return std::bit_width(x) - 1; }

constexpr std::size_t level_from_length(std::size_t length,
                                        std::size_t problem_length,
                                        std::size_t nlev) {

  std::size_t fac = problem_length / length;
  std::size_t x = ilog2(fac);

  return nlev - x;
};

template <std::size_t nlev = nlev>
using Domain_Type =
    Multigrid_domain<Dim, nlev, base_length, base_length, base_length>;

template <std::size_t N>
using MemFnPtrd_type = decltype(&Domain_Type<nlev>::template get_domain<N, 0>);

template <std::size_t N>
using d_type = std::remove_reference_t<
    std::invoke_result_t<MemFnPtrd_type<N>, Domain_Type<nlev>>>;

template <std::size_t level = nlev>
using map_type = std::function<DataType(const d_type<level>, sycl::id<Dim>)>;

using Domain_Type_upper = decltype(Domain_Type<>::domain_t_v)::domain_t;

template <std::size_t level = nlev>
constexpr auto strides = Domain_Type<level>::length;

// Defining the pmgc length of the level
template <std::size_t level = nlev> int nx = std::get<0>(strides<level>) + 2;

template <std::size_t level = nlev> int ny = std::get<1>(strides<level>) + 2;

template <std::size_t level = nlev> int nz = std::get<2>(strides<level>) + 2;

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

template <Dimension Dim, Length... strides_all, typename... domains,
          std::size_t... Directions>
void compute_center_domain(std::index_sequence<Directions...>,
                           domain::Domain<Dim, strides_all...> &center_domain,
                           domains &...other_domains) {
  static_assert(
      (std::is_same_v<domain::Domain<Dim, strides_all...>, domains> && ...),
      "One of the other domains is not of the right type");

  static_assert(sizeof...(Directions) == sizeof...(other_domains));

  center_domain.q.parallel_for(
      sycl::range<Dim>(strides_all...), [=](sycl::id<Dim> I) {
        ((I[Directions] += center_domain.padding_width), ...);

        auto domain_offset = [=](auto domain, int dir) {
          auto I2 = I;
          I2[dir] -= 1;
          return domain(I2[Directions]...);
        };

        auto domain_no_offset = [=](auto domain, int dir) {
          return (domain(I[Directions]...));
        };

        center_domain(I[Directions]...) = 0;
        center_domain(I[Directions]...) =
            ((domain_no_offset(other_domains, Directions) +
              domain_offset(other_domains, Directions)) +
             ...);
      });
}

template <Dimension Dim, Length... strides_all, typename... domains>
void compute_center_domain(domain::Domain<Dim, strides_all...> &center_domain,
                           domains &...other_domains) {
  compute_center_domain(std::make_index_sequence<Dim>{}, center_domain,
                        other_domains...);
}

int main(int argc, char *argv[]) {

  //  constexpr DataType epsilon_p = 1.0;
  //  constexpr DataType epsilon_r = 1.0;
  //  constexpr DataType delta_epsilon = 0;

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

  auto dev = q.get_device();

  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << std::endl;

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

  // d_type<nlev> boundary_domain(Paddings::PERIODIC, q, 1);

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

    //  int nx = std::get<0>(length) + 2;
    //  int ny = std::get<1>(length) + 2;
    //  int nz = std::get<2>(length) + 2;

    //  int nxc = std::get<0>(length_coarse) + 2;
    //  int nyc = std::get<1>(length_coarse) + 2;
    //  int nzc = std::get<2>(length_coarse) + 2;

    // This is the problematic position

    auto &epsilonc_domain = epsilon_oC_map.template get_domain<nlev>();
    compute_center_domain(epsilonc_domain, epsilonuC_domain, epsilonoN_domain,
                          epsilonoE_domain);

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

    std::array<Domain<Dim, std::get<0>(length), std::get<1>(length),
                      std::get<2>(length)>,
               Dim>
        epsilon_domains{epsilonx2_domain, epsilony2_domain, epsilonz2_domain};

    //  convolution::PBE_Convolve(rhs_domain.get_domain(), boundary_domain,
    //                            kappa_.get_domain(), epsilon_domains, kappa_2,
    //                            grid_step, epsilon_r, delta_epsilon);

    // rhs_domain.get_domain().print_domain();

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

    pmgc::VbuildPb_op7(
        &nx<nlev>, &ny<nlev>, &nz<nlev>, &nx<nlev - 1>, &ny<nlev - 1>,
        &nz<nlev - 1>, (int *)nullptr, (DataType *)nullptr,
        epsilonc_domain.values_buff, epsilonoE_domain.values_buff,
        epsilonoN_domain.values_buff, epsilonuC_domain.values_buff,
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

    pmgc::VbuildG_7(
        &nx<nlev>, &ny<nlev>, &nz<nlev>, &nx<nlev - 1>, &ny<nlev - 1>,
        &nz<nlev - 1>, oPC.template get_domain<nlev - 1>().values_buff,
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
        epsilon_uSW_map.template get_domain<nlev - 1>().values_buff, q);
    // auto &defect_p = lhs_domain1.get_domain();
    // auto &defect_r = lhs_domain2.get_domain();

    DataType *DT_null = nullptr;

    //  int nx = std::get<0>(length) + 2;
    //  int ny = std::get<1>(length) + 2;
    //  int nz = std::get<2>(length) + 2;

    std::cout << "nx: " << nx<nlev> << std::endl;
    std::cout << "ny: " << ny<nlev> << std::endl;
    std::cout << "nz: " << nz<nlev> << std::endl;

    std::cout << "nxc: " << nx<nlev - 1> << std::endl;
    std::cout << "nyc: " << ny<nlev - 1> << std::endl;
    std::cout << "nzc: " << nz<nlev - 1> << std::endl;

    //  int i = 1;

    cycles::Gauss_Seidel_PBE j_smoother{sol};
    //  auto *a = &sol;
    //  auto *b = &lhs_domain1;
    //  std::cout << "Before the iterations: " << std::endl;
    // sol.get_domain().print_domain();

    auto solver = cg_solver::make_solver(Float<1e-8>{},
                                         sol.template get_domain<nlev - 1>());

    //  constexpr std::size_t level =
    //      level_from_length(std::get<0>(d_type<nlev - 3>::length),
    //                        std::get<0>(d_type<nlev>::length), nlev);

    //  std::cout << "The level is: " << level << std::endl;

    // return 0;

    auto cg_map = [=](const auto domain, sycl::id<Dim> I) {
      using d_type_inner = decltype(domain);
      constexpr std::size_t nxc_i = std::get<0>(d_type_inner::length) + 2;
      constexpr std::size_t nyc_i = std::get<1>(d_type_inner::length) + 2;
      constexpr std::size_t nzc_i = std::get<2>(d_type_inner::length) + 2;

      constexpr std::size_t level =
          level_from_length(std::get<0>(d_type_inner::length),
                            std::get<0>(d_type<nlev>::length), nlev);

      if constexpr (level == nlev) {

        return pmgc::matveckernel7<nxc_i, nyc_i, nzc_i>(
            I[0] + 1, I[1] + 1, I[2] + 1,
            epsilon_oC_map.template get_domain<level>().values_buff,
            kappa_.template get_domain<level>().values_buff,
            epsilon_oE_map.template get_domain<level>().values_buff,
            epsilon_oN_map.template get_domain<level>().values_buff,
            epsilon_uC_map.template get_domain<level>().values_buff,
            domain.values_buff);
      } else {

        return pmgc::matveckernel27<nxc_i, nyc_i, nzc_i>(
            I[0] + 1, I[1] + 1, I[2] + 1,
            epsilon_oC_map.template get_domain<level>().values_buff,
            kappa_.template get_domain<level>().values_buff,
            epsilon_oE_map.template get_domain<level>().values_buff,
            epsilon_oN_map.template get_domain<level>().values_buff,
            epsilon_uC_map.template get_domain<level>().values_buff,
            epsilon_oNE_map.template get_domain<level>().values_buff,
            epsilon_oNW_map.template get_domain<level>().values_buff,
            epsilon_uE_map.template get_domain<level>().values_buff,
            epsilon_uW_map.template get_domain<level>().values_buff,
            epsilon_uN_map.template get_domain<level>().values_buff,
            epsilon_uS_map.template get_domain<level>().values_buff,
            epsilon_uNE_map.template get_domain<level>().values_buff,
            epsilon_uNW_map.template get_domain<level>().values_buff,
            epsilon_uSE_map.template get_domain<level>().values_buff,
            epsilon_uSW_map.template get_domain<level>().values_buff,
            domain.values_buff);
      }
    };

    // std::function cg_map_inst = cg_map<nlev-1>;

    //  convolution::Convolve_map(sol2.get_domain(), boundary_domain, cg_map);
    //  sol2.get_domain().print_domain();

    for (int i = 0; i < num_iters + 1; i++) {

      int smoothing_iters = 2;

      // cg_solver(init_guess, rhs, kappa_map, epsilon_domains, kappa_2,
      // m grid_step,
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
      pmgc::Vgsrb7x(&nx<nlev>, &ny<nlev>, &nz<nlev>, (int *)nullptr, DT_null,
                    epsilon_oC_map.get_domain().values_buff,
                    kappa_domain.values_buff, rhs.values_buff,
                    epsilonoE_domain.values_buff, epsilonoN_domain.values_buff,
                    epsilonuC_domain.values_buff, sol.get_domain().values_buff,
                    &smoothing_iters, q);

      //  std::cout << "The sol domain after the gsb is: " << std::endl;
      //  sol.get_domain().print_domain();

      //  auto *values_buff = sol.get_domain().values_buff;

      //  //   // Defect computation
      //  //   // This computes -A, in this case
      //  // std::cout << "Before the convolve the sol2 domain is: " <<
      //  // std::endl;
      //  // sol2.get_domain().print_domain();
      convolution::PBE_Convolve(sol2.get_domain(), sol.get_domain(),
                                kappa_.get_domain(), epsilon_domains, kappa_2,
                                grid_step, epsilon_r, delta_epsilon);

      //  std::cout << "After the convolution the sol2 domain is: " <<
      //  std::endl;

      //  // sol2.get_domain().print_domain();

      //  //   //   // this needs to be add, because the convolve returns the
      //  //   negative
      add_domains(sol2.get_domain(), sol2.get_domain(),
                  rhs_domain.get_domain());

      //   std::cout << "The right hand side domain is: " << std::endl;
      //   rhs_domain.template get_domain<nlev>().print_domain();

      //   std::cout << "After the addition the sol2 domain is: " <<
      // std::endl;
      //   sol2.template get_domain<nlev>().print_domain();

      // Restriction
      pmgc::Vrestrc2(&nx<nlev>, &ny<nlev>, &nz<nlev>, &nx<nlev - 1>,
                     &ny<nlev - 1>, &nz<nlev - 1>,
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

      q.wait();
      //   // rhs_domain.template get_domain<nlev - 1>().print_domain();

      int itmax = 1000;

      q.memset(sol2.template get_domain<nlev - 1>().values_buff, 0,
               sol2.template get_domain<nlev - 1>().num_values *
                   sizeof(DataType));

      //  // Coarse grid
      //  pmgc::Vgsrb27x(
      //      &nxc, &nyc, &nzc, (int *)nullptr, (DataType *)nullptr,
      //      epsilon_oC_map.template get_domain<nlev - 1>().values_buff,
      //      kappa_.template get_domain<nlev - 1>().values_buff,
      //      rhs_domain.template get_domain<nlev - 1>().values_buff,
      //      epsilon_oE_map.template get_domain<nlev - 1>().values_buff,
      //      epsilon_oN_map.template get_domain<nlev - 1>().values_buff,
      //      epsilon_uC_map.template get_domain<nlev - 1>().values_buff,
      //      epsilon_oNE_map.template get_domain<nlev - 1>().values_buff,
      //      epsilon_oNW_map.template get_domain<nlev - 1>().values_buff,
      //      epsilon_uE_map.template get_domain<nlev - 1>().values_buff,
      //      epsilon_uW_map.template get_domain<nlev - 1>().values_buff,
      //      epsilon_uN_map.template get_domain<nlev - 1>().values_buff,
      //      epsilon_uS_map.template get_domain<nlev - 1>().values_buff,
      //      epsilon_uNE_map.template get_domain<nlev - 1>().values_buff,
      //      epsilon_uNW_map.template get_domain<nlev - 1>().values_buff,
      //      epsilon_uSE_map.template get_domain<nlev - 1>().values_buff,
      //      epsilon_uSW_map.template get_domain<nlev - 1>().values_buff,
      //      sol2.template get_domain<nlev - 1>().values_buff, &itmax, q);

      //  std::cout << "After the 27x Gauss-Seidel: " << std::endl;
      //  sol2.template get_domain<nlev - 1>().print_domain();

      solver(sol2.template get_domain<nlev - 1>(),
             rhs_domain.template get_domain<nlev - 1>(), cg_map);

      q.memset(sol2.template get_domain<nlev>().values_buff, 0,
               sol2.template get_domain<nlev>().num_values * sizeof(DataType));

      //  //  std::cout << "Before the interpolate the sol2 is: " <<
      //  // std::endl;
      //  //  sol2.template get_domain<nlev>().print_domain();

      pmgc::VinterpPMG2(&nx<nlev - 1>, &nx<nlev - 1>, &nx<nlev - 1>, &nx<nlev>,
                        &ny<nlev>, &nz<nlev>,
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

      // std::cout << "After the prolongation the domain is: " << std::endl;
      //   sol2.template get_domain<nlev>().print_domain();

      // Adding correction to the current guess
      add_domains(sol.get_domain(), sol2.get_domain(), sol.get_domain());

      //  //   // Postsmoothing
      pmgc::Vgsrb7x(&nx<nlev>, &ny<nlev>, &nz<nlev>, (int *)nullptr, DT_null,
                    epsilon_oC_map.get_domain().values_buff,
                    kappa_domain.values_buff, rhs.values_buff,
                    epsilonoE_domain.values_buff, epsilonoN_domain.values_buff,
                    epsilonuC_domain.values_buff, sol.get_domain().values_buff,
                    &smoothing_iters, q);
    }
  }

  std::ofstream outfile{"single_atom_potential.dx"};

  sol.get_domain().print_dx_to_stream(outfile, x_min, y_min, z_min, box_length);

  return 0;
}
