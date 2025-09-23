#include "Atom_types.h"
#include "Debye_Hueckel_functions.h"
#include "MultigridDomain.h"
#include "create_charge_distribution.h"
#include "dot_finder.h"
#include "pmgc/buildPd.h"
#include "predefinitions.h"
#include "scientific_quantities.h"
#include "utils.h"
#include <iostream>
#include <sycl/sycl.hpp>

#ifndef PBE_PMGC_CLASS_H
#define PBE_PMGC_CLASS_H

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

template <std::size_t base_length, std::size_t nlev, DataType box_length>
class PBE_linear_problem {

  constexpr static Dimension Dim = 3;

  template <std::size_t level = nlev>
  using Domain_Type =
      multigrid_domain::Multigrid_domain<Dim, level, base_length, base_length,
                                         base_length>;

  template <std::size_t N>
  using MemFnPtrd_type =
      decltype(&Domain_Type<nlev>::template get_domain<N, 0>);

  template <std::size_t N>
  using d_type = std::remove_reference_t<
      std::invoke_result_t<MemFnPtrd_type<N>, Domain_Type<nlev>>>;

  using Domain_Type_upper = decltype(Domain_Type<>::domain_t_v)::domain_t;

  PBE_linear_problem(DataType ionic_strength, DataType ion_radius,
                     sycl::queue &q)
      : ion_radius(ion_radius), ionic_strength(ionic_strength), q{q}, sol(q),
        sol2(q), lhs_domain1(q), lhs_domain2(q), rhs_domain(q),
        epsilon_oC_map(q), epsilon_oE_map(q), epsilon_oN_map(q),
        epsilon_uC_map(q), kappa_(q), epsilon_oNE_map(q), epsilon_oNW_map(q),
        epsilon_uE_map(q), epsilon_uW_map(q), epsilon_uN_map(q),
        epsilon_uS_map(q), epsilon_uNE_map(q), epsilon_uNW_map(q),
        epsilon_uSE_map(q), epsilon_uSW_map(q), oPC(q), oPN(q), oPS(q), oPE(q),
        oPW(q), oPNE(q), oPNW(q), oPSE(q), oPSW(q), uPC(q), uPN(q), uPS(q),
        uPE(q), uPW(q), uPNE(q), uPNW(q), uPSE(q), uPSW(q), dPC(q), dPN(q),
        dPS(q), dPE(q), dPW(q), dPNE(q), dPNW(q), dPSE(q), dPSW(q)

  {

    this->kappa = KappaA(ionic_strength);
    this->kappa_2 = kappa * kappa;

    q.fill(epsilon_uC_map.get_domain().values_buff,
           (DataType)epsilon_r / (grid_step * grid_step),
           epsilon_uC_map.get_domain().num_values);

    q.fill(epsilon_oN_map.get_domain().values_buff,
           (DataType)epsilon_r / (grid_step * grid_step),
           epsilon_oN_map.get_domain().num_values);

    q.fill(epsilon_oE_map.get_domain().values_buff,
           (DataType)epsilon_r / (grid_step * grid_step),
           epsilon_oE_map.get_domain().num_values);
  }

  inline void Set_boundary_conditions(Atom<DataType> *atoms,
                                      std::size_t num_atoms,
                                      const Domain_Type_upper &domain,
                                      std::size_t x, std::size_t y,
                                      std::size_t z) {
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

  void initialize_boundary(std::vector<Atom<DataType>> atom_list) {

    constexpr auto length = Domain_Type<>::length;

    Atom<DataType> *atoms_device =
        sycl::malloc_device<Atom<DataType>>(atom_list.size(), q);

    auto num_atoms = atom_list.size();
    auto &boundary_domain = sol.template get_domain<nlev>();

    q.memcpy(atoms_device, atom_list.data(),
             atom_list.size() * sizeof(Atom<DataType>));

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
  }

  void initialize_epsilons(std::vector<Atom<DataType>> atom_list) {

    Atom<DataType> *atoms_device =
        sycl::malloc_device<Atom<DataType>>(atom_list.size(), q);

    q.memcpy(atoms_device, atom_list.data(),
             atom_list.size() * sizeof(Atom<DataType>));

    auto &epsilonuC_domain = epsilon_uC_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atom_list.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[0] -= 0.5 * grid_step;
      find_dots_in_sphere(Atom, epsilonuC_domain,
                          static_cast<DataType>(grid_step),
                          (DataType)epsilon_p / (grid_step * grid_step));
    });

    auto &epsilonoN_domain = epsilon_oN_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atom_list.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[1] -= 0.5 * grid_step;
      find_dots_in_sphere(Atom, epsilonoN_domain,
                          static_cast<DataType>(grid_step),
                          (DataType)epsilon_p / (grid_step * grid_step));
    });

    auto &epsilonoE_domain = epsilon_oE_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atom_list.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[2] -= 0.5 * grid_step;
      find_dots_in_sphere(Atom, epsilonoE_domain,
                          static_cast<DataType>(grid_step),
                          (DataType)epsilon_p / (grid_step * grid_step));
    });

    auto &epsilonc_domain = epsilon_oC_map.template get_domain<nlev>();
    compute_center_domain(epsilonc_domain, epsilonuC_domain, epsilonoN_domain,
                          epsilonoE_domain);

    auto &kappa_domain = kappa_.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atom_list.size()), [=](sycl::id<1> I) {
      auto atom = atoms_device[I];
      atom.radius += 1.5;
      find_dots_in_sphere(atom, kappa_domain, static_cast<DataType>(grid_step));
    });

    // Inverting the kappa domain because the original functions marks the
    // points inside the protein with 1. And assigning the right value

    const auto *values_buff = kappa_domain.values_buff;
    const DataType kappa_2 = this->kappa_2;

    q.parallel_for(sycl::range<1>(kappa_domain.num_values), [=](sycl::id<1> I) {
      kappa_domain.values_buff[I] != 0
          ? kappa_domain.values_buff[I] = 0
          : kappa_domain.values_buff[I] = kappa_2 * epsilon_r;
    });
  }

  void set_up_rhs(std::vector<Atom<DataType>> atom_list) {
    auto &rhs = rhs_domain.template get_domain<nlev>();

    const auto num_atoms = atom_list.size();

    Atom<DataType> *atoms_device =
        sycl::malloc_device<Atom<DataType>>(atom_list.size(), q);

    q.memcpy(atoms_device, atom_list.data(),
             atom_list.size() * sizeof(Atom<DataType>));

    q.submit([=](sycl::handler &h) {
      h.single_task([=]() {
        for (int I = 0; I < num_atoms; I++)
          add_charges_to_distribution(
              rhs, atoms_device[I].Position,
              static_cast<DataType>(atoms_device[I].charge / epsilon),
              spacing<DataType, static_cast<DataType>(grid_step)>{});
      });
    });
  }

  template <std::size_t level = nlev> void buildmultilevelprolop() {
    if constexpr (level == 1) {
      return;
    } else if constexpr (level == nlev) {

      pmgc::VbuildPb_op7(&nx<nlev>, &ny<nlev>, &nz<nlev>, &nx<nlev - 1>,
                         &ny<nlev - 1>, &nz<nlev - 1>, (int *)nullptr,
                         (DataType *)nullptr,
                         epsilon_oC_map.template get_domain<nlev>().values_buff,
                         epsilon_oE_map.template get_domain<nlev>().values_buff,
                         epsilon_oN_map.template get_domain<nlev>().values_buff,
                         epsilon_uC_map.template get_domain<nlev>().values_buff,
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

      buildmultilevelprolop<level - 1>();
    } else {

      pmgc::VbuildPb_op27(
          &nx<level>, &ny<level>, &nz<level>, &nx<level - 1>, &ny<level - 1>,
          &nz<level - 1>, (int *)nullptr, (DataType *)nullptr,
          epsilon_oC_map.template get_domain<level>().values_buff,
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
          oPC.template get_domain<level - 1>().values_buff,
          oPN.template get_domain<level - 1>().values_buff,
          oPS.template get_domain<level - 1>().values_buff,
          oPE.template get_domain<level - 1>().values_buff,
          oPW.template get_domain<level - 1>().values_buff,
          oPNE.template get_domain<level - 1>().values_buff,
          oPNW.template get_domain<level - 1>().values_buff,
          oPSE.template get_domain<level - 1>().values_buff,
          oPSW.template get_domain<level - 1>().values_buff,
          uPC.template get_domain<level - 1>().values_buff,
          uPN.template get_domain<level - 1>().values_buff,
          uPS.template get_domain<level - 1>().values_buff,
          uPE.template get_domain<level - 1>().values_buff,
          uPW.template get_domain<level - 1>().values_buff,
          uPNE.template get_domain<level - 1>().values_buff,
          uPNW.template get_domain<level - 1>().values_buff,
          uPSE.template get_domain<level - 1>().values_buff,
          uPSW.template get_domain<level - 1>().values_buff,
          dPC.template get_domain<level - 1>().values_buff,
          dPN.template get_domain<level - 1>().values_buff,
          dPS.template get_domain<level - 1>().values_buff,
          dPE.template get_domain<level - 1>().values_buff,
          dPW.template get_domain<level - 1>().values_buff,
          dPNE.template get_domain<level - 1>().values_buff,
          dPNW.template get_domain<level - 1>().values_buff,
          dPSE.template get_domain<level - 1>().values_buff,
          dPSW.template get_domain<level - 1>().values_buff, q);

      buildmultilevelprolop<level - 1>();
    }
  }

  static DataType sqr(DataType val) { return val * val; }

  sycl::queue &q;

  Domain_Type<> sol, sol2, lhs_domain1, lhs_domain2, rhs_domain, epsilon_oC_map,
      epsilon_oE_map, epsilon_oN_map, epsilon_uC_map, kappa_;

  Domain_Type<nlev - 1> epsilon_oNE_map, epsilon_oNW_map, epsilon_uE_map,
      epsilon_uW_map, epsilon_uN_map, epsilon_uS_map, epsilon_uNE_map,
      epsilon_uNW_map, epsilon_uSE_map, epsilon_uSW_map;

  Domain_Type<nlev - 1> oPC, oPN, oPS, oPE, oPW, oPNE, oPNW, oPSE, oPSW, uPC,
      uPN, uPS, uPE, uPW, uPNE, uPNW, uPSE, uPSW, dPC, dPN, dPS, dPE, dPW, dPNE,
      dPNW, dPSE, dPSW;

  DataType ionic_strength;
  DataType kappa;
  DataType kappa_2;
  DataType ion_radius;

  constexpr static DataType grid_step =
      (DataType)box_length /
      ((DataType)base_length * utils::power_off(2, nlev));
  constexpr static DataType delta_epsilon = (epsilon_p - epsilon_r);

  template <std::size_t level = nlev>
  constexpr static auto strides = Domain_Type<level>::length;

  // Defining the pmgc length of the level
  template <std::size_t level = nlev>
  static const int nx = std::get<0>(strides<level>) + 2;

  template <std::size_t level = nlev>
  static const int ny = std::get<1>(strides<level>) + 2;

  template <std::size_t level = nlev>
  static const int nz = std::get<2>(strides<level>) + 2;
};

#endif
