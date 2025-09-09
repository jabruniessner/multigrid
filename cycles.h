#include "Convolution.h"
#include "MultigridDomain.h"
#include "bitshift_lib.h"
#include "get_epsilon_values.h"
#include "level_transition.h"
#include "profiling_library.h"
#include "scientific_quantities.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>

#ifndef CYCLES_H
#define CYCLES_H

namespace cycles {

using namespace multigrid_domain;

// template<std::size_t Num_Iters, Dimension Dim,
//	typename DataType, typename OffsetType,
//	std::size_t length, Length... strides_all>
// struct Jacobi_Smoother
//{
//
//	Jacobi_Smoother(){};
//
//	Jacobi_Smoother(Integer<Num_Iters> integer,
//			Domain<Dim, strides_all...> domain,
//			std::array<DataType, length> values,
//			std::array<OffsetType, length> offsets){}
//
//	void operator()(Domain<Dim, strides_all...>& dest,
//		 Domain<Dim, strides_all...>& src,
//		 Domain<Dim, strides_all...>& rhs,
//		 std::array<DataType, length>& values,
//		 std::array<OffsetType, length>& offsets)
//	{
//		for(int i = 0; i< Num_Iters; i++)
//		{
//			convolution::Convolve(dest, src, values, offsets);
//			subtract_domains(dest, rhs, dest);
//
//			DataType* temp = dest.values_buff;
//		        dest.values_buff = src.values_buff;
//			src.values_buff = temp;
//		}
//
//
//		DataType* temp = dest.values_buff;
//		dest.values_buff = src.values_buff;
//		src.values_buff = temp;
//	}
//
//
// };

template <std::size_t level, std::size_t... Num_Iters>
constexpr std::size_t get_num_iters() {

  auto Iter_Tuple = std::make_tuple(Num_Iters...);
  if constexpr (sizeof...(Num_Iters) == 1) {
    return std::get<0>(Iter_Tuple);
  } else {
    return std::get<level - 2>(Iter_Tuple);
  }
}

template <Dimension Dim, Length... strides_all, typename DataType,
          typename OffsetType, size_t size>
DataType compute_residual(Domain<Dim, strides_all...> &rhs,
                          Domain<Dim, strides_all...> &sol,
                          Domain<Dim, strides_all...> &helper,
                          const std::array<DataType, size> &values,
                          const std::array<OffsetType, size> &offsets) {
  convolution::Subtract_Convolve(helper, sol, rhs, values, offsets);
  DataType result;
  DataType rhs_norm;
  domain_compute_norm_squared(result, helper);
  // domain_compute_norm_squared(rhs_norm, rhs);
  return std::sqrt(result / helper.num_dofs);
}

template <Dimension Dim, Length... strides_all, typename DataType,
          typename OffsetType, size_t size>
DataType compute_truth_deviation(Domain<Dim, strides_all...> &truth,
                                 Domain<Dim, strides_all...> &sol,
                                 Domain<Dim, strides_all...> &helper,
                                 const std::array<DataType, size> &values,
                                 const std::array<OffsetType, size> &offsets) {
  domain::subtract_domains(helper, sol, truth);
  DataType result;
  domain_compute_norm_squared(result, helper);
  return std::sqrt(result / helper.num_dofs);
}

template <Dimension Dim, Length... strides_all, typename DataType,
          typename OffsetType, size_t size>
DataType compute_truth_deviaton_gradient(
    Domain<Dim, strides_all...> &truth, Domain<Dim, strides_all...> &sol,
    Domain<Dim, strides_all...> &helper, Domain<Dim, strides_all...> &helper2,
    const std::array<DataType, size> &values,
    const std::array<OffsetType, size> &offsets, DataType grid_step) {

  domain::subtract_domains(helper, sol, truth);
  convolution::Convolve(helper2, helper, values, offsets);
  DataType result = domain::domain_scalar_product(helper2, helper);
  return std::sqrt(result / helper.num_dofs / (grid_step * grid_step));
}

template <Dimension Dim, Length... strides_all>
DataType compute_residual_PBE(Domain<Dim, strides_all...> &rhs,
                              Domain<Dim, strides_all...> &sol,
                              Domain<Dim, strides_all...> &helper,
                              Domain<Dim, strides_all...> &kappa_map,
                              Domain<Dim, strides_all...> &epsilon_x,
                              Domain<Dim, strides_all...> &epsilon_y,
                              Domain<Dim, strides_all...> &epsilon_z,
                              DataType Kappa_2, DataType grid_step,
                              DataType epsilon_r, DataType delta_epsilon) {

  std::array<Domain<Dim, strides_all...>, Dim> epsilon_maps{
      epsilon_x, epsilon_y, epsilon_z};

  //  std::cout << "The sol domain at the beginning is: " << std::endl;
  //  sol.print_domain();

  convolution::PBE_Convolve(helper, sol, kappa_map, epsilon_maps, Kappa_2,
                            grid_step, epsilon_r, delta_epsilon);

  add_domains(helper, rhs, helper);

  DataType result;
  domain_compute_norm_squared(result, helper);
  return std::sqrt(result / sol.num_values);
}

template <Dimension Dim, typename DataType, typename OffsetType,
          std::size_t length, std::size_t nlev, Length... base_length>
struct Jacobi_Smoother {
  Jacobi_Smoother() {};

  Jacobi_Smoother(Multigrid_domain<Dim, nlev, base_length...>,
                  std::array<DataType, length> values,
                  std::array<OffsetType, length> offsets) {}

  template <std::size_t num> struct TD;

  template <std::size_t level, std::size_t... Num_Iters>
  void operator()(Integer<level>, std::index_sequence<Num_Iters...>,
                  Multigrid_domain<Dim, nlev, base_length...> &dest,
                  Multigrid_domain<Dim, nlev, base_length...> &src,
                  Multigrid_domain<Dim, nlev, base_length...> &rhs,
                  std::array<DataType, length> &values,
                  std::array<OffsetType, length> &offsets,
                  const DataType &box_length, const DataType omega)

  {

    static_assert(sizeof...(Num_Iters) == 1 ||
                  sizeof...(Num_Iters) == nlev - 1);

    constexpr std::size_t num_iters = get_num_iters<level, Num_Iters...>();

    auto &dest_domain = dest.template get_domain<level>();
    auto &src_domain = src.template get_domain<level>();
    auto &rhs_domain = rhs.template get_domain<level>();
    const DataType h =
        box_length / (std::get<0>(rhs.template get_length<level>()) + 1);
    const DataType diag_inverse = omega * (h * h) / (2 * Dim);

    if constexpr (num_iters == 0) {
      return;
    } else {
      for (int i = 0; i < num_iters; i++) {
        convolution::Convolve(dest_domain, src_domain, values, offsets);
        subtract_and_multiply_domains(dest_domain, rhs_domain, dest_domain,
                                      diag_inverse);

        std::swap(dest_domain.values_buff, src_domain.values_buff);
      }

      std::swap(dest_domain.values_buff, src_domain.values_buff);
    }
  }

  template <std::size_t... Num_Iters>
  void operator()(std::index_sequence<Num_Iters...>,
                  Multigrid_domain<Dim, nlev, base_length...> &dest,
                  Multigrid_domain<Dim, nlev, base_length...> &src,
                  Multigrid_domain<Dim, nlev, base_length...> &rhs,
                  std::array<DataType, length> &values,
                  std::array<OffsetType, length> &offsets, DataType &box_length)

  {
    this->operator()(Integer<nlev>{}, dest, src, rhs, values, offsets,
                     box_length);
  }
};

template <Dimension Dim, std::size_t nlev, Length... base_length>
struct GS_Smoother {
  GS_Smoother() {};

  GS_Smoother(Multigrid_domain<Dim, nlev, base_length...>) {}

  template <std::size_t num> struct TD;

  template <std::size_t level, typename DataType, std::size_t... Num_Iters,
            std::size_t... dims>
  void operator()(Integer<level>, std::index_sequence<Num_Iters...>,
                  Multigrid_domain<Dim, nlev, base_length...> &dest,
                  Multigrid_domain<Dim, nlev, base_length...> &src,
                  Multigrid_domain<Dim, nlev, base_length...> &rhs,
                  const DataType grid_step, const DataType omega,
                  std::index_sequence<dims...>, bool zeros_start = false)

  {

    static_assert(sizeof...(Num_Iters) == 1 ||
                  sizeof...(Num_Iters) == nlev - 1);

    static_assert(Dim == sizeof...(dims));

    constexpr std::size_t num_iters = get_num_iters<level, Num_Iters...>();

    auto &dest_domain = dest.template get_domain<level>();
    auto &src_domain = src.template get_domain<level>();
    auto &rhs_domain = rhs.template get_domain<level>();
    using d_type = std::remove_reference_t<decltype(src_domain)>;

    const DataType h = grid_step;
    const DataType diag_inverse = omega / (2 * Dim);

    std::array<Dimension, Dim> strides_array =
        std::to_array(src_domain.strides);
    auto range = std::make_from_tuple<sycl::range<Dim>>(strides_array);

    auto index_add_in_place = [=](int place, const d_type domain,
                                  auto... elems) {
      std::array<std::size_t, sizeof...(elems)> indices{elems...};
      indices[place] += 1;
      return std::apply(domain, indices);
    };

    auto index_sub_in_place = [=](int place, const d_type domain,
                                  auto... elems) {
      std::array<std::size_t, sizeof...(elems)> indices{elems...};
      indices[place] -= 1;
      return std::apply(domain, indices);
    };

    if constexpr (num_iters == 0) {
      return;
    } else {
      for (int i = 0; i < num_iters; i++) {
        for (int color = 0; color < 2; color++) {
          src_domain.q.parallel_for(range, [=](sycl::id<Dim> I) {
            ((I[dims] += src_domain.padding_width), ...);

            if ((I[dims] + ...) % 2 == color) {
              const auto subs =
                  (index_sub_in_place(dims, src_domain, I[dims]...) + ...) *
                  (zeros_start && i == 0 ? color : 1);

              const auto adds =
                  (index_add_in_place(dims, src_domain, I[dims]...) + ...) *
                  (zeros_start && i == 0 ? color : 1);

              src_domain(I[dims]...) =
                  diag_inverse * (adds + subs + h * h * rhs_domain(I[dims]...));
            }
          });
        }
      }
    }

    std::swap(dest_domain.values_buff, src_domain.values_buff);
  }

  template <std::size_t level, typename DataType, std::size_t... Num_Iters>
  void operator()(Integer<level>, std::index_sequence<Num_Iters...>,
                  Multigrid_domain<Dim, nlev, base_length...> &dest,
                  Multigrid_domain<Dim, nlev, base_length...> &src,
                  Multigrid_domain<Dim, nlev, base_length...> &rhs,
                  DataType grid_step, DataType omega,
                  const bool zeros_initialize = false)

  {
    this->operator()(Integer<level>{}, std::index_sequence<Num_Iters...>{},
                     dest, src, rhs, grid_step, omega,
                     std::make_index_sequence<Dim>{}, zeros_initialize);
  }
};

template <typename num_type, Dimension Dim, std::size_t nlev,
          Length... base_length>
struct GS_Smoother_epsilon {
  GS_Smoother_epsilon() {};

  GS_Smoother_epsilon(Multigrid_domain<Dim, nlev, base_length...>,
                      Multigrid_domain_t<num_type, Dim, 1, nlev, base_length...,
                                         utils::factorial(Dim) - 2>
                          epsilon)
      : epsilon(epsilon) {}

  template <std::size_t num> struct TD;

  template <std::size_t level, std::size_t... Num_Iters, std::size_t... dims>
  void operator()(Integer<level>, std::index_sequence<Num_Iters...>,
                  Multigrid_domain<Dim, nlev, base_length...> &dest,
                  Multigrid_domain<Dim, nlev, base_length...> &src,
                  Multigrid_domain<Dim, nlev, base_length...> &rhs,
                  const DataType grid_step, const DataType omega,
                  std::index_sequence<dims...>, bool zeros_start = false)

  {

    static_assert(sizeof...(Num_Iters) == 1 ||
                  sizeof...(Num_Iters) == nlev - 1);

    static_assert(Dim == sizeof...(dims));

    constexpr std::size_t num_iters = get_num_iters<level, Num_Iters...>();

    auto &dest_domain = dest.template get_domain<level>();
    auto &src_domain = src.template get_domain<level>();
    auto rhs_domain = rhs.template get_domain<level>();
    auto epsilon_domain = epsilon.template get_domain<level>();
    using d_type = std::remove_reference_t<decltype(src_domain)>;

    const DataType h = grid_step;
    const DataType diag_inverse = omega / (2 * Dim);

    constexpr auto strides_array =
        std::remove_reference_t<decltype(dest_domain)>::length;

    auto range = std::make_from_tuple<sycl::range<Dim>>(strides_array);
    auto index_add_in_place = [=](int place, const d_type domain,
                                  auto epsilon_values, auto... elems) {
      std::array<std::size_t, sizeof...(elems)> indices{elems...};
      indices[place] += 1;
      return std::apply(domain, indices) *
             get_local_epsilon(
                 static_cast<num_type>(2 * Dim - epsilon_values[place]),
                 static_cast<num_type>(epsilon_values[place]));
    };

    auto index_sub_in_place = [=](int place, const d_type domain,
                                  auto epsilon_values, auto... elems) {
      std::array<std::size_t, sizeof...(elems)> indices{elems...};
      indices[place] -= 1;
      return std::apply(domain, indices) *
             get_local_epsilon(
                 static_cast<num_type>(2 * Dim - epsilon_values[place]),
                 static_cast<num_type>(epsilon_values[place]));
    };

    if constexpr (num_iters == 0) {
      return;
    } else {

      for (int i = 0; i < num_iters; i++) {
        for (int color = 0; color < 2; color++) {
          src_domain.q.parallel_for(range, [=](sycl::id<Dim> I) {
            ((I[dims] += src_domain.padding_width), ...);

            std::array<num_type, 2 * Dim> epsilon_values =
                get_tet_vals_dim<num_type, Dim,
                                 std::get<dims>(strides_array)...>(
                    epsilon_domain, (int)I[dims]...);

            if ((I[dims] + ...) % 2 == color) {
              const auto subs =
                  (index_sub_in_place(dims, src_domain, epsilon_values,
                                      I[dims]...) +
                   ...) *
                  (zeros_start && i == 0 ? color : 1);

              const auto adds =
                  (index_add_in_place(dims, src_domain, epsilon_values,
                                      I[dims]...) +
                   ...) *
                  (zeros_start && i == 0 ? color : 1);

              DataType middle_value = 0;
              for (int i = 0; i < 2 * Dim; i++)
                middle_value += get_local_epsilon(
                    static_cast<num_type>(2 * Dim - epsilon_values[i]),
                    static_cast<num_type>(epsilon_values[i]));

              constexpr auto fac_inv = 1. / utils::factorial(Dim);

              src_domain(I[dims]...) =
                  (adds + subs + h * h * rhs_domain(I[dims]...)) / middle_value;
            }
          });
        }
      }
    }

    src.domain.q.wait();

    std::swap(dest_domain.values_buff, src_domain.values_buff);
  }

  template <std::size_t level, typename DataType, std::size_t... Num_Iters>
  void operator()(Integer<level>, std::index_sequence<Num_Iters...>,
                  Multigrid_domain<Dim, nlev, base_length...> &dest,
                  Multigrid_domain<Dim, nlev, base_length...> &src,
                  Multigrid_domain<Dim, nlev, base_length...> &rhs,
                  DataType grid_step, DataType omega,
                  const bool zeros_initialize = false)

  {
    this->operator()(Integer<level>{}, std::index_sequence<Num_Iters...>{},
                     dest, src, rhs, grid_step, omega,
                     std::make_index_sequence<Dim>{}, zeros_initialize);
  }

  Multigrid_domain_t<num_type, Dim, 1, nlev, base_length...,
                     utils::factorial(Dim) - 2>
      epsilon;
};

template <typename num_type, Dimension Dim, std::size_t nlev,
          Length... base_length>
auto make_GS_Smoother_epsilon(
    Multigrid_domain<Dim, nlev, base_length...> a,
    Multigrid_domain_t<num_type, Dim, 1, nlev, base_length...,
                       utils::factorial(Dim) - 2>
        epsilon) {
  return GS_Smoother_epsilon<num_type, Dim, nlev, base_length...>(a, epsilon);
}

template <Dimension Dim, std::size_t nlev, Length... base_length>
struct Jacobi_Smoother_PBE {
  Jacobi_Smoother_PBE() {};

  Jacobi_Smoother_PBE(Multigrid_domain<Dim, nlev, base_length...>) {}

  template <std::size_t num> struct TD;

  template <std::size_t level, std::size_t... Num_Iters>
  void operator()(Integer<level>, std::index_sequence<Num_Iters...>,
                  Multigrid_domain<Dim, nlev, base_length...> &dest,
                  Multigrid_domain<Dim, nlev, base_length...> &src,
                  Multigrid_domain<Dim, nlev, base_length...> &rhs,
                  Multigrid_domain<Dim, nlev, base_length...> &kappa,
                  Multigrid_domain<Dim, nlev, base_length...> &epsilon_x,
                  Multigrid_domain<Dim, nlev, base_length...> &epsilon_y,
                  Multigrid_domain<Dim, nlev, base_length...> &epsilon_z,
                  const DataType &kappa_2, const DataType grid_step,
                  const DataType epsilon_r, const DataType delta_epsilon,
                  const DataType omega)

  {

    static_assert(sizeof...(Num_Iters) == 1 ||
                  sizeof...(Num_Iters) == nlev - 1);

    constexpr std::size_t num_iters = get_num_iters<level, Num_Iters...>();

    std::cout << "The number of iterations is " << num_iters << std::endl;

    auto &dest_domain = dest.template get_domain<level>();
    auto &src_domain = src.template get_domain<level>();
    auto &rhs_domain = rhs.template get_domain<level>();
    auto &kappa_domain = kappa.template get_domain<level>();
    auto epsilon_x_domain = epsilon_x.template get_domain<level>();
    auto epsilon_y_domain = epsilon_y.template get_domain<level>();
    auto epsilon_z_domain = epsilon_z.template get_domain<level>();
    // using domain_type = decltype(epsilon_x_domain);

    //  std::array<domain_type, 3> epsilon_maps{epsilon_x_domain,
    //  epsilon_y_domain,
    //                                          epsilon_z_domain};
    //
    constexpr auto lengths = decltype(epsilon_x.domain)::length;
    std::array<Domain<Dim, std::get<0>(lengths), std::get<1>(lengths),
                      std::get<2>(lengths)>,
               Dim>
        epsilon_maps{epsilon_x_domain, epsilon_y_domain, epsilon_z_domain};

    const DataType h = grid_step;
    const DataType diag_inverse_helper = omega * (h * h);

    //  std::cout << "Diag inverse helper: " << diag_inverse_helper <<
    //  std::endl; std::cout << "omega: " << omega << std::endl; std::cout <<
    //  "h: " << h << std::endl; std::cout << "Box length: " << box_length <<
    //  std::endl;

    if constexpr (num_iters == 0) {
      return;
    } else {
      for (int i = 0; i < num_iters; i++) {
        std::array<Dimension, Dim> strides_array =
            std::to_array(dest_domain.strides);

        auto range = std::make_from_tuple<sycl::range<Dim>>(strides_array);

        dest_domain.q.parallel_for(range, [=](sycl::id<Dim> I) {
          I[0] += src_domain.padding_width;
          I[1] += src_domain.padding_width;
          I[2] += src_domain.padding_width;

          DataType DinvA = -convolution::PBE_Convolve_kernel(
              src_domain, kappa_domain, epsilon_maps, kappa_2, grid_step,
              epsilon_r, delta_epsilon, I);

          // std::cout << DinvA << std::endl;

          DataType diag_inverse_denominator =
              kappa_domain(I[0], I[1], I[2]) * kappa_2 * epsilon_r;

          // We first need to compute the right diagonal value
          for (int j = 0; j < Dim; j++) {
            sycl::id<Dim> I2{I}, I3{I};
            I2[j] += 1;
            I3[j] -= 1;

            const DataType epsilon_lower =
                epsilon_r +
                epsilon_maps[j](I3[0], I3[1], I3[2]) * delta_epsilon;
            const DataType epsilon_upper =
                epsilon_r + epsilon_maps[j](I[0], I[1], I[2]) * delta_epsilon;

            diag_inverse_denominator += epsilon_lower + epsilon_upper;
          }

          //  if (I == sycl::id<Dim>{1, 1, 1}) {
          //    std::cout << "src_domain value: " << src_domain(I[0], I[1],
          //    I[2])
          //              << std::endl;
          //    std::cout << "dest_domain value: " << dest_domain(I[0], I[1],
          //    I[2])
          //              << std::endl;
          //    std::cout << "diag_inverse value: " << diag_inverse <<
          //    std::endl; std::cout << "rhs_domain value: " << rhs_domain(I[0],
          //    I[1], I[2])
          //              << std::endl;

          //    std::cout << "DinvA: " << DinvA << std::endl;

          //    std::cout << "The result is: "
          //              << src_domain(I[0], I[1], I[2]) - diag_inverse * DinvA
          //              +
          //                     diag_inverse * rhs_domain(I[0], I[1], I[2])
          //              << std::endl;
          //}

          const DataType diag_inverse =
              diag_inverse_helper / (diag_inverse_denominator);

          dest_domain(I[0], I[1], I[2]) =
              src_domain(I[0], I[1], I[2]) - diag_inverse * DinvA +
              diag_inverse * rhs_domain(I[0], I[1], I[2]);

          // if (I == sycl::id<Dim>{1, 1, 1}) {
          //   std::cout << "The result is: " << dest_domain(I[0], I[1], I[2])
          //             << std::endl;

          //   std::cout << "The diag_inverse is: " << diag_inverse <<
          //   std::endl; std::cout << "The diag_inverse_helper is: " <<
          //   diag_inverse_helper
          //             << std::endl;
          //   std::cout << "The diag_inverse_denominator is: "
          //             << diag_inverse_denominator << std::endl;
          // }
        });

        std::swap(dest_domain.values_buff, src_domain.values_buff);
      }

      std::swap(dest_domain.values_buff, src_domain.values_buff);
    }
  }

  template <std::size_t... Num_Iters>
  void operator()(std::index_sequence<Num_Iters...>,
                  Multigrid_domain<Dim, nlev, base_length...> &dest,
                  Multigrid_domain<Dim, nlev, base_length...> &src,
                  Multigrid_domain<Dim, nlev, base_length...> &rhs,
                  DataType &box_length)

  {
    this->operator()(Integer<nlev>{}, dest, src, rhs, box_length);
  }
};

// template <Dimension Dim, std::size_t nlev, Length... base_length>
// struct Gauss_Seidel_PBE {
//   Gauss_Seidel_PBE() {};
//
//   Gauss_Seidel_PBE(Multigrid_domain<Dim, nlev, base_length...>) {}
//
//   template <std::size_t num> struct TD;
//
//   template <std::size_t level, std::size_t... Num_Iters>
//   void operator()(Integer<level>, std::index_sequence<Num_Iters...>,
//                   Multigrid_domain<Dim, nlev, base_length...> &dest,
//                   Multigrid_domain<Dim, nlev, base_length...> &src,
//                   Multigrid_domain<Dim, nlev, base_length...> &rhs,
//                   Multigrid_domain<Dim, nlev, base_length...> &kappa,
//                   Multigrid_domain<Dim, nlev, base_length...> &epsilon_x,
//                   Multigrid_domain<Dim, nlev, base_length...> &epsilon_y,
//                   Multigrid_domain<Dim, nlev, base_length...> &epsilon_z,
//                   const DataType &kappa_2, const DataType grid_step,
//                   const DataType epsilon_r, const DataType delta_epsilon,
//                   const DataType omega)
//
//   {
//
//     // std::cout << "We are doing a Gauss-Seidel smoothing" << std::endl;
//
//     static_assert(sizeof...(Num_Iters) == 1 ||
//                   sizeof...(Num_Iters) == nlev - 1);
//
//     constexpr std::size_t num_iters = get_num_iters<level, Num_Iters...>();
//
//     //  std::cout << "The number of iterations is " << num_iters <<
//     std::endl;
//
//     auto &dest_domain = dest.template get_domain<level>();
//     auto &src_domain = src.template get_domain<level>();
//     auto &rhs_domain = rhs.template get_domain<level>();
//     auto &kappa_domain = kappa.template get_domain<level>();
//     auto epsilon_x_domain = epsilon_x.template get_domain<level>();
//     auto epsilon_y_domain = epsilon_y.template get_domain<level>();
//     auto epsilon_z_domain = epsilon_z.template get_domain<level>();
//     constexpr auto lengths = decltype(epsilon_x_domain)::length;
//     std::array<Domain<Dim, std::get<0>(lengths), std::get<1>(lengths),
//                       std::get<2>(lengths)>,
//                Dim>
//         epsilon_maps{epsilon_x_domain, epsilon_y_domain, epsilon_z_domain};
//
//     const DataType h = grid_step;
//     const DataType diag_inverse_helper = 1; // (h * h);
//
//     if constexpr (num_iters == 0) {
//       return;
//     } else {
//       for (int i = 0; i < num_iters; i++) {
//         std::array<Dimension, Dim> strides_array =
//             std::to_array(src_domain.strides);
//
//         //  std::cout << std::endl;
//         //  std::cout << std::endl;
//
//         //  std::cout << "Before " << i << " iterations " << std::endl;
//         //  std::cout << "<<==================================>>" <<
//         std::endl;
//
//         //  src_domain.print_domain();
//
//         auto range = std::make_from_tuple<sycl::range<Dim>>(strides_array);
//
//         for (int color = 0; color < 2; color++)
//           src_domain.q.parallel_for(range, [=](sycl::id<Dim> I) {
//             I[0] += src_domain.padding_width;
//             I[1] += src_domain.padding_width;
//             I[2] += src_domain.padding_width;
//
//             if ((I[0] + I[1] + I[2]) % 2 == color) {
//
//               auto &intermediate = color == 0 ? src_domain : dest_domain;
//               DataType Off_diagonal_contribution =
//               convolution::PBE_GS_kernel(
//                   intermediate, epsilon_maps, epsilon_r, delta_epsilon, I);
//
//               DataType diag_inverse_denominator =
//                   kappa_domain(I[0], I[1], I[2]) * h * h * kappa_2 *
//                   epsilon_r;
//
//               for (int j = 0; j < Dim; j++) {
//                 sycl::id<Dim> I2{I}, I3{I};
//                 I2[j] += 1;
//                 I3[j] -= 1;
//
//                 const DataType epsilon_lower =
//                     epsilon_r +
//                     epsilon_maps[j](I3[0], I3[1], I3[2]) * delta_epsilon;
//                 const DataType epsilon_upper =
//                     epsilon_r +
//                     epsilon_maps[j](I[0], I[1], I[2]) * delta_epsilon;
//
//                 diag_inverse_denominator += epsilon_lower + epsilon_upper;
//               }
//
//               dest_domain(I[0], I[1], I[2]) =
//                   (1 - omega) * src_domain(I[0], I[1], I[2]) +
//                   (omega) / diag_inverse_denominator *
//                       (rhs_domain(I[0], I[1], I[2]) * h * h +
//                        Off_diagonal_contribution);
//             }
//           });
//
//         //  std::cout << std::endl;
//         //  std::cout << std::endl;
//
//         //  std::cout << "After " << i << " iterations " << std::endl;
//         //  std::cout << "<<==================================>>" <<
//         std::endl;
//
//         //  src_domain.print_domain();
//
//         // std::swap(dest_domain.values_buff, src_domain.values_buff);
//       }
//
//       // std::swap(dest_domain.values_buff, src_domain.values_buff);
//     }
//   }

// template <std::size_t... Num_Iters>
// void operator()(std::index_sequence<Num_Iters...>,
//                 Multigrid_domain<Dim, nlev, base_length...> &dest,
//                 Multigrid_domain<Dim, nlev, base_length...> &src,
//                 Multigrid_domain<Dim, nlev, base_length...> &rhs,
//                 DataType &box_length)
//
//{
//   this->operator()(Integer<nlev>{}, dest, src, rhs, box_length);
// }
// };

template <Dimension Dim, std::size_t nlev, Length... base_length>
struct Gauss_Seidel_PBE {
  Gauss_Seidel_PBE() {};

  Gauss_Seidel_PBE(Multigrid_domain<Dim, nlev, base_length...>) {}

  template <std::size_t num> struct TD;

  template <std::size_t level, std::size_t... Num_Iters>
  void operator()(Integer<level>, std::index_sequence<Num_Iters...>,
                  Multigrid_domain<Dim, nlev, base_length...> &src,
                  Multigrid_domain<Dim, nlev, base_length...> &rhs,
                  Multigrid_domain<Dim, nlev, base_length...> &kappa,
                  Multigrid_domain<Dim, nlev, base_length...> &epsilon_x,
                  Multigrid_domain<Dim, nlev, base_length...> &epsilon_y,
                  Multigrid_domain<Dim, nlev, base_length...> &epsilon_z,
                  const DataType &kappa_2, const DataType grid_step,
                  const DataType epsilon_r, const DataType delta_epsilon,
                  const DataType omega)

  {

    std::cout << "We are doing a Gauss-Seidel smoothing" << std::endl;

    static_assert(sizeof...(Num_Iters) == 1 ||
                  sizeof...(Num_Iters) == nlev - 1);

    constexpr std::size_t num_iters = get_num_iters<level, Num_Iters...>();

    std::cout << "The number of iterations is " << num_iters << std::endl;

    auto &src_domain = src.template get_domain<level>();
    auto &rhs_domain = rhs.template get_domain<level>();
    auto &kappa_domain = kappa.template get_domain<level>();
    auto epsilon_x_domain = epsilon_x.template get_domain<level>();
    auto epsilon_y_domain = epsilon_y.template get_domain<level>();
    auto epsilon_z_domain = epsilon_z.template get_domain<level>();
    constexpr auto lengths = decltype(epsilon_x_domain)::length;
    std::array<Domain<Dim, std::get<0>(lengths), std::get<1>(lengths),
                      std::get<2>(lengths)>,
               Dim>
        epsilon_maps{epsilon_x_domain, epsilon_y_domain, epsilon_z_domain};

    const DataType h = grid_step;
    const DataType diag_inverse_helper = (h * h);

    if constexpr (num_iters == 0) {
      return;
    } else {
      for (int i = 0; i < num_iters; i++) {
        std::array<Dimension, Dim> strides_array =
            std::to_array(src_domain.strides);

        auto range = std::make_from_tuple<sycl::range<Dim>>(strides_array);

        int color = 0;

        for (int color = 0; color < 2; color++)
          src_domain.q.parallel_for(range, [=](sycl::id<Dim> I) {
            I[0] += src_domain.padding_width;
            I[1] += src_domain.padding_width;
            I[2] += src_domain.padding_width;
            if ((I[0] + I[1] + I[2]) % 2 == color) {

              auto &intermediate = src_domain;
              DataType Off_diagonal_contribution = convolution::PBE_GS_kernel(
                  intermediate, epsilon_maps, epsilon_r, delta_epsilon, I);

              DataType diag_inverse_denominator =
                  kappa_domain(I[0], I[1], I[2]) * h * h * kappa_2 * epsilon_r;

              // We first need to compute the right diagonal value
              for (int j = 0; j < Dim; j++) {
                sycl::id<Dim> I2{I}, I3{I};
                I2[j] += 1;
                I3[j] -= 1;

                const DataType epsilon_lower =
                    epsilon_r +
                    epsilon_maps[j](I3[0], I3[1], I3[2]) * delta_epsilon;
                const DataType epsilon_upper =
                    epsilon_r +
                    epsilon_maps[j](I[0], I[1], I[2]) * delta_epsilon;

                diag_inverse_denominator += epsilon_lower + epsilon_upper;
              }

              src_domain(I[0], I[1], I[2]) =
                  (1 - omega) * src_domain(I[0], I[1], I[2]) +
                  (omega) / diag_inverse_denominator *
                      (rhs_domain(I[0], I[1], I[2]) * h * h +
                       Off_diagonal_contribution);
            }
          });
      }
    }
  }

  template <std::size_t... Num_Iters>
  void operator()(std::index_sequence<Num_Iters...>,
                  Multigrid_domain<Dim, nlev, base_length...> &dest,
                  Multigrid_domain<Dim, nlev, base_length...> &src,
                  Multigrid_domain<Dim, nlev, base_length...> &rhs,
                  DataType &box_length)

  {
    this->operator()(Integer<nlev>{}, dest, src, rhs, box_length);
  }
};

template <typename Pre_Smoother, typename Post_Smoother, typename Solver,
          Dimension Dim, typename DataType, std::size_t nlev,
          std::size_t level = nlev, std::size_t base_length1 = 1,
          std::size_t... base_length>
struct V_Cycle_base {
  V_Cycle_base(Pre_Smoother &presmoother, Post_Smoother &postsmoother,
               Solver &solver,
               Multigrid_domain_t<DataType, Dim, 0u, nlev, base_length1,
                                  base_length...> &)
      : solver(solver), presmoother(presmoother), postsmoother(postsmoother) {}

  //  template <std::size_t... Ts> struct TD;
  //
  template <std::size_t iter_level = level,
            template <Dimension, std::size_t> typename FuncType,
            std::size_t... Num_Iters, std::size_t... Num_Iters_Smoother_Pre,
            std::size_t... Num_Iters_Smoother_Post>
  void iteration(
      Multigrid_domain<Dim, nlev, base_length1, base_length...> &next,
      Multigrid_domain<Dim, nlev, base_length1, base_length...> &current,
      Multigrid_domain<Dim, nlev, base_length1, base_length...> &rhs_domain,
      Multi_Level_map<Dim, DataType, FuncType, nlev> func, DataType grid_step,
      DataType omega, std::index_sequence<Num_Iters...> num_iters_,
      std::index_sequence<Num_Iters_Smoother_Pre...> smoother_iters_pre,
      std::index_sequence<Num_Iters_Smoother_Post...> smoother_iters_post,
      const bool zero_initialize = false) {

    PROFILE_ADD

    static_assert(sizeof...(Num_Iters) == 1 ||
                  sizeof...(Num_Iters) == nlev - 1);

    if constexpr (iter_level == 1) {
      solver(next.template get_domain<iter_level>(),
             rhs_domain.template get_domain<iter_level>(),
             func.template get_map<iter_level>());
      return;
    } else {

      constexpr std::size_t num_iters =
          get_num_iters<iter_level, Num_Iters...>();

      for (int j = 0; j < num_iters; j++) {
        PROFILE_START(overall_time)

        //  if (iter_level == nlev) {
        //    PROFILE_START(residual_computation)

        //    DataType residual;
        //    domain_compute_norm_squared(residual,
        //                                rhs_domain.template
        //                                get_domain<nlev>());
        //    residual = std::sqrt(residual /
        //                         rhs_domain.template
        //                         get_domain<nlev>().num_dofs);
        //    std::cout << "The residual after " << j << " iterations is "
        //              << residual << std::endl;
        //    PROFILE_END(residual_computation)
        //  }

        PROFILE_START(pre_smoothing)
        presmoother(Integer<iter_level>{}, smoother_iters_pre, next, current,
                    rhs_domain, grid_step, omega, zero_initialize);
        PROFILE_END(pre_smoothing)
        PRINT_DOMAIN(pre_smoothing, next, iter_level, nlev)

        //  next.template get_domain<nlev>().q.wait();

        PROFILE_START(defect_computation)
        convolution::Subtract_Convolve_map(
            current.template get_domain<iter_level>(),
            next.template get_domain<iter_level>(),
            rhs_domain.template get_domain<iter_level>(),
            func.template get_map<iter_level>());

        PRINT_DOMAIN(Subtract_Convolve, current, iter_level, nlev)

        PROFILE_END(defect_computation)

        PROFILE_START(restriction)
        level_transition::coarsening_inject(
            rhs_domain.template get_domain<iter_level - 1>(),
            current.template get_domain<iter_level>());
        PROFILE_END(restriction)

        iteration<iter_level - 1>(
            next, current, rhs_domain, func, sqrt2 * grid_step, omega,
            num_iters_, smoother_iters_pre, smoother_iters_post, true);

        PROFILE_START(refinement)
        level_transition::refinement(
            current.template get_domain<iter_level>(),
            next.template get_domain<iter_level - 1>());

        PROFILE_END(refinement)

        if (omega == (DataType)0) {
          omega = get_ideal_omega(next.template get_domain<iter_level>(),
                                  current.template get_domain<iter_level>(),
                                  rhs_domain.template get_domain<iter_level>(),
                                  func.template get_map<iter_level>());

          //  std::cout << "The value for omega on iter_level " << iter_level
          //            << " is " << omega << std::endl;
        }

        add_and_multiply_domains(next.template get_domain<iter_level>(),
                                 next.template get_domain<iter_level>(),
                                 current.template get_domain<iter_level>(),
                                 omega);

        PRINT_DOMAIN(correction, next, iter_level, nlev)

        PROFILE_START(post_smoothing)
        postsmoother(Integer<iter_level>{}, smoother_iters_post, next, next,
                     rhs_domain, grid_step, omega, false);
        PROFILE_END(post_smoothing)

        PROFILE_END(overall_time)
      }
    }
  }
  Solver &solver;
  Pre_Smoother &presmoother;
  Post_Smoother &postsmoother;
};

} // namespace cycles

#endif
