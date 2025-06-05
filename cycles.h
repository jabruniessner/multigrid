#include "Convolution.h"
#include "MultigridDomain.h"
#include "level_transition.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
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
  domain_compute_norm_squared(result, helper);
  return std::sqrt(result / sol.num_values);
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

template <Dimension Dim, std::size_t nlev, Length... base_length>
struct Gauss_Seidel_PBE {
  Gauss_Seidel_PBE() {};

  Gauss_Seidel_PBE(Multigrid_domain<Dim, nlev, base_length...>) {}

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

    std::cout << "We are doing a Gauss-Seidel smoothing" << std::endl;

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
    constexpr auto lengths = decltype(epsilon_x_domain)::length;
    std::array<Domain<Dim, std::get<0>(lengths), std::get<1>(lengths),
                      std::get<2>(lengths)>,
               Dim>
        epsilon_maps{epsilon_x_domain, epsilon_y_domain, epsilon_z_domain};

    const DataType h = grid_step;
    const DataType diag_inverse_helper = 1; // (h * h);

    if constexpr (num_iters == 0) {
      return;
    } else {
      for (int i = 0; i < num_iters; i++) {
        std::array<Dimension, Dim> strides_array =
            std::to_array(src_domain.strides);

        //  std::cout << std::endl;
        //  std::cout << std::endl;

        //  std::cout << "Before " << i << " iterations " << std::endl;
        //  std::cout << "<<==================================>>" << std::endl;

        //  src_domain.print_domain();

        auto range = std::make_from_tuple<sycl::range<Dim>>(strides_array);

        for (int color = 0; color < 2; color++)
          // src_domain.q.parallel_for(range, [=](sycl::id<Dim> I) {
          for (std::size_t i = 0; i < strides_array[0]; i++)
            for (std::size_t j = 0; j < strides_array[1]; j++)
              for (std::size_t k = 0; k < strides_array[2]; k++) {
                sycl::id<Dim> I{i, j, k};
                I[0] += src_domain.padding_width;
                I[1] += src_domain.padding_width;
                I[2] += src_domain.padding_width;

                if ((I[0] + I[1] + I[2]) % 2 == color) {

                  DataType diag_inverse_denominator =
                      kappa_domain(I[0], I[1], I[2]) * h * h * kappa_2 *
                      epsilon_r;

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

                  volatile DataType val = src_domain(I[0], I[1], I[2]);

#define GET_EPSILON(eps, x, y, z) (epsilon_r + eps(x, y, z) * delta_epsilon)

                  dest_domain(I[0], I[1], I[2]) = src_domain(I[0], I[1], I[2]) =
                      (rhs_domain(I[0], I[1], I[2]) * h * h +
                       GET_EPSILON(epsilon_y_domain, I[0], I[1], I[2]) *
                           src_domain(I[0], I[1] + 1, I[2]) +
                       GET_EPSILON(epsilon_y_domain, I[0], I[1] - 1, I[2]) *
                           src_domain(I[0], I[1] - 1, I[2]) +
                       GET_EPSILON(epsilon_x_domain, I[0], I[1], I[2]) *
                           src_domain(I[0] + 1, I[1], I[2]) +
                       GET_EPSILON(epsilon_x_domain, I[0] - 1, I[1], I[2]) *
                           src_domain(I[0] - 1, I[1], I[2]) +
                       GET_EPSILON(epsilon_z_domain, I[0], I[1], I[2] - 1) *
                           src_domain(I[0], I[1], I[2] - 1) +
                       GET_EPSILON(epsilon_z_domain, I[0], I[1], I[2]) *
                           src_domain(I[0], I[1], I[2] + 1)) /
                      (diag_inverse_denominator);

                  val = src_domain(I[0], I[1], I[2]);

                  volatile int dummy = 0;
                }
              }

        //);

        //  std::cout << std::endl;
        //  std::cout << std::endl;

        //  std::cout << "After " << i << " iterations " << std::endl;
        //  std::cout << "<<==================================>>" << std::endl;

        //  src_domain.print_domain();

        // std::swap(dest_domain.values_buff, src_domain.values_buff);
      }

      // std::swap(dest_domain.values_buff, src_domain.values_buff);
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
          Dimension Dim, std::size_t length, std::size_t length_diff_op,
          std::size_t length_coarsening_op, typename DataType, std::size_t nlev,
          std::size_t level = nlev, std::size_t base_length1 = 1,
          std::size_t... base_length>
struct V_Cycle_base {
  V_Cycle_base(
      Pre_Smoother &presmoother, Post_Smoother &post_smoother, Solver &solver,
      Multigrid_domain<Dim, nlev, base_length1, base_length...> &,
      Multi_Level_operator<Dim, DataType, length, base_length1, nlev> &,
      Multi_Level_operator<Dim, DataType, length_diff_op, base_length1, nlev> &,
      Multi_Level_operator<Dim, DataType, length_coarsening_op, base_length1,
                           nlev> &)
      : solver(solver) {}

  //  template <std::size_t... Ts> struct TD;
  //
  template <std::size_t iter_level = level, std::size_t... Num_Iters,
            std::size_t... Num_Iters_Smoother_Pre,
            std::size_t... Num_Iters_Smoother_Post>
  void iteration(
      Multigrid_domain<Dim, nlev, base_length1, base_length...> &next,
      Multigrid_domain<Dim, nlev, base_length1, base_length...> &current,
      Multigrid_domain<Dim, nlev, base_length1, base_length...> &rhs_domain,
      Multi_Level_operator<Dim, DataType, length, base_length1, nlev>
          &Smooth_operator,
      Multi_Level_operator<Dim, DataType, length_diff_op, base_length1, nlev>
          &Diff_operator,
      Multi_Level_operator<Dim, DataType, length_coarsening_op, base_length1,
                           nlev> &coarsening_operator,
      DataType box_length, DataType omega,
      std::index_sequence<Num_Iters...> num_iters_,
      std::index_sequence<Num_Iters_Smoother_Pre...> smoother_iters_pre,
      std::index_sequence<Num_Iters_Smoother_Post...> smoother_iters_post) {

    static_assert(sizeof...(Num_Iters) == 1 ||
                  sizeof...(Num_Iters) == nlev - 1);

    if constexpr (iter_level == 1) {
      solver(next.template get_domain<iter_level>(),
             rhs_domain.template get_domain<iter_level>(),
             Diff_operator.template get_values<iter_level>(),
             Diff_operator.template get_offsets<iter_level>());
      return;
    } else {

      constexpr std::size_t num_iters =
          get_num_iters<iter_level, Num_Iters...>();

      for (int j = 0; j < num_iters; j++) {
        pre_smoother(Integer<iter_level>{}, smoother_iters_pre, next, current,
                     rhs_domain,
                     Smooth_operator.template get_values<iter_level>(),
                     Smooth_operator.template get_offsets<iter_level>(),
                     box_length, omega);

        convolution::Convolve(current.template get_domain<iter_level>(),
                              next.template get_domain<iter_level>(),
                              Diff_operator.template get_values<iter_level>(),
                              Diff_operator.template get_offsets<iter_level>());

        subtract_domains(current.template get_domain<iter_level>(),
                         rhs_domain.template get_domain<iter_level>(),
                         current.template get_domain<iter_level>());

        level_transition::coarsening(
            rhs_domain.template get_domain<iter_level - 1>(),
            current.template get_domain<iter_level>(),
            coarsening_operator.template get_values<iter_level>(),
            coarsening_operator.template get_offsets<iter_level>());

        iteration<iter_level - 1>(next, current, rhs_domain, Smooth_operator,
                                  Diff_operator, coarsening_operator,
                                  box_length, omega, num_iters_,
                                  smoother_iters_pre, smoother_iters_post);

        level_transition::refinement(
            current.template get_domain<iter_level>(),
            next.template get_domain<iter_level - 1>());

        add_domains(next.template get_domain<iter_level>(),
                    next.template get_domain<iter_level>(),
                    current.template get_domain<iter_level>());
      }

      post_smoother(Integer<iter_level>{}, smoother_iters_post, current, next,
                    rhs_domain,
                    Smooth_operator.template get_values<iter_level>(),
                    Smooth_operator.template get_offsets<iter_level>(),
                    box_length, omega);
    }
  }
  Solver &solver;
  inline static Pre_Smoother pre_smoother{};
  inline static Post_Smoother post_smoother{};
};

template <typename Pre_Smoother, typename Post_Smoother, typename Solver,
          Dimension Dim, typename DataType, std::size_t nlev,
          std::size_t length_coarsening_op, std::size_t level = nlev,
          std::size_t base_length1 = 1, std::size_t... base_length>
struct V_Cycle_PBE {
  V_Cycle_PBE(Pre_Smoother &presmoother, Post_Smoother &post_smoother,
              Solver &solver,
              Multigrid_domain<Dim, nlev, base_length1, base_length...> &,
              Multi_Level_operator<Dim, DataType, length_coarsening_op,
                                   base_length1, nlev>)
      : solver(solver) {}

  //  template <std::size_t... Ts> struct TD;
  //
  template <std::size_t iter_level = level, std::size_t... Num_Iters,
            std::size_t... Num_Iters_Smoother_Pre,
            std::size_t... Num_Iters_Smoother_Post>
  void iteration(
      Multigrid_domain<Dim, nlev, base_length1, base_length...> &next,
      Multigrid_domain<Dim, nlev, base_length1, base_length...> &current,
      Multigrid_domain<Dim, nlev, base_length1, base_length...> &rhs_domain,
      Multigrid_domain<Dim, nlev, base_length1, base_length...> &epsilon_x,
      Multigrid_domain<Dim, nlev, base_length1, base_length...> &epsilon_y,
      Multigrid_domain<Dim, nlev, base_length1, base_length...> &epsilon_z,
      Multigrid_domain<Dim, nlev, base_length1, base_length...> &kappa_map,
      DataType kappa_2, DataType grid_step, DataType epsilon_r,
      DataType delta_epsilon, DataType omega,
      std::index_sequence<Num_Iters...> num_iters_,
      Multi_Level_operator<Dim, DataType, length_coarsening_op, base_length1,
                           nlev> &coarsening_operator,

      std::index_sequence<Num_Iters_Smoother_Pre...> smoother_iters_pre,
      std::index_sequence<Num_Iters_Smoother_Post...> smoother_iters_post) {

    static_assert(sizeof...(Num_Iters) == 1 ||
                  sizeof...(Num_Iters) == nlev - 1);

    if constexpr (iter_level == 1) {

      auto epsilon_x_domain = epsilon_x.template get_domain<iter_level>();
      auto epsilon_y_domain = epsilon_y.template get_domain<iter_level>();
      auto epsilon_z_domain = epsilon_z.template get_domain<iter_level>();

      constexpr auto lengths = decltype(epsilon_x_domain)::length;
      std::array<Domain<Dim, std::get<0>(lengths), std::get<1>(lengths),
                        std::get<2>(lengths)>,
                 Dim>
          epsilon_maps{epsilon_x_domain, epsilon_y_domain, epsilon_z_domain};

      solver(next.template get_domain<iter_level>(),
             rhs_domain.template get_domain<iter_level>(),
             kappa_map.template get_domain<iter_level>(), epsilon_maps, kappa_2,
             grid_step, epsilon_r, delta_epsilon);
      return;
    } else {

      constexpr std::size_t num_iters =
          get_num_iters<iter_level, Num_Iters...>();

      std::cout << "The number of iterations on level " << iter_level << " is "
                << num_iters << std::endl;

      auto epsilon_x_domain = epsilon_x.template get_domain<iter_level>();
      auto epsilon_y_domain = epsilon_y.template get_domain<iter_level>();
      auto epsilon_z_domain = epsilon_z.template get_domain<iter_level>();

      constexpr auto lengths = decltype(epsilon_x.domain)::length;
      std::array<Domain<Dim, std::get<0>(lengths), std::get<1>(lengths),
                        std::get<2>(lengths)>,
                 Dim>
          epsilon_maps{epsilon_x_domain, epsilon_y_domain, epsilon_z_domain};

      for (int j = 0; j < num_iters; j++) {

        pre_smoother(Integer<iter_level>{}, smoother_iters_pre, next, current,
                     rhs_domain, kappa_map, epsilon_x, epsilon_y, epsilon_z,
                     kappa_2, grid_step, epsilon_r, delta_epsilon, omega);

        // This computes -A, in this case
        convolution::PBE_Convolve(current.template get_domain<iter_level>(),
                                  next.template get_domain<iter_level>(),
                                  kappa_map.template get_domain<iter_level>(),
                                  epsilon_maps, kappa_2, grid_step, epsilon_r,
                                  delta_epsilon);

        // This needs to be add, because the Convolve returns the negative
        add_domains(current.template get_domain<iter_level>(),
                    rhs_domain.template get_domain<iter_level>(),
                    current.template get_domain<iter_level>());

        level_transition::coarsening(
            rhs_domain.template get_domain<iter_level - 1>(),
            current.template get_domain<iter_level>(),
            coarsening_operator.template get_values<iter_level>(),
            coarsening_operator.template get_offsets<iter_level>());

        iteration<iter_level - 1>(next, current, rhs_domain, epsilon_x,
                                  epsilon_y, epsilon_z, kappa_map, kappa_2,
                                  2 * grid_step, epsilon_r, delta_epsilon,
                                  omega, num_iters_, coarsening_operator,
                                  smoother_iters_pre, smoother_iters_post);

        level_transition::refinement(
            current.template get_domain<iter_level>(),
            next.template get_domain<iter_level - 1>());

        add_domains(next.template get_domain<iter_level>(),
                    next.template get_domain<iter_level>(),
                    current.template get_domain<iter_level>());
      }

      post_smoother(Integer<iter_level>{}, smoother_iters_pre, current, next,
                    rhs_domain, kappa_map, epsilon_x, epsilon_y, epsilon_z,
                    kappa_2, grid_step, epsilon_r, delta_epsilon, omega);

      //  post_smoother(Integer<iter_level>{}, smoother_iters_post, current,
      //  next,
      //                rhs_domain,
      //                Smooth_operator.template get_values<iter_level>(),
      //                Smooth_operator.template get_offsets<iter_level>(),
      //                box_length, omega);
    }
  }
  Solver &solver;
  inline static Pre_Smoother pre_smoother{};
  inline static Post_Smoother post_smoother{};
};

} // namespace cycles

#endif
