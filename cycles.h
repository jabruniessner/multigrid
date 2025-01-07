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

    constexpr std::size_t num_iters = get_num_iters<level, Num_Iters...>();

    auto &dest_domain = dest.template get_domain<level>();
    auto &src_domain = src.template get_domain<level>();
    auto &rhs_domain = rhs.template get_domain<level>();
    const DataType h =
        box_length / (std::get<0>(rhs.template get_length<level>()) + 1);
    const DataType diag_inverse = omega * (h * h) / 4;

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

        post_smoother(Integer<iter_level>{}, smoother_iters_post, current, next,
                      rhs_domain,
                      Smooth_operator.template get_values<iter_level>(),
                      Smooth_operator.template get_offsets<iter_level>(),
                      box_length, omega);
      }
    }
  }
  Solver &solver;
  inline static Pre_Smoother pre_smoother{};
  inline static Post_Smoother post_smoother{};
};

} // namespace cycles

#endif
