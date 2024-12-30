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

template <Dimension Dim, Length... strides_all, typename DataType,
          typename OffsetType, size_t size>
DataType compute_residual(Domain<Dim, strides_all...> &rhs,
                          Domain<Dim, strides_all...> &sol,
                          Domain<Dim, strides_all...> &helper,
                          std::array<DataType, size> &values,
                          std::array<OffsetType, size> &offsets) {
  convolution::Subtract_Convolve(helper, sol, rhs, values, offsets);
  DataType result;
  domain_compute_norm_squared(result, helper);
  return std::sqrt(result / sol.num_values);
}

template <std::size_t Num_Iters, Dimension Dim, typename DataType,
          typename OffsetType, std::size_t length, std::size_t base_length,
          std::size_t nlev>
struct Jacobi_Smoother {
  Jacobi_Smoother() {};

  Jacobi_Smoother(Integer<Num_Iters> integer,
                  Multigrid_domain<Dim, base_length, nlev>,
                  std::array<DataType, length> values,
                  std::array<OffsetType, length> offsets) {};

  template <std::size_t level = nlev>
  void operator()(Integer<level>,
                  Multigrid_domain<Dim, base_length, nlev> &dest,
                  Multigrid_domain<Dim, base_length, nlev> &src,
                  Multigrid_domain<Dim, base_length, nlev> &rhs,
                  std::array<DataType, length> &values,
                  std::array<OffsetType, length> &offsets, DataType &box_length)

  {

    auto &dest_domain = dest.template get_domain<level>();
    auto &src_domain = src.template get_domain<level>();
    auto &rhs_domain = rhs.template get_domain<level>();
    const DataType h = box_length / (rhs.template get_length<level>() + 1);
    const DataType diag_inverse = (h * h) / 4;

    if constexpr (Num_Iters == 0) {
      return;
    } else {
      for (int i = 0; i < Num_Iters; i++) {
        convolution::Convolve(dest_domain, src_domain, values, offsets);
        subtract_and_multiply_domains(dest_domain, rhs_domain, dest_domain,
                                      diag_inverse);

        std::swap(dest_domain.values_buff, src_domain.values_buff);
      }

      std::swap(dest_domain.values_buff, src_domain.values_buff);
    }
  }

  void operator()(Multigrid_domain<Dim, base_length, nlev> &dest,
                  Multigrid_domain<Dim, base_length, nlev> &src,
                  Multigrid_domain<Dim, base_length, nlev> &rhs,
                  std::array<DataType, length> &values,
                  std::array<OffsetType, length> &offsets, DataType &box_length)

  {
    this->operator()(Integer<nlev>{}, dest, src, rhs, values, offsets,
                     box_length);
  }
};

template <typename Pre_Smoother, typename Post_Smoother, typename Solver,
          Dimension Dim, std::size_t base_length, std::size_t length,
          std::size_t length_diff_op, std::size_t length_coarsening_op,
          typename DataType, std::size_t nlev, std::size_t level = nlev>
struct V_Cycle_base {
  V_Cycle_base(
      Pre_Smoother &presmoother, Post_Smoother &post_smoother, Solver &solver,
      Multigrid_domain<Dim, base_length, nlev> &,
      Multi_Level_operator<Dim, DataType, length, base_length, nlev> &,
      Multi_Level_operator<Dim, DataType, length_diff_op, base_length, nlev> &,
      Multi_Level_operator<Dim, DataType, length_coarsening_op, base_length,
                           nlev> &)
      : solver(solver) {}

  template <std::size_t iter_level = level>
  void iteration(Multigrid_domain<Dim, base_length, nlev> &next,
                 Multigrid_domain<Dim, base_length, nlev> &current,
                 Multigrid_domain<Dim, base_length, nlev> &rhs_domain,
                 Multi_Level_operator<Dim, DataType, length, base_length, nlev>
                     &Smooth_operator,
                 Multi_Level_operator<Dim, DataType, length_diff_op,
                                      base_length, nlev> &Diff_operator,
                 Multi_Level_operator<Dim, DataType, length_coarsening_op,
                                      base_length, nlev> &coarsening_operator,
                 DataType box_length) {

    if constexpr (iter_level == 1) {
      solver(next.template get_domain<iter_level>(),
             rhs_domain.template get_domain<iter_level>(),
             Diff_operator.template get_values<iter_level>(),
             Diff_operator.template get_offsets<iter_level>());
      return;
    } else {
      pre_smoother(Integer<iter_level>{}, next, current, rhs_domain,
                   Smooth_operator.template get_values<iter_level>(),
                   Smooth_operator.template get_offsets<iter_level>(),
                   box_length);

      std::cout << "After presmoothing the state is: " << std::endl;
      std::cout << "Next: " << std::endl;

      print_multigrid_domain(next);

      std::cout << "Current: " << std::endl;
      print_multigrid_domain(current);

      // Computing offsets
      convolution::Convolve(current.template get_domain<iter_level>(),
                            next.template get_domain<iter_level>(),
                            Diff_operator.template get_values<iter_level>(),
                            Diff_operator.template get_offsets<iter_level>());

      subtract_domains(current.template get_domain<iter_level>(),
                       rhs_domain.template get_domain<iter_level>(),
                       current.template get_domain<iter_level>());
      // Finished computing the offset

      // Now we need to coarsen the domain
      level_transition::coarsening_and_copy(
          rhs_domain.template get_domain<iter_level - 1>(),
          current.template get_domain<iter_level - 1>(),
          current.template get_domain<iter_level>(),
          coarsening_operator.template get_values<iter_level>(),
          coarsening_operator.template get_offsets<iter_level>());

      //      level_transition::coarsening(
      //          current.template get_domain<iter_level - 1>(),
      //          current.template get_domain<iter_level>(),
      //          coarsening_operator.template get_values<iter_level>(),
      //          coarsening_operator.template get_offsets<iter_level>());

      iteration<iter_level - 1>(next, current, rhs_domain, Smooth_operator,
                                Diff_operator, coarsening_operator, box_length);

      level_transition::refinement(current.template get_domain<iter_level>(),
                                   next.template get_domain<iter_level - 1>());

      add_domains(next.template get_domain<iter_level>(),
                  next.template get_domain<iter_level>(),
                  current.template get_domain<iter_level>());

      // post_smoother(Integer<iter_level>{}, current, next, rhs_domain,
      //               Smooth_operator.template get_values<iter_level>(),
      //               Smooth_operator.template get_offsets<iter_level>());
    }
  }
  Solver &solver;
  inline static Pre_Smoother pre_smoother{};
  inline static Post_Smoother post_smoother{};
};

} // namespace cycles

#endif
