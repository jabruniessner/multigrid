#include "Convolution.h"
#include "Domain.h"
#include <algorithm>
#include <array>
#include <boost/iterator/counting_iterator.hpp>
#include <cmath>
#include <cstddef>
#include <execution>
#include <numeric>
#include <ostream>
#include <utility>

#ifndef CG_SOLVER_H
#define CG_SOLVER_H

namespace cg_solver {

using namespace domain;

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all, std::size_t... dims>
void CG_solver(Domain<Dim, strides_all...> init_guess,
               Domain<Dim, strides_all...> rhs,
               Domain<Dim, strides_all...> defect_r,
               Domain<Dim, strides_all...> defect_p,
               const std::array<DataType, size> values,
               const std::array<Offsets, size> offsets, DataType thresh,
               const std::index_sequence<dims...> &) {

  auto &strides = init_guess.strides;
  Length &padding_width = init_guess.padding_width;

  DataType r_squared_next = 0;
  DataType beta = 0;

  boost::iterators::counting_iterator<int> begin(0);
  boost::iterators::counting_iterator<int> end(init_guess.num_dofs);

  DataType r_squared = std::transform_reduce(
      std::execution::par_unseq, begin, end, 0.0, std::plus<>{}, [=](int idx) {
        auto I = domain::flat_to_multi_index<strides_all...>(idx);
        ((I[dims] += init_guess.padding_width), ...);
        // Computing the convolution for the initial residual
        DataType result = 0;

        for (int k = 0; k < size; k++) {
          result += init_guess((I[dims] + offsets[k][dims])...) * values[k];
        }
        // Assigning values and reducing to the sum
        defect_r(I[dims]...) = defect_p(I[dims]...) = rhs(I[dims]...) - result;

        return defect_r(I[dims]...) * defect_r(I[dims]...);
      });
  // init_guess.print_domain();

  DataType p_squared_A = std::transform_reduce(
      std::execution::par_unseq, begin, end, 0.0, std::plus<>{}, [=](int idx) {
        auto I = domain::flat_to_multi_index<strides_all...>(idx);
        ((I[dims] += init_guess.padding_width), ...);

        // Computing the convolution for the initial residual
        DataType result = 0;
        for (int k = 0; k < size; k++) {
          result += defect_p((I[dims] + offsets[k][dims])...) * values[k];
        }

        return result * defect_p(I[dims]...);
      });

  // init_guess.print_domain();

  if (p_squared_A == 0)
    return;

  // Computing initial alpha

  DataType alpha = r_squared / p_squared_A;

  DataType residual = r_squared;
  residual = std::sqrt(residual);
  thresh = thresh * residual;

  // std::cout << "The residual before the conjugate gradient is: " <<
  // residual
  //           << std::endl;

  int count = 0;
  while (thresh < residual) {
    count++;

    DataType r_squared_next = std::transform_reduce(
        std::execution::par_unseq, begin, end, 0.0, std::plus<>{},
        [=](int idx) {
          auto I = domain::flat_to_multi_index<strides_all...>(idx);
          ((I[dims] += padding_width), ...);
          // Update the solution
          init_guess(I[dims]...) += (alpha)*defect_p(I[dims]...);

          // Computing the convolution for the residual update
          DataType result = 0;
          for (int k = 0; k < size; k++) {
            result += defect_p((I[dims] + offsets[k][dims])...) * values[k];
          }

          // Updating the residual
          defect_r(I[dims]...) -= (alpha)*result;

          return defect_r(I[dims]...) * defect_r(I[dims]...);
        });

    // init_guess.print_domain();
    DataType beta = r_squared == 0 ? 0. : r_squared_next / r_squared;
    r_squared = r_squared_next;
    r_squared_next = 0;

    std::for_each(std::execution::par_unseq, begin, end, [=](int idx) {
      auto I = domain::flat_to_multi_index<strides_all...>(idx);
      ((I[dims] += padding_width), ...);
      defect_p(I[dims]...) = defect_r(I[dims]...) + (beta)*defect_p(I[dims]...);
    });

    p_squared_A = std::transform_reduce(
        std::execution::par_unseq, begin, end, 0.0, std::plus<>{},
        [=](int idx) {
          auto I = domain::flat_to_multi_index<strides_all...>(idx);
          ((I[dims] += padding_width), ...);

          DataType result = 0;
          for (int k = 0; k < size; k++) {
            result += defect_p((I[dims] + offsets[k][dims])...) * values[k];
          }

          return result * defect_p(I[dims]...);
        });

    if (count % 10 == 0)
      residual = std::sqrt(r_squared / defect_r.num_dofs);

    alpha = p_squared_A == 0 ? 0. : r_squared / p_squared_A;
    p_squared_A = 0;
  }

  //  std::cout << "We made " << count << " CG iterations." << std::endl;
  residual = std::sqrt(residual);
  // std::cout << "The residual after the CG is: " << residual << std::endl;
}

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all>
void CG_solver(Domain<Dim, strides_all...> &init_guess,
               Domain<Dim, strides_all...> &rhs,
               Domain<Dim, strides_all...> &defect_r,
               Domain<Dim, strides_all...> &defect_p,
               const std::array<DataType, size> &values,
               const std::array<Offsets, size> &offsets, DataType m) {
  CG_solver(init_guess, rhs, defect_r, defect_p, values, offsets, m,
            std::make_index_sequence<Dim>());
}

template <typename DataType, typename Offsets, std::size_t size,
          DataType thresh, Dimension Dim, Length... strides_all>
struct Solver_CG {
  Solver_CG(Float<thresh>, Domain<Dim, strides_all...> &sample_domain,
            const std::array<DataType, size> &,
            const std::array<Offsets, size> &)
      : defect_r(Paddings::PERIODIC, 1), defect_p(Paddings::PERIODIC, 1) {};

  void operator()(Domain<Dim, strides_all...> &init_guess,
                  Domain<Dim, strides_all...> &rhs,
                  const std::array<DataType, size> &values,
                  const std::array<Offsets, size> &offsets) {
    CG_solver(init_guess, rhs, defect_r, defect_p, values, offsets, thresh);
  }

  Domain<Dim, strides_all...> defect_r;
  Domain<Dim, strides_all...> defect_p;
};

template <typename DataType, Dimension Dim, Length... strides_all,
          std::size_t... dims>
void CG_solver_PBE(Domain<Dim, strides_all...> &init_guess,
                   Domain<Dim, strides_all...> &rhs,
                   Domain<Dim, strides_all...> &defect_r,
                   Domain<Dim, strides_all...> &defect_p,
                   Domain<Dim, strides_all...> &kappa_map,
                   std::array<Domain<Dim, strides_all...>, Dim> &epsilon_maps,
                   const DataType &kappa_2, const DataType grid_step,
                   const DataType epsilon_r, const DataType delta_epsilon,
                   DataType thresh, const std::index_sequence<dims...> &) {

  auto &strides = init_guess.strides;
  Length &padding_width = init_guess.padding_width;
  boost::iterators::counting_iterator<int> begin(0);
  boost::iterators::counting_iterator<int> end(init_guess.num_dofs);

  // init_guess.print_domain();
  //
  DataType r_squared = std::transform_reduce(
      std::execution::par_unseq, begin, end, 0.0, std::plus<>{}, [=](int idx) {
        auto I = domain::flat_to_multi_index<strides_all...>(idx);
        ((I[dims] += padding_width), ...);

        DataType result = -convolution::PBE_Convolve_kernel(
            init_guess, kappa_map, epsilon_maps, kappa_2, grid_step, epsilon_r,
            delta_epsilon, I);

        defect_r(I[dims]...) = defect_p(I[dims]...) =
            rhs(I[dims]...) - result; // Shouldn't this be a plus?

        return defect_r(I[dims]...) * defect_r(I[dims]...);
      });

  // DataType r_squared_host = 0;
  // q.memcpy(&r_squared_host, r_squared, sizeof(DataType)).wait();
  // std::cout << "The residual is: " << r_squared_host << std::endl;
  // init_guess.print_domain();

  // Computing the initial pAp
  //
  DataType p_squared_A = std::transform_reduce(
      std::execution::par_unseq, begin, end, 0.0, std::plus<>{}, [=](int idx) {
        auto I = domain::flat_to_multi_index<strides_all...>(idx);

        ((I[dims] += padding_width), ...);

        DataType result = -convolution::PBE_Convolve_kernel(
            defect_p, kappa_map, epsilon_maps, kappa_2, grid_step, epsilon_r,
            delta_epsilon, I);

        return result * defect_p(I[dims]...);
      });

  // init_guess.print_domain();

  if (p_squared_A == 0)
    return;

  DataType alpha = r_squared / p_squared_A;

  DataType residual = std::sqrt(r_squared);
  thresh = thresh * residual;

  // std::cout << "The residual before the conjugate gradient is: " << residual
  //           << std::endl;
  int count = 0;
  // for (int i = 0; i < num_iters; i++)
  while (thresh < residual) {
    count++;

    DataType r_squared_next = std::transform_reduce(
        std::execution::par_unseq, begin, end, 0.0, std::plus<>{},
        [=](int idx) {
          auto I = domain::flat_to_multi_index<strides_all...>(idx);

          ((I[dims] += padding_width), ...);
          init_guess(I[dims]...) += (*alpha) * defect_p(I[dims]...);

          DataType result = -convolution::PBE_Convolve_kernel(
              defect_p, kappa_map, epsilon_maps, kappa_2, grid_step, epsilon_r,
              delta_epsilon, I);
          //  DataType result = 0;
          //  for (int k = 0; k < size; k++) {
          //    result += defect_p((I[dims] + offsets[k][dims])...) *
          //    values[k];
          //  }

          defect_r(I[dims]...) -= (*alpha) * result;

          return defect_r(I[dims]...) * defect_r(I[dims]...);
        });

    // init_guess.print_domain();
    // init_guess.print_domain();
    DataType beta = r_squared == 0 ? 0. : r_squared_next / r_squared;
    r_squared = r_squared_next;
    r_squared_next = 0;

    std::for_each(std::execution::par_unseq, begin, end, [=](int idx) {
      auto I = domain::flat_to_multi_index<strides_all...>(idx);
      ((I[dims] += padding_width), ...);
      defect_p(I[dims]...) = defect_r(I[dims]...) + (beta)*defect_p(I[dims]...);
    });

    // init_guess.print_domain();

    p_squared_A = std::transform_reduce(
        std::execution::par_unseq, begin, end, 0.0, std::plus<>{},
        [=](int idx) {
          auto I = domain::flat_to_multi_index<strides_all...>(idx);

          ((I[dims] += padding_width), ...);

          DataType result = -convolution::PBE_Convolve_kernel(
              defect_p, kappa_map, epsilon_maps, kappa_2, grid_step, epsilon_r,
              delta_epsilon, I);
          // DataType result = 0;

          // for (int k = 0; k < size; k++) {
          //   result += defect_p((I[dims] + offsets[k][dims])...)
          //   *
          //             values[k];
          // }

          return result * defect_p(I[dims]...);
        });

    if (count % 10 == 0)
      residual = std::sqrt(r_squared / defect_r.num_dofs);

    //  if (r_squared_value == 0)
    //    return;

    //  if (p_squared_A_value == 0)
    //    return;
    // init_guess.print_domain();

    alpha = p_squared_A == 0 ? 0. : r_squared / p_squared_A;
    p_squared_A = 0;
  }

  //  std::cout << "We made " << count << " CG iterations." << std::endl;
  residual = std::sqrt(r_squared);
  // std::cout << "The residual after the CG is: " << residual << std::endl;
}

template <typename DataType, Dimension Dim, Length... strides_all,
          std::size_t... dims>
void CG_solver_PBE(Domain<Dim, strides_all...> &init_guess,
                   Domain<Dim, strides_all...> &rhs,
                   Domain<Dim, strides_all...> &defect_r,
                   Domain<Dim, strides_all...> &defect_p,
                   Domain<Dim, strides_all...> &kappa_map,
                   std::array<Domain<Dim, strides_all...>, Dim> &epsilon_maps,
                   const DataType &kappa_2, const DataType grid_step,
                   const DataType epsilon_r, const DataType delta_epsilon,
                   DataType thresh) {
  CG_solver_PBE(init_guess, rhs, defect_r, defect_p, kappa_map, epsilon_maps,
                kappa_2, grid_step, epsilon_r, delta_epsilon, thresh,
                std::make_index_sequence<Dim>{});
}

template <typename DataType, typename Offsets, std::size_t size,
          DataType thresh, Dimension Dim, Length... strides_all>
struct PBE_Solver_CG {
  PBE_Solver_CG(Float<thresh>, Domain<Dim, strides_all...> &sample_domain,
                const std::array<DataType, size> &,
                const std::array<Offsets, size> &)
      : defect_r(Paddings::PERIODIC, sample_domain.q, 1),
        defect_p(Paddings::PERIODIC, sample_domain.q, 1) {};

  void operator()(Domain<Dim, strides_all...> &init_guess,
                  Domain<Dim, strides_all...> &rhs,
                  // Domain<Dim, strides_all...> &defect_r,
                  // Domain<Dim, strides_all...> &defectr_p,
                  Domain<Dim, strides_all...> &kappa_map,
                  std::array<Domain<Dim, strides_all...>, Dim> &epsilon_maps,
                  const DataType &kappa_2, const DataType grid_step,
                  const DataType epsilon_r, const DataType delta_epsilon) {
    CG_solver_PBE(init_guess, rhs, defect_r, defect_p, kappa_map, epsilon_maps,
                  kappa_2, grid_step, epsilon_r, delta_epsilon, thresh);
  }

  Domain<Dim, strides_all...> defect_r;
  Domain<Dim, strides_all...> defect_p;
};

} // namespace cg_solver

#endif
