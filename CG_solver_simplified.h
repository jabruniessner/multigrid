#include "Convolution.h"
#include "Domain.h"
#include "cuda_reduce.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <ostream>
#include <utility>

#ifndef CG_SOLVER_H
#define CG_SOLVER_H

namespace cg_solver {

using namespace domain;

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all, std::size_t... dims>
void CG_solver(Domain<Dim, strides_all...> &init_guess,
               Domain<Dim, strides_all...> &rhs,
               Domain<Dim, strides_all...> &defect_r,
               Domain<Dim, strides_all...> &defect_p,
               const std::array<DataType, size> &values,
               const std::array<Offsets, size> &offsets, DataType thresh,
               const std::index_sequence<dims...> &) {

  auto &strides = init_guess.strides;
  Length &padding_width = init_guess.padding_width;

  DataType *r_squared;
  cudaMalloc(&r_squared, sizeof(DataType));
  cudaMemset(r_squared, 0, sizeof(DataType));
  DataType *r_squared_next;
  cudaMalloc(&r_squared_next, sizeof(DataType));
  cudaMemset(r_squared_next, 0, sizeof(DataType));
  DataType *p_squared_A;
  cudaMalloc(&p_squared_A, sizeof(DataType));
  cudaMemset(p_squared_A, 0, sizeof(DataType));
  DataType *alpha;
  cudaMalloc(&alpha, sizeof(DataType));
  cudaMemset(alpha, 0, sizeof(DataType));
  DataType *beta;
  cudaMalloc(&beta, sizeof(DataType));
  cudaMemset(beta, 0, sizeof(DataType));

  int num_required_threads =
      (init_guess.num_dofs + reduction_kernel::seq_size - 1);

  int num_blocks = (num_required_threads + reduction_kernel::gbs - 1) /
                   reduction_kernel::gbs;

  DataType *results;
  cudaMalloc(&results, sizeof(DataType) * num_blocks);
  cudaMemset(results, 0, sizeof(DataType) * num_blocks);

  reduction_kernel::cudaParallelTransformReduce(
      init_guess.num_dofs, r_squared, results, std::plus<>(),

      [=] __device__(std::size_t idx) {
        auto I = domain::flat_to_multi_index<strides_all...>(idx);
        ((I[dims] += padding_width), ...);

        // Computing the convolution for the initial residual
        DataType result = 0;

        for (int k = 0; k < size; k++) {
          result += init_guess((I[dims] + offsets[k][dims])...) * values[k];
        }
        // Assigning values and reducing to the sum
        defect_r(I[dims]...) = defect_p(I[dims]...) = rhs(I[dims]...) - result;

        return defect_r(I[dims]...) * defect_r(I[dims]...);
      });

  //  // Computing the initial pAp
  reduction_kernel::cudaParallelTransformReduce(
      init_guess.num_dofs, p_squared_A, results, std::plus<>(),
      [=] __device__(std::size_t idx) {
        auto I = domain::flat_to_multi_index<strides_all...>(idx);
        ((I[dims] += padding_width), ...);

        // Computing the convolution for the initial residual
        DataType result = 0;
        for (int k = 0; k < size; k++) {
          result += defect_p((I[dims] + offsets[k][dims])...) * values[k];
        }

        return result * defect_p(I[dims]...);
      });

  DataType p_squared_A_value = 0;
  cudaMemcpy(&p_squared_A_value, p_squared_A, sizeof(DataType),
             cudaMemcpyDeviceToHost);

  // Computing initial alpha

  cudaParallelFor(1, 1, [=] __device__() {
    *alpha = (*r_squared) / (*p_squared_A);
    *p_squared_A = 0;
  });

  DataType residual = 0;
  cudaMemcpy(&residual, r_squared, sizeof(DataType), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (p_squared_A_value == 0)
    return;

  residual = std::sqrt(residual);
  thresh = thresh * residual;

  int count = 0;

  while (thresh < residual) {
    // for (int k = 0; k < 90; k++) {
    count++;
    reduction_kernel::cudaParallelTransformReduce(
        init_guess.num_dofs, r_squared_next, results, std::plus<>(),
        [=] __device__(std::size_t idx) {
          auto I = domain::flat_to_multi_index<strides_all...>(idx);
          ((I[dims] += padding_width), ...);
          init_guess(I[dims]...) += (*alpha) * defect_p(I[dims]...);

          DataType result = 0;
          for (int k = 0; k < size; k++) {
            result += defect_p((I[dims] + offsets[k][dims])...) * values[k];
          }

          defect_r(I[dims]...) -= (*alpha) * result;

          return defect_r(I[dims]...) * defect_r(I[dims]...);
        });

    cudaParallelFor(1, 1, [=] __device__() {
      if (*r_squared == 0) {
        *beta = 0.;
      } else {
        *beta = (*r_squared_next) / (*r_squared);
      }
      *r_squared = *r_squared_next;
      *r_squared_next = 0;
    });

    constexpr int block_size = reduction_kernel::gbs;
    int num_blocks = (init_guess.num_dofs + block_size - 1) / block_size;

    cudaParallelFor(num_blocks, block_size, [=] __device__() {
      int gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < init_guess.num_dofs) {
        auto I = domain::flat_to_multi_index<strides_all...>(gid);
        ((I[dims] += padding_width), ...);
        defect_p(I[dims]...) =
            defect_r(I[dims]...) + (*beta) * defect_p(I[dims]...);
      }
    });

    reduction_kernel::cudaParallelTransformReduce(
        init_guess.num_dofs, p_squared_A, results, std::plus<>{},
        [=] __device__(int idx) {
          auto I = domain::flat_to_multi_index<strides_all...>(idx);
          ((I[dims] += padding_width), ...);

          DataType result = 0;
          for (int k = 0; k < size; k++) {
            result += defect_p((I[dims] + offsets[k][dims])...) * values[k];
          }

          return result * defect_p(I[dims]...);
        });

    if (count % 10 == 0) {
      DataType r_squared_value = 0;
      DataType p_squared_A_value = 0;

      cudaMemcpy(&r_squared_value, r_squared, sizeof(DataType),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(&p_squared_A_value, p_squared_A, sizeof(DataType),
                 cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      residual = std::sqrt(r_squared_value / defect_r.num_dofs);
    }

    cudaParallelFor(1, 1, [=] __device__() {
      if (*p_squared_A == 0) {
        *alpha = 0;
      } else {
        *alpha = (*r_squared) / (*p_squared_A);
      }
      *p_squared_A = 0;
    });
  }

  // cudaMemcpy(&residual, r_squared, sizeof(DataType),
  // cudaMemcpyDeviceToHost); cudaDeviceSynchronize(); residual =
  // std::sqrt(residual);
  // std::cout << "We made " << count << " iterations" << std::endl;
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

// template <typename DataType, Dimension Dim, Length... strides_all,
//           std::size_t... dims>
// void CG_solver_PBE(Domain<Dim, strides_all...> &init_guess,
//                    Domain<Dim, strides_all...> &rhs,
//                    Domain<Dim, strides_all...> &defect_r,
//                    Domain<Dim, strides_all...> &defect_p,
//                    Domain<Dim, strides_all...> &kappa_map,
//                    std::array<Domain<Dim, strides_all...>, Dim>
//                    &epsilon_maps, const DataType &kappa_2, const DataType
//                    grid_step, const DataType epsilon_r, const DataType
//                    delta_epsilon, DataType thresh, const
//                    std::index_sequence<dims...> &) {
//   assert(defect_r.q == defect_p.q && init_guess.q == defect_p.q);
//
//   assert(defect_r.num_values == defect_p.num_values &&
//          init_guess.num_values == defect_p.num_values);
//
//   assert(defect_r.padding_width == defect_p.padding_width &&
//          init_guess.padding_width == defect_p.padding_width);
//
//   sycl::queue &q = init_guess.q;
//   auto &strides = init_guess.strides;
//   Length &padding_width = init_guess.padding_width;
//
//   DataType *r_squared = sycl::malloc_device<DataType>(1, q);
//   q.memset(r_squared, 0, sizeof(DataType));
//   DataType *r_squared_next = sycl::malloc_device<DataType>(1, q);
//   q.memset(r_squared_next, 0, sizeof(DataType));
//   DataType *p_squared_A = sycl::malloc_device<DataType>(1, q);
//   q.memset(p_squared_A, 0, sizeof(DataType));
//   DataType *alpha = sycl::malloc_device<DataType>(1, q);
//   q.memset(alpha, 0, sizeof(DataType));
//   DataType *beta = sycl::malloc_device<DataType>(1, q);
//   q.memset(beta, 0, sizeof(DataType));
//
//   q.wait();
//
//   // init_guess.print_domain();
//
//   q.parallel_for(sycl::range<Dim>((strides[dims])...),
//                  sycl::reduction(r_squared, sycl::plus<>()),
//
//                  [=](sycl::id<Dim> I, auto &r) {
//                    ((I[dims] += padding_width), ...);
//
//                    DataType result = -convolution::PBE_Convolve_kernel(
//                        init_guess, kappa_map, epsilon_maps, kappa_2,
//                        grid_step, epsilon_r, delta_epsilon, I);
//
//                    // Computing the convolution for the initial residual
//                    // DataType result = 0;
//
//                    // for (int k = 0; k < size; k++) {
//                    //   result += init_guess((I[dims] + offsets[k][dims])...)
//                    *
//                    //   values[k];
//                    // }
//                    // Assigning values and reducing to the sum
//                    defect_r(I[dims]...) = defect_p(I[dims]...) =
//                        rhs(I[dims]...) - result; // Shouldn't this be a plus?
//
//                    r += defect_r(I[dims]...) * defect_r(I[dims]...);
//                  })
//       .wait();
//
//   // DataType r_squared_host = 0;
//   // q.memcpy(&r_squared_host, r_squared, sizeof(DataType)).wait();
//   // std::cout << "The residual is: " << r_squared_host << std::endl;
//   // init_guess.print_domain();
//
//   // Computing the initial pAp
//   q.parallel_for(sycl::range<Dim>((strides[dims])...),
//                  sycl::reduction(p_squared_A, sycl::plus<>()),
//                  [=](sycl::id<Dim> I, auto &pAp) {
//                    ((I[dims] += padding_width), ...);
//
//                    DataType result = -convolution::PBE_Convolve_kernel(
//                        defect_p, kappa_map, epsilon_maps, kappa_2, grid_step,
//                        epsilon_r, delta_epsilon, I);
//
//                    // Computing the convolution for the initial residual
//                    //  DataType result = 0;
//                    //  for (int k = 0; k < size; k++) {
//                    //    result +=
//                    //        defect_p((I[dims] + offsets[k][dims])...) *
//                    //        values[k];
//                    //  }
//
//                    pAp += result * defect_p(I[dims]...);
//                  })
//       .wait();
//
//   // init_guess.print_domain();
//
//   DataType p_squared_A_value = 0;
//   q.memcpy(&p_squared_A_value, p_squared_A, sizeof(DataType)).wait();
//
//   if (p_squared_A_value == 0)
//     return;
//
//   // Computing initial alpha
//   q.submit([&](sycl::handler &h) {
//      h.single_task([=]() {
//        *alpha = (*r_squared) / (*p_squared_A);
//        *p_squared_A = 0;
//      });
//    }).wait();
//
//   DataType residual = 0;
//   q.memcpy(&residual, r_squared, sizeof(DataType)).wait();
//   residual = std::sqrt(residual);
//   thresh = thresh * residual;
//
//   // std::cout << "The residual before the conjugate gradient is: " <<
//   residual
//   //           << std::endl;
//   int count = 0;
//   // for (int i = 0; i < num_iters; i++)
//   while (thresh < residual) {
//     count++;
//     q.parallel_for(sycl::range<Dim>(strides[dims]...),
//                    sycl::reduction(r_squared_next, sycl::plus<>()),
//                    [=](sycl::id<Dim> I, auto &r_squared_plus_1) {
//                      ((I[dims] += padding_width), ...);
//                      init_guess(I[dims]...) += (*alpha) *
//                      defect_p(I[dims]...);
//
//                      DataType result = -convolution::PBE_Convolve_kernel(
//                          defect_p, kappa_map, epsilon_maps, kappa_2,
//                          grid_step, epsilon_r, delta_epsilon, I);
//                      //  DataType result = 0;
//                      //  for (int k = 0; k < size; k++) {
//                      //    result += defect_p((I[dims] +
//                      offsets[k][dims])...) *
//                      //    values[k];
//                      //  }
//
//                      defect_r(I[dims]...) -= (*alpha) * result;
//
//                      r_squared_plus_1 +=
//                          defect_r(I[dims]...) * defect_r(I[dims]...);
//                    })
//         .wait();
//
//     // init_guess.print_domain();
//
//     q.submit([&](sycl::handler &h) {
//        h.single_task([=]() {
//          if (*r_squared == 0) {
//            *beta = 0.;
//          } else {
//            *beta = (*r_squared_next) / (*r_squared);
//          }
//          *r_squared = *r_squared_next;
//          *r_squared_next = 0;
//        });
//      }).wait();
//
//     q.parallel_for(sycl::range<Dim>(strides[dims]...), [=](sycl::id<Dim> I) {
//        ((I[dims] += padding_width), ...);
//        defect_p(I[dims]...) =
//            defect_r(I[dims]...) + (*beta) * defect_p(I[dims]...);
//      }).wait();
//
//     // init_guess.print_domain();
//
//     q.parallel_for(sycl::range<Dim>(strides[dims]...),
//                    sycl::reduction(p_squared_A, sycl::plus<>()),
//                    [=](sycl::id<Dim> I, auto &pAp) {
//                      ((I[dims] += padding_width), ...);
//
//                      DataType result = -convolution::PBE_Convolve_kernel(
//                          defect_p, kappa_map, epsilon_maps, kappa_2,
//                          grid_step, epsilon_r, delta_epsilon, I);
//                      // DataType result = 0;
//
//                      // for (int k = 0; k < size; k++) {
//                      //   result += defect_p((I[dims] + offsets[k][dims])...)
//                      //   *
//                      //             values[k];
//                      // }
//
//                      pAp += result * defect_p(I[dims]...);
//                    })
//         .wait();
//
//     if (count % 10 == 0) {
//       DataType r_squared_value = 0;
//       DataType p_squared_A_value = 0;
//
//       q.memcpy(&r_squared_value, r_squared, sizeof(DataType));
//       q.memcpy(&p_squared_A_value, p_squared_A, sizeof(DataType)).wait();
//       residual = std::sqrt(r_squared_value / defect_r.num_dofs);
//     }
//
//     //  if (r_squared_value == 0)
//     //    return;
//
//     //  if (p_squared_A_value == 0)
//     //    return;
//     // init_guess.print_domain();
//
//     q.submit([&](sycl::handler &h) {
//        h.single_task([=]() {
//          if (*p_squared_A == 0) {
//            *alpha = 0;
//          } else {
//            *alpha = (*r_squared) / (*p_squared_A);
//          }
//          *p_squared_A = 0;
//        });
//      }).wait();
//   }
//
//   //  std::cout << "We made " << count << " CG iterations." << std::endl;
//   q.wait();
//   q.memcpy(&residual, r_squared, sizeof(DataType)).wait();
//   residual = std::sqrt(residual);
//   // std::cout << "The residual after the CG is: " << residual << std::endl;
// }
//
// template <typename DataType, Dimension Dim, Length... strides_all,
//           std::size_t... dims>
// void CG_solver_PBE(Domain<Dim, strides_all...> &init_guess,
//                    Domain<Dim, strides_all...> &rhs,
//                    Domain<Dim, strides_all...> &defect_r,
//                    Domain<Dim, strides_all...> &defect_p,
//                    Domain<Dim, strides_all...> &kappa_map,
//                    std::array<Domain<Dim, strides_all...>, Dim>
//                    &epsilon_maps, const DataType &kappa_2, const DataType
//                    grid_step, const DataType epsilon_r, const DataType
//                    delta_epsilon, DataType thresh) {
//   CG_solver_PBE(init_guess, rhs, defect_r, defect_p, kappa_map, epsilon_maps,
//                 kappa_2, grid_step, epsilon_r, delta_epsilon, thresh,
//                 std::make_index_sequence<Dim>{});
// }
//
// template <typename DataType, typename Offsets, std::size_t size,
//           DataType thresh, Dimension Dim, Length... strides_all>
// struct PBE_Solver_CG {
//   PBE_Solver_CG(Float<thresh>, Domain<Dim, strides_all...> &sample_domain,
//                 const std::array<DataType, size> &,
//                 const std::array<Offsets, size> &)
//       : defect_r(Paddings::PERIODIC, sample_domain.q, 1),
//         defect_p(Paddings::PERIODIC, sample_domain.q, 1) {};
//
//   void operator()(Domain<Dim, strides_all...> &init_guess,
//                   Domain<Dim, strides_all...> &rhs,
//                   // Domain<Dim, strides_all...> &defect_r,
//                   // Domain<Dim, strides_all...> &defectr_p,
//                   Domain<Dim, strides_all...> &kappa_map,
//                   std::array<Domain<Dim, strides_all...>, Dim> &epsilon_maps,
//                   const DataType &kappa_2, const DataType grid_step,
//                   const DataType epsilon_r, const DataType delta_epsilon) {
//     CG_solver_PBE(init_guess, rhs, defect_r, defect_p, kappa_map,
//     epsilon_maps,
//                   kappa_2, grid_step, epsilon_r, delta_epsilon, thresh);
//   }
//
//   Domain<Dim, strides_all...> defect_r;
//   Domain<Dim, strides_all...> defect_p;
// };

} // namespace cg_solver

#endif
