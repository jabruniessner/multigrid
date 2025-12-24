#include "Convolution.h"
#include "Domain.h"
#include "bitshift_lib.h"
#include "concepts.h"
#include "hipSYCL/sycl/libkernel/reduction.hpp"
#include "tprint.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <ostream>
#include <utility>

#ifndef CG_SOLVER_H
#define CG_SOLVER_H

namespace cg_solver {

using namespace domain;

struct IdentityPreconditioner {
  template <class Vec> Vec operator()(const Vec r) const { return r; }
};

template <typename DataType> struct thresh_criterion {
  DataType thresh;
  DataType thresh_;

  thresh_criterion(DataType thresh) : thresh{thresh}, thresh_{} {}

  bool operator()(DataType residual) { return (residual > thresh); }

  void initialize_residual_dependancy(DataType residual) {
    this->thresh_ = this->thresh * residual;
  };
};

struct counting_criterion {
  std::size_t num_iters;
  std::size_t current_iters = 0;
  counting_criterion(std::size_t num_iters) : num_iters{num_iters} {};
  bool operator()(DataType residual) { return (num_iters > current_iters++); }
};

template <typename DataType, Dimension Dim, typename Finish_crit,
          Length... strides_all, std::size_t... dims>
void CG_solver(Domain<Dim, strides_all...> &init_guess,
               Domain<Dim, strides_all...> &rhs,
               Domain<Dim, strides_all...> &defect_r,
               Domain<Dim, strides_all...> &defect_p, auto map,
               Finish_crit thresh, const std::index_sequence<dims...> &,
               auto precond = IdentityPreconditioner{}, bool verbose = false) {
  assert(defect_r.q == defect_p.q && init_guess.q == defect_p.q);

  assert(defect_r.num_values == defect_p.num_values &&
         init_guess.num_values == defect_p.num_values);

  assert(defect_r.padding_width == defect_p.padding_width &&
         init_guess.padding_width == defect_p.padding_width);

  // std::cout << "Using the preconditioned Conjugate Gradient" << std::endl;

  sycl::queue &q = init_guess.q;
  auto &strides = init_guess.strides;
  Length &padding_width = init_guess.padding_width;

  DataType *r_squared = sycl::malloc_device<DataType>(1, q);
  q.memset(r_squared, 0, sizeof(DataType));
  DataType *r_squared_next = sycl::malloc_device<DataType>(1, q);
  q.memset(r_squared_next, 0, sizeof(DataType));
  DataType *p_squared_A = sycl::malloc_device<DataType>(1, q);
  q.memset(p_squared_A, 0, sizeof(DataType));
  DataType *alpha = sycl::malloc_device<DataType>(1, q);
  q.memset(alpha, 0, sizeof(DataType));
  DataType *beta = sycl::malloc_device<DataType>(1, q);
  q.memset(beta, 0, sizeof(DataType));

  // q.wait();

  // init_guess.print_domain();

  q.parallel_for(sycl::range<Dim>((strides[dims])...), [=](sycl::id<Dim> I) {
    ((I[dims] += padding_width), ...);

    // Computing the convolution for the initial residual
    DataType result = map(init_guess, I);

    // Assigning values and reducing to the sum
    defect_r(I[dims]...) = rhs(I[dims]...) - result;
  });

  auto s = precond(defect_r);
  q.memcpy(defect_p.values_buff, s.values_buff,
           sizeof(DataType) * defect_p.num_values);

  q.parallel_for(sycl::range<Dim>((strides[dims])...),
                 sycl::reduction(r_squared, sycl::plus<>()),
                 [=](sycl::id<Dim> I, auto &r) {
                   ((I[dims] += padding_width), ...);
                   r += defect_r(I[dims]...) * s(I[dims]...);
                 });

  // init_guess.print_domain();

  // Computing the initial pAp
  q.parallel_for(sycl::range<Dim>((strides[dims])...),
                 sycl::reduction(p_squared_A, sycl::plus<>()),
                 [=](sycl::id<Dim> I, auto &pAp) {
                   ((I[dims] += padding_width), ...);

                   // Computing the convolution for the initial residual
                   DataType result = map(defect_p, I);

                   pAp += result * defect_p(I[dims]...);
                 });

  // init_guess.print_domain();

  DataType p_squared_A_value = 0;
  q.memcpy(&p_squared_A_value, p_squared_A, sizeof(DataType)).wait();

  if (p_squared_A_value == 0)
    return;

  // Computing initial alpha
  q.submit([&](sycl::handler &h) {
     h.single_task([=]() {
       *alpha = (*r_squared) / (*p_squared_A);
       *p_squared_A = 0;
     });
   }).wait();

  DataType residual = 0;
  q.memcpy(&residual, r_squared, sizeof(DataType)).wait();
  residual = std::sqrt(residual);
  if constexpr (has_initialize_residual_dependancy<decltype(thresh)>) {
    thresh.initialize_residual_dependancy(residual);
  }

  residual = std::sqrt(residual);

  //  std::cout << "The residual before the conjugate gradient is: " << residual
  //            << std::endl;
  //
  //  std::cout << "The threshold is: " << thresh << std::endl;
  DataType true_residual;
  domain_compute_norm_squared(true_residual, defect_r);

  auto residual_init = std::sqrt(true_residual);

  int count = 0;
  while (thresh(residual)) {

    if (verbose) {
      domain_compute_norm_squared(true_residual, defect_r);
      true_residual = std::sqrt(true_residual);
      std::cout << "The residual after " << count << " iterations is "
                << true_residual / residual_init << std::endl;
    }

    count++;
    //  if (count % 1000 == 0) {
    //    std::cout << "The residual after " << count << " iterations is "
    //              << residual << std::endl;
    //  }

    q.parallel_for(sycl::range<Dim>(strides[dims]...), [=](sycl::id<Dim> I) {
      ((I[dims] += padding_width), ...);
      init_guess(I[dims]...) += (*alpha) * defect_p(I[dims]...);

      DataType result = map(defect_p, I);

      defect_r(I[dims]...) -= (*alpha) * result;
    });

    auto s = precond(defect_r);

    q.parallel_for(sycl::range<Dim>(strides[dims]...),
                   sycl::reduction(r_squared_next, sycl::plus<>()),
                   [=](sycl::id<Dim> I, auto &r_squared_plus_1) {
                     ((I[dims] += padding_width), ...);
                     r_squared_plus_1 += defect_r(I[dims]...) * s(I[dims]...);
                   });

    // init_guess.print_domain();

    q.submit([&](sycl::handler &h) {
      h.single_task([=]() {
        if (*r_squared == 0) {
          *beta = 0.;
        } else {
          *beta = (*r_squared_next) / (*r_squared);
        }
        *r_squared = *r_squared_next;
        *r_squared_next = 0;
      });
    });

    q.parallel_for(sycl::range<Dim>(strides[dims]...), [=](sycl::id<Dim> I) {
      ((I[dims] += padding_width), ...);
      defect_p(I[dims]...) = s(I[dims]...) + (*beta) * defect_p(I[dims]...);
    });

    // init_guess.print_domain();

    q.parallel_for(sycl::range<Dim>(strides[dims]...),
                   sycl::reduction(p_squared_A, sycl::plus<>()),
                   [=](sycl::id<Dim> I, auto &pAp) {
                     ((I[dims] += padding_width), ...);

                     DataType result = map(defect_p, I);
                     pAp += result * defect_p(I[dims]...);
                   });

    //  if (r_squared_value == 0)
    //    return;

    if (p_squared_A_value == 0)
      return;
    // init_guess.print_domain();

    q.submit([&](sycl::handler &h) {
      h.single_task([=]() {
        if (*p_squared_A == 0) {
          *alpha = 0;
        } else {
          *alpha = (*r_squared) / (*p_squared_A);
        }
        *p_squared_A = 0;
      });
    });

    q.memcpy(&residual, r_squared, sizeof(DataType)).wait();
    residual = std::sqrt(residual);
  }

  //  std::cout << "We made " << count << " CG iterations." << std::endl;
  q.wait();
  q.memcpy(&residual, r_squared, sizeof(DataType)).wait();
  residual = std::sqrt(residual);
  // std::cout << "The residual after the CG is: " << residual << std::endl;
}

template <typename DataType, Dimension Dim,
          typename Finish_crit = thresh_criterion<DataType>,
          Length... strides_all>
void CG_solver(Domain<Dim, strides_all...> &init_guess,
               Domain<Dim, strides_all...> &rhs,
               Domain<Dim, strides_all...> &defect_r,
               Domain<Dim, strides_all...> &defect_p, auto map, Finish_crit m,
               auto precond = IdentityPreconditioner{}, bool verbose = false) {
  // std::cout << "The value for verbose is " << verbose << std::endl;
  CG_solver<DataType, Dim, Finish_crit, strides_all...>(
      init_guess, rhs, defect_r, defect_p, map, m,
      std::make_index_sequence<Dim>(), precond, verbose);
}

template <typename DataType, DataType thresh, Dimension Dim,
          Length... strides_all>
struct Solver_CG {
  Solver_CG(Float<thresh>, Domain<Dim, strides_all...> &sample_domain)
      : defect_r(Paddings::PERIODIC, sample_domain.q, 1),
        defect_p(Paddings::PERIODIC, sample_domain.q, 1) {};

  template <class preconditioner = IdentityPreconditioner,
            class Finish_crit = thresh_criterion<DataType>>
  void operator()(Domain<Dim, strides_all...> &init_guess,
                  Domain<Dim, strides_all...> &rhs, auto map,
                  preconditioner precond = IdentityPreconditioner{}) {
    CG_solver<DataType, Dim, Finish_crit, strides_all...>(
        init_guess, rhs, defect_r, defect_p, map, Finish_crit(thresh), precond);
  }

  Domain<Dim, strides_all...> defect_r;
  Domain<Dim, strides_all...> defect_p;
};

template <typename DataType, Dimension Dim, Length... strides_all>
struct Solver_CG_general {
  Solver_CG_general(Domain<Dim, strides_all...> &sample_domain)
      : defect_r(Paddings::PERIODIC, sample_domain.q, 1),
        defect_p(Paddings::PERIODIC, sample_domain.q, 1) {};

  template <class Finish_crit, class preconditioner = IdentityPreconditioner>
  void operator()(Domain<Dim, strides_all...> &init_guess,
                  Domain<Dim, strides_all...> &rhs, auto map,
                  Finish_crit finish,
                  preconditioner precond = IdentityPreconditioner{},
                  bool verbose = false) {

    // std::cout << "The value for verbose is: " << verbose << std::endl;
    CG_solver<DataType, Dim, Finish_crit, strides_all...>(
        init_guess, rhs, defect_r, defect_p, map, finish, precond, verbose);
  }

  Domain<Dim, strides_all...> defect_r;
  Domain<Dim, strides_all...> defect_p;
};

template <typename DataType, Dimension Dim, Length... strides_all>
auto make_general_solver(Domain<Dim, strides_all...> &domain) {
  return Solver_CG_general<DataType, Dim, strides_all...>(domain);
}

template <typename DataType, DataType thresh, Dimension Dim,
          Length... strides_all>
Solver_CG(Float<thresh>, Domain<Dim, strides_all...>)
    -> Solver_CG<DataType, thresh, Dim, strides_all...>;

template <typename DataType, Dimension Dim, DataType thresh,
          Length... strides_all>
auto make_solver(Float<thresh> float_num, Domain<Dim, strides_all...> &domain) {
  return Solver_CG<DataType, thresh, Dim, strides_all...>{float_num, domain};
}

} // namespace cg_solver

#endif
