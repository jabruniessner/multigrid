#include "Domain.h"
#include "predefinitions.h"
#include <array>
#include <sycl/sycl.hpp>
#include <utility>

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

namespace convolution {

using namespace domain;

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all, std::size_t... dims>

int Convolve(Domain<Dim, strides_all...> &dest,
             Domain<Dim, strides_all...> &src,
             const std::array<DataType, size> &values,
             const std::array<Offsets, size> &offsets,
             std::index_sequence<dims...>) {
  assert(dest.q == src.q);
  assert(dest.padding_width == src.padding_width);

  dest.q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::range<Dim>(dest.strides[dims]...), [=](sycl::id<Dim> I) {
          ((I[dims] += dest.padding_width), ...);

          DataType result = 0;

          for (int k = 0; k < size; k++) {
            result += src((I[dims] + offsets[k][dims])...) * values[k];
          }

          dest(I[dims]...) = result;
        });
  });

  // dest.q.wait();

  return 0;
}

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all>
int Convolve(Domain<Dim, strides_all...> &dest,
             Domain<Dim, strides_all...> &src,
             const std::array<DataType, size> &values,
             const std::array<Offsets, size> &offsets) {
  return Convolve(dest, src, values, offsets, std::make_index_sequence<Dim>());
}

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all, std::size_t... dims>

int Subtract_Convolve(Domain<Dim, strides_all...> &dest,
                      Domain<Dim, strides_all...> &src,
                      Domain<Dim, strides_all...> &rhs,
                      const std::array<DataType, size> &values,
                      const std::array<Offsets, size> &offsets,
                      std::index_sequence<dims...>) {
  assert(dest.q == src.q);
  assert(dest.padding_width == src.padding_width);

  dest.q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::range<Dim>(dest.strides[dims]...), [=](sycl::id<Dim> I) {
          ((I[dims] += dest.padding_width), ...);

          DataType result = 0;

          for (int k = 0; k < size; k++) {
            result += src((I[dims] + offsets[k][dims])...) * values[k];
          }

          dest(I[dims]...) = rhs(I[dims]...) - result;
        });
  });

  // dest.q.wait();

  return 0;
}

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all>
int Subtract_Convolve(Domain<Dim, strides_all...> &dest,
                      Domain<Dim, strides_all...> &src,
                      Domain<Dim, strides_all...> &rhs,
                      const std::array<DataType, size> &values,
                      const std::array<Offsets, size> &offsets) {
  return Subtract_Convolve(dest, src, rhs, values, offsets,
                           std::make_index_sequence<Dim>());
}

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all, int Dim_2, std::size_t... dims>
DataType PBE_Convolve_kernel(const Domain<Dim, strides_all...> &src,
                             const Domain<Dim, strides_all...> &kappa_map,
                             const Domain<Dim, strides_all...> &epsilon_map,
                             const DataType &kappa_2, const DataType grid_step,
                             const DataType epsilon_r,
                             const DataType delta_epsilon,
                             const std::array<DataType, size> &values,
                             const std::array<Offsets, size> &offsets,
                             sycl::id<Dim_2> I, std::index_sequence<dims...>) {

  static_assert(
      Dim_2 == Dim,
      "The dimension of the index does not match the index of the domain");

  DataType result = 0;

  // Performing the gradient convolution
  for (int k = 0; k < size; k++) {
    result += src((I[dims] + offsets[k][dims])...) * values[k];
  }

  // result *= epsilon_r + (delta_epsilon * epsilon_map(I[dims]...));
  //
  //  // Adding the gradient part of the position dependence of epsilon
  //  for (int k = 0; k < Dim; k++) {
  //    sycl::id<Dim> I2{I}, I3{I};
  //    I2[k] += 1;
  //    I3[k] -= 1;
  //    result += (src(I2[dims]...) - src(I3[dims]...)) * (delta_epsilon) *
  //              (epsilon_map(I2[dims]...) - epsilon_map(I3[dims]...)) /
  //              (4 * grid_step);
  //  }

  // result += kappa_map(I[dims]...) * epsilon_r * kappa_2 * src(I[dims]...);
  return result;
}

template <typename DataType, typename Offsets, Dimension Dim, std::size_t size,
          Length... strides_all, int Dim2>
DataType PBE_Convolve_kernel(const Domain<Dim, strides_all...> &src,
                             const Domain<Dim, strides_all...> &kappa_map,
                             const Domain<Dim, strides_all...> &epsilon_map,
                             const DataType &kappa_2, const DataType grid_step,
                             const DataType epsilon_r,
                             const DataType delta_epsilon,
                             const std::array<DataType, size> &values,
                             const std::array<Offsets, size> &offsets,
                             sycl::id<Dim2> I) {

  return PBE_Convolve_kernel(src, kappa_map, epsilon_map, kappa_2, grid_step,
                             epsilon_r, delta_epsilon, values, offsets, I,
                             std::make_index_sequence<Dim>{});

  // return 0;
}

// dest.q.wait();

template <typename DataType, typename Offsets, std::size_t size, Dimension Dim,
          Length... strides_all, std::size_t... dims>
int PBE_Convolve(Domain<Dim, strides_all...> &dest,
                 Domain<Dim, strides_all...> &src,
                 Domain<Dim, strides_all...> &kappa_map,
                 Domain<Dim, strides_all...> &epsilon_map,
                 const DataType &kappa_2, const DataType grid_step,
                 const DataType epsilon_r, const DataType delta_epsilon,
                 std::array<DataType, size> values,
                 std::array<Offsets, size> offsets,
                 std::index_sequence<dims...>) {
  assert(dest.q == src.q);
  assert(dest.padding_width == src.padding_width);

  dest.q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<Dim>(dest.strides[dims]...),
                   [=](sycl::id<Dim> I) {
                     ((I[dims] += src.padding_width), ...);
                     DataType result = PBE_Convolve_kernel(
                         src, kappa_map, epsilon_map, kappa_2, grid_step,
                         epsilon_r, delta_epsilon, values, offsets, I);

                     dest(I[dims]...) = result;
                   });
  });

  // dest.q.wait();

  return 0;
}

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all, std::size_t... dims>
int PBE_Convolve(Domain<Dim, strides_all...> &dest,
                 Domain<Dim, strides_all...> &src,
                 Domain<Dim, strides_all...> &kappa_map,
                 Domain<Dim, strides_all...> &epsilon_map,
                 const DataType &kappa_2, const DataType grid_step,
                 const DataType epsilon_r, const DataType delta_epsilon,
                 const std::array<DataType, size> &values,
                 const std::array<Offsets, size> &offsets) {

  PBE_Convolve(dest, src, kappa_map, epsilon_map, kappa_2, grid_step, epsilon_r,
               delta_epsilon, values, offsets, std::make_index_sequence<Dim>{});
  return 0;
}

constexpr std::array<OffsetType, 5> vec_offsets = {
    {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
constexpr std::array<DataType, 5> vec_val{4, -1, -1, -1, -1};

} // End namespace convolution

#endif
