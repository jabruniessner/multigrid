#include "Domain.h"
#include "predefinitions.h"
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#ifdef __CUDACC__
#include <thrust/iterator/counting_iterator.h>
#else
#include <boost/iterator/counting_iterator.hpp>
#endif // __CUDACC__

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

namespace convolution {

#ifdef __CUDACC__
using namespace thrust;
#else
using namespace boost;
#endif

using namespace domain;

template <typename Domain_type, typename Offsets, std::size_t size,
          std::size_t... dims>
struct Convolution_functor {
  __device__ __host__ void operator()(int i) const {
    constexpr auto strides = Domain_type::length;
    auto I = domain::flat_to_multi_index<strides[dims]...>(i);
    ((I[dims] += dest.padding_width), ...);
    DataType result = 0;

    for (int k = 0; k < size; k++) {
      result += src((I[dims] + offsets[k][dims])...) * values[k];
    }

    dest(I[dims]...) = result;
  }

  const domain::array<Offsets, size> offsets;
  const domain::array<DataType, size> values;
  const Domain_type dest;
  const Domain_type src;
};

template <typename T> struct TD;

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all, std::size_t... dims>

int Convolve(Domain<Dim, strides_all...> &dest,
             Domain<Dim, strides_all...> &src,
             const domain::array<DataType, size> values,
             const domain::array<Offsets, size> offsets,
             std::index_sequence<dims...>) {

  counting_iterator<int> start(0);
  counting_iterator<int> end(dest.num_dofs);

  using d_type = std::remove_reference_t<decltype(dest)>;

  Convolution_functor<d_type, Offsets, size, dims...> f{offsets, values, dest,
                                                        src};

  using UnaryFunction = decltype(f);
  using RandAccIt = decltype(start);

  thrust::for_each<RandAccIt, UnaryFunction, thrust::gbs>(start, end, f);

  return 0;
}

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all>
int Convolve(Domain<Dim, strides_all...> &dest,
             Domain<Dim, strides_all...> &src,
             const domain::array<DataType, size> values,
             const domain::array<Offsets, size> offsets) {
  return Convolve<DataType, Offsets, size, Dim, strides_all...>(
      dest, src, values, offsets, std::make_index_sequence<Dim>());

  // return 0;
}

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all, std::size_t... dims>

int Subtract_Convolve(Domain<Dim, strides_all...> &dest,
                      Domain<Dim, strides_all...> &src,
                      Domain<Dim, strides_all...> &rhs,
                      const domain::array<DataType, size> values,
                      const domain::array<Offsets, size> offsets,
                      std::index_sequence<dims...>) {

  counting_iterator<int> start(0);
  counting_iterator<int> end(dest.num_dofs);

  thrust::for_each(start, end, [=](int i) {
    auto I = domain::flat_to_multi_index<strides_all...>(i);
    ((I[dims] += dest.padding_width), ...);

    DataType result = 0;

    for (int k = 0; k < size; k++) {
      result += src((I[dims] + offsets[k][dims])...) * values[k];
    }

    dest(I[dims]...) = rhs(I[dims]...) - result;
  });

  return 0;
}

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all>
int Subtract_Convolve(Domain<Dim, strides_all...> &dest,
                      Domain<Dim, strides_all...> &src,
                      Domain<Dim, strides_all...> &rhs,
                      const domain::array<DataType, size> values,
                      const domain::array<Offsets, size> offsets) {
  return Subtract_Convolve(dest, src, rhs, values, offsets,
                           std::make_index_sequence<Dim>());
}

template <int direction, typename DataType, Dimension Dim,
          Length... strides_all, int Dim_2, std::size_t... dims>
inline DataType directional_derivative(
    const Domain<Dim, strides_all...> &src,
    const Domain<Dim, strides_all...> &epsilon_map, const DataType grid_step,
    const DataType epsilon_r, const DataType delta_epsilon,
    std::array<std::size_t, Dim_2> I, std::index_sequence<dims...>) {

  std::array<std::size_t, Dim_2> I2{I}, I3{I};
  I2[direction] += 1;
  I3[direction] -= 1;

  const DataType epsilon_lower =
      (epsilon_r + epsilon_map(I3[dims]...) * delta_epsilon);
  const DataType epsilon_upper =
      (epsilon_r + epsilon_map(I[dims]...) * delta_epsilon);
  const DataType result =
      (epsilon_upper * (src(I2[dims]...) - src(I[dims]...)) -
       epsilon_lower * (src(I[dims]...) - src(I3[dims]...))) /
      (grid_step * grid_step);

  return result;
}

template <int direction, typename DataType, Dimension Dim,
          Length... strides_all, int Dim_2>
inline DataType
directional_derivative(const Domain<Dim, strides_all...> &src,
                       const Domain<Dim, strides_all...> &epsilon_map,
                       const DataType grid_step, const DataType epsilon_r,
                       const DataType delta_epsilon,
                       std::array<std::size_t, Dim_2> I) {
  return directional_derivative<direction>(src, epsilon_map, grid_step,
                                           epsilon_r, delta_epsilon, I,
                                           std::make_index_sequence<Dim>{});
}

template <typename DataType, Dimension Dim, Length... strides_all, int Dim_2,
          std::size_t... dims>
DataType PBE_Convolve_kernel(
    const Domain<Dim, strides_all...> &src,
    const Domain<Dim, strides_all...> &kappa_map,
    const std::array<Domain<Dim, strides_all...>, Dim> &epsilon_maps,
    const DataType &kappa_2, const DataType grid_step, const DataType epsilon_r,
    const DataType delta_epsilon, std::array<std::size_t, Dim_2> I,
    std::index_sequence<dims...>) {

  static_assert(
      Dim_2 == Dim,
      "The dimension of the index does not match the index of the domain");

  DataType result = 0;

  ((result += directional_derivative<dims>(src, epsilon_maps[dims], grid_step,
                                           epsilon_r, delta_epsilon, I)),
   ...);

  result -= kappa_map(I[dims]...) * epsilon_r * kappa_2 * src(I[dims]...);
  return result;
}

template <typename DataType, Dimension Dim, Length... strides_all, int Dim2>
DataType PBE_Convolve_kernel(
    const Domain<Dim, strides_all...> &src,
    const Domain<Dim, strides_all...> &kappa_map,
    const std::array<Domain<Dim, strides_all...>, Dim> &epsilon_maps,
    const DataType &kappa_2, const DataType grid_step, const DataType epsilon_r,
    const DataType delta_epsilon, std::array<std::size_t, Dim2> I) {

  return PBE_Convolve_kernel(src, kappa_map, epsilon_maps, kappa_2, grid_step,
                             epsilon_r, delta_epsilon, I,
                             std::make_index_sequence<Dim>{});

  // return 0;
}

template <int direction, typename DataType, Dimension Dim,
          Length... strides_all, int Dim_2, std::size_t... dims>
inline DataType
directional_GS(const Domain<Dim, strides_all...> &src,
               const Domain<Dim, strides_all...> &epsilon_map,
               const DataType epsilon_r, const DataType delta_epsilon,
               std::array<std::size_t, Dim_2> I, std::index_sequence<dims...>) {

  std::array<std::size_t, Dim_2> I2{I}, I3{I};
  I2[direction] += 1;
  I3[direction] -= 1;

  const DataType epsilon_lower =
      (epsilon_r + epsilon_map(I3[dims]...) * delta_epsilon);
  const DataType epsilon_upper =
      (epsilon_r + epsilon_map(I[dims]...) * delta_epsilon);
  const DataType result =
      (epsilon_upper * (src(I2[dims]...)) + epsilon_lower * (src(I3[dims]...)));

  return result;
}

template <int direction, typename DataType, Dimension Dim,
          Length... strides_all, int Dim_2>
inline DataType directional_GS(const Domain<Dim, strides_all...> &src,
                               const Domain<Dim, strides_all...> &epsilon_map,
                               const DataType epsilon_r,
                               const DataType delta_epsilon,
                               std::array<std::size_t, Dim_2> I) {
  return directional_GS<direction>(src, epsilon_map, epsilon_r, delta_epsilon,
                                   I, std::make_index_sequence<Dim>{});
}

template <typename DataType, Dimension Dim, Length... strides_all, int Dim_2,
          std::size_t... dims>
DataType

PBE_GS_kernel(const Domain<Dim, strides_all...> &src,
              const std::array<Domain<Dim, strides_all...>, Dim> &epsilon_maps,
              const DataType epsilon_r, const DataType delta_epsilon,
              std::array<std::size_t, Dim_2> I, std::index_sequence<dims...>) {

  static_assert(
      Dim_2 == Dim,
      "The dimension of the index does not match the index of the domain");

  DataType result = 0;

  ((result +=
    directional_GS<dims>(src, epsilon_maps[dims], epsilon_r, delta_epsilon, I)),
   ...);
  return result;
}

template <typename DataType, Dimension Dim, Length... strides_all, int Dim2>
DataType
PBE_GS_kernel(const Domain<Dim, strides_all...> &src,
              const std::array<Domain<Dim, strides_all...>, Dim> &epsilon_maps,
              const DataType epsilon_r, const DataType delta_epsilon,
              std::array<std::size_t, Dim2> I) {

  return PBE_GS_kernel(src, epsilon_maps, epsilon_r, delta_epsilon, I,
                       std::make_index_sequence<Dim>{});

  // return 0;
}

// dest.q.wait();

template <typename DataType, Dimension Dim, Length... strides_all,
          std::size_t... dims>
int PBE_Convolve(Domain<Dim, strides_all...> &dest,
                 Domain<Dim, strides_all...> &src,
                 Domain<Dim, strides_all...> &kappa_map,
                 std::array<Domain<Dim, strides_all...>, Dim> &epsilon_maps,
                 const DataType &kappa_2, const DataType grid_step,
                 const DataType epsilon_r, const DataType delta_epsilon,
                 std::index_sequence<dims...>) {

  counting_iterator<int> start(0);
  counting_iterator<int> end(dest.num_dofs);
  thrust::for_each(start, end, [=](int i) {
    auto I = domain::flat_to_multi_index<strides_all...>(i);
    ((I[dims] += dest.padding_width), ...);
    DataType result =
        PBE_Convolve_kernel(src, kappa_map, epsilon_maps, kappa_2, grid_step,
                            epsilon_r, delta_epsilon, I);

    dest(I[dims]...) = result;
  });

  return 0;
}

template <typename DataType, Dimension Dim, Length... strides_all,
          std::size_t... dims>
int PBE_Convolve(Domain<Dim, strides_all...> &dest,
                 Domain<Dim, strides_all...> &src,
                 Domain<Dim, strides_all...> &kappa_map,
                 std::array<Domain<Dim, strides_all...>, Dim> &epsilon_maps,
                 const DataType &kappa_2, const DataType grid_step,
                 const DataType epsilon_r, const DataType delta_epsilon) {

  PBE_Convolve(dest, src, kappa_map, epsilon_maps, kappa_2, grid_step,
               epsilon_r, delta_epsilon, std::make_index_sequence<Dim>{});
  return 0;
}

constexpr domain::array<OffsetType, 5> vec_offsets{
    {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
constexpr domain::array<DataType, 5> vec_val{4., -1., -1., -1., -1.};

} // End namespace convolution

#endif
