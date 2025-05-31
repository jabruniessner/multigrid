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

template <int direction, typename DataType, Dimension Dim,
          Length... strides_all, int Dim_2, std::size_t... dims>
inline DataType
directional_derivative(const Domain<Dim, strides_all...> &src,
                       const Domain<Dim, strides_all...> &epsilon_map,
                       const DataType grid_step, const DataType epsilon_r,
                       const DataType delta_epsilon, sycl::id<Dim_2> I,
                       std::index_sequence<dims...>) {
  sycl::id<Dim_2> I2{I}, I3{I};
  I2[direction] += 1;
  I3[direction] -= 1;

  // Evaluating epsilon at the desired points
  // const DataType epsilon_here =
  //     (epsilon_r + epsilon_map(I[dims]...) * delta_epsilon);
  // const DataType epsilon_after =
  //     (epsilon_r + epsilon_map(I2[dims]...) * delta_epsilon);
  // const DataType epsilon_before =
  //     (epsilon_r + epsilon_map(I3[dims]...) * delta_epsilon);

  // const DataType epsilon_upper =
  //     2 * epsilon_here * epsilon_after / (epsilon_here + epsilon_after);
  // const DataType epsilon_lower =
  //     2 * epsilon_here * epsilon_before / (epsilon_here + epsilon_before);

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
                       const DataType delta_epsilon, sycl::id<Dim_2> I) {
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
    const DataType delta_epsilon, sycl::id<Dim_2> I,
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
    const DataType delta_epsilon, sycl::id<Dim2> I) {

  return PBE_Convolve_kernel(src, kappa_map, epsilon_maps, kappa_2, grid_step,
                             epsilon_r, delta_epsilon, I,
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
  assert(dest.q == src.q);
  assert(dest.padding_width == src.padding_width);

  dest.q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::range<Dim>(dest.strides[dims]...), [=](sycl::id<Dim> I) {
          ((I[dims] += src.padding_width), ...);
          DataType result =
              PBE_Convolve_kernel(src, kappa_map, epsilon_maps, kappa_2,
                                  grid_step, epsilon_r, delta_epsilon, I);

          dest(I[dims]...) = result;
        });
  });

  // dest.q.wait();

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

constexpr std::array<OffsetType, 5> vec_offsets = {
    {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
constexpr std::array<DataType, 5> vec_val{4, -1, -1, -1, -1};

} // End namespace convolution

#endif
