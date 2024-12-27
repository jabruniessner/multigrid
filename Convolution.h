#include "Domain.h"
#include "predefinitions.h"
#include <sycl/sycl.hpp>

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

  dest.q.wait();

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

  dest.q.wait();

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

constexpr std::array<OffsetType, 5> vec_offsets = {
    {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
constexpr std::array<DataType, 5> vec_val{4, -1, -1, -1, -1};

} // End namespace convolution

#endif
