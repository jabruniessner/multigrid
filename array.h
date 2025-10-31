#include "concepts.h"

#ifndef ARRAY_H
#define ARRAY_H

namespace array {

template <typename T> inline constexpr T square(T t) { return t * t; }

template <typename DataType, std::size_t Dim> struct vector {

  //  template <typename... T>
  //  __host__ __device__ constexpr vector(const T... t) : data{t...} {}
  //  __host__ __device__ constexpr vector(const DataType val) {
  //    for (auto &i : data) {
  //      i = val;
  //    }
  //  }
  //
  //  template <typename T>
  //  __host__ __device__ constexpr vector(const std::array<T, Dim> &convertee)
  //  {
  //    std::copy(convertee.begin(), convertee.end(), data);
  //  }
  //
  //  template <typename T>
  //  __host__ __device__ constexpr vector(const std::array<T, Dim> convertee) {
  //    std::copy(convertee.begin(), convertee.end(), data);
  //  }

  __host__ __device__ constexpr DataType &operator[](std::size_t i) {
    return data[i];
  }

  __host__ __device__ constexpr const DataType &
  operator[](std::size_t i) const {
    return data[i];
  }

  DataType data[Dim];
};

} // namespace array
#endif // !BLAS_H
