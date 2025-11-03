#include "concepts.h"
#include <iostream>
#include <utility>

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

namespace detail {

template <std::size_t I, typename DataType, std::size_t Dim>
constexpr __host__ __device__ DataType &
get(::array::vector<DataType, Dim> &vec) noexcept {
  static_assert(I < Dim, "The index is larger then the Dimension");
  return vec[I];
}

template <typename F, typename DataType, std::size_t Dim, std::size_t... dims>
constexpr __host__ __device__ decltype(auto)
apply_impl(F f, ::array::vector<DataType, Dim> vec,
           std::index_sequence<dims...>) {
  return f(get<dims>(vec)...);
};
} // namespace detail

} // namespace array

namespace std {
template <typename DataType, std::size_t Dim>
struct tuple_size<::array::vector<DataType, Dim>>
    : std::integral_constant<std::size_t, Dim> {};

template <std::size_t I, typename DataType, std::size_t Dim>
struct tuple_element<I, ::array::vector<DataType, Dim>> {
  using type = DataType;
};

template <std::size_t I, typename DataType, std::size_t Dim>
constexpr __host__ __device__ DataType &
get(::array::vector<DataType, Dim> &vec) noexcept {
  static_assert(I < Dim, "The index is larger then the Dimension");
  return vec[I];
}

template <std::size_t I, typename DataType, std::size_t Dim>
constexpr __host__ __device__ DataType &
get(::array::vector<DataType, Dim> &&vec) noexcept {
  static_assert(I < Dim, "The index is larger then the Dimension");
  return vec[I];
}

template <std::size_t I, typename T, std::size_t N>
constexpr const T &get(const ::array::vector<T, N> &v) noexcept {
  static_assert(I < N);
  return v[I];
}

template <std::size_t I, typename T, std::size_t N>
constexpr T &&get(::array::vector<T, N> &&v) noexcept {
  static_assert(I < N);
  return std::move(v[I]);
}

template <std::size_t I, typename T, std::size_t N>
constexpr const T &&get(const ::array::vector<T, N> &&v) noexcept {
  static_assert(I < N);
  return std::move(v[I]);
}

template <typename F, typename DataType, std::size_t Dim>
constexpr __host__ __device__ decltype(auto)
apply(F f, ::array::vector<DataType, Dim> tuple) {
  return ::array::detail::apply_impl(f, tuple, std::make_index_sequence<Dim>{});
}

} // namespace std

#endif // !BLAS_H
