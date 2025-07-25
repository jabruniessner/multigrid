#include "concepts.h"
#include "predefinitions.h"
#include <array>

#ifndef BLAS_H
#define BLAS_H

namespace blas {

template <typename T> inline constexpr T square(T t) { return t * t; }

template <typename DataType, Dimension Dim>
struct vector : std::array<DataType, Dim> {

  template <typename... T>
  constexpr vector(const T... t) : std::array<DataType, Dim>{t...} {}

  inline constexpr vector operator+(const vector &vector2) const {
    vector new_vec;
    for (int i = 0; i < Dim; i++)
      new_vec[i] = vector2[i] + (*this)[i];

    return new_vec;
  }

  inline constexpr vector operator-(const vector &vector2) const {
    vector new_vec;
    for (int i = 0; i < Dim; i++)
      new_vec[i] = (*this)[i] - vector2[i];

    return new_vec;
  }

  inline constexpr vector operator*(const DataType &value) const {
    vector new_vec;
    for (int i = 0; i < Dim; i++) {
      new_vec[i] = (*this)[i] * value;
    }
    return new_vec;
  }

  inline constexpr DataType operator*(const vector &vector2) const {
    DataType return_value = 0;
    for (int i = 0; i < Dim; i++)
      return_value += vector2[i] * (*this)[i];

    return return_value;
  }

  inline constexpr vector operator/=(const DataType divisor) {
    for (DataType &i : *this) {
      i /= divisor;
    }
    return *this;
  }

  inline constexpr vector operator*=(const DataType factor) {
    for (DataType &i : (*this)) {
      i *= factor;
    }
    return *this;
  }
};

template <typename DataType, Dimension Dim>
inline constexpr DataType norm(const vector<DataType, Dim> vec) {
  return std::sqrt(vec * vec);
}

template <typename DataType, Dimension Dim>
inline constexpr DataType norm_squared(const vector<DataType, Dim> vec) {
  return vec * vec;
}

template <typename DataType, std::size_t Dim,
          subscriptable<std::size_t>... Vector_Types>
inline DataType determinant(const Vector_Types &...vectors) {

  vector<DataType, Dim> Matrix[] = {vectors...};
  return Matrix[0][0] *
             (Matrix[1][1] * Matrix[2][2] - Matrix[1][2] * Matrix[2][1]) -
         Matrix[0][1] *
             (Matrix[1][0] * Matrix[2][2] - Matrix[1][2] * Matrix[2][0]) +
         Matrix[0][2] *
             (Matrix[1][0] * Matrix[2][1] - Matrix[1][1] * Matrix[2][0]);
}

inline DataType determinant(DataType Matrix[3][3]) {
  return Matrix[0][0] *
             (Matrix[1][1] * Matrix[2][2] - Matrix[1][2] * Matrix[2][1]) -
         Matrix[0][1] *
             (Matrix[1][0] * Matrix[2][2] - Matrix[1][2] * Matrix[2][0]) +
         Matrix[0][2] *
             (Matrix[1][0] * Matrix[2][1] - Matrix[1][1] * Matrix[2][0]);
}

} // namespace blas

// makeing the vector tuple like
namespace std {
template <typename T, std::size_t N>
struct tuple_size<blas::vector<T, N>> : tuple_size<array<T, N>> {};

template <std::size_t I, typename T, std::size_t N>
struct tuple_element<I, blas::vector<T, N>> : tuple_element<I, array<T, N>> {};
} // namespace std

#endif // !BLAS_H
