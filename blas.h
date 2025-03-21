#include "predefinitions.h"
#include <array>

#ifndef BLAS_H
#define BLAS_H

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
};

template <typename DataType, Dimension Dim>
inline constexpr DataType norm(const vector<DataType, Dim> vec) {
  return std::sqrt(vec * vec);
}

template <typename DataType, Dimension Dim>
inline constexpr DataType norm_squared(const vector<DataType, Dim> vec) {
  return vec * vec;
}

#endif // !BLAS_H
