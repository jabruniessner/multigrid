#include "matrix_utilities.h"
#include "utils.h"
#include <array>
#include <cmath>
#include <format>
#include <iostream>
#include <string>
#include <tuple>

template <std::size_t stride> int get_index(std::size_t i, std::size_t j) {
  return i * stride + j;
}

template <std::size_t problem_size, typename DataType>
void print_matrix(std::array<DataType, problem_size * problem_size> &matrix) {
  std::cout << "Matrix: " << std::endl;
  for (int i = 0; i < problem_size; i++) {
    for (int j = 0; j < problem_size; j++) {
      std::cout << std::format("{:6.3f} ",
                               matrix[get_index<problem_size>(i, j)]);
    }
    std::cout << std::endl;
  }
}

template <std::size_t problem_size, typename DataType>
void matrix_vector_multiply(
    std::array<DataType, problem_size * problem_size> &matrix,
    std::array<DataType, problem_size> &vector,
    std::array<DataType, problem_size> &result) {
  for (int i = 0; i < problem_size; i++) {
    result[i] = 0;
    for (int j = 0; j < problem_size; j++) {
      result[i] += matrix[get_index<problem_size>(i, j)] * vector[j];
    }
  }
}

template <std::size_t problem_size, typename DataType>
void vector_norm(std::array<DataType, problem_size> &vector, DataType &result) {
  result = 0;
  for (int i = 0; i < problem_size; i++) {
    result += vector[i] * vector[i];
  }
  result = std::sqrt(result);
}

template <std::size_t problem_size, typename DataType>
void vector_subtract(std::array<DataType, problem_size> &vector1,
                     std::array<DataType, problem_size> &vector2,
                     std::array<DataType, problem_size> &result) {
  for (int i = 0; i < problem_size; i++) {
    result[i] = vector1[i] - vector2[i];
  }
}

template <std::size_t problem_size, typename DataType>
void Factorize_ILU(std::array<DataType, problem_size * problem_size> &matrix) {
  for (int k = 0; k < problem_size - 1; k++) {
    for (int i = k + 1; i < problem_size; i++) {
      auto &a = matrix[get_index<problem_size>(i, k)];
      a /= matrix[get_index<problem_size>(k, k)];
      for (int j = k + 1; j < problem_size; j++) {
        auto &a_ij = matrix[get_index<problem_size>(i, j)];
        a_ij -= matrix[get_index<problem_size>(i, k)] *
                matrix[get_index<problem_size>(k, j)];
      }
    }
  }
}

template <std::size_t problem_size, typename DataType>
void solve_ILU(std::array<DataType, problem_size * problem_size> &matrix,
               std::array<DataType, problem_size> &vector) {

  // Solveing the lower triangular matrix
  for (int i = 1; i < problem_size; i++) {
    for (int j = 0; j < i; j++) {
      vector[i] -= matrix[get_index<problem_size>(i, j)] * vector[j];
    }
  }

  // solveing the upper triangular matrix
  for (int i = problem_size - 1; i >= 0; i--) {
    for (int j = i + 1; j < problem_size; j++) {
      vector[i] -= matrix[get_index<problem_size>(i, j)] * vector[j];
    }
    vector[i] /= matrix[get_index<problem_size>(i, i)];
  }
};

template <std::size_t problem_size, typename DataType, typename OffsetType,
          std::size_t stencil_length, std::size_t FirstStride,
          std::size_t... Strides, std::size_t... All_Strides,
          typename... Indices>
void create_matrix_from_stencil(
    std::array<DataType, problem_size * problem_size> &matrix,
    std::array<double, stencil_length> &values,
    std::array<OffsetType, stencil_length> &offsets,
    std::index_sequence<FirstStride, Strides...>,
    std::index_sequence<All_Strides...>, Indices... indices) {

  static_assert(sizeof...(Indices) + sizeof...(Strides) + 1 ==
                sizeof...(All_Strides));

  if constexpr (sizeof...(Strides) == 0) {

    for (int i = 0; i < FirstStride; i++) {
      auto index_tuple = std::make_tuple(indices..., i);
      auto row_index = std::apply(
          [&](auto... elems) {
            return flatten_index<All_Strides...>(0, elems...);
          },
          index_tuple); // computing the row_index

      // utils::print_tuple(index_tuple);
      for (int j = 0; j < stencil_length; j++) {

        auto index_tuple_new = index_tuple;
        auto flat_index_row = std::apply(
            [&](auto... elems) {
              return flatten_index<All_Strides...>(0, elems...);
            },
            index_tuple_new);
        utils::add_to_tuple(index_tuple_new, offsets[j]);
        auto flat_index_column = std::apply(
            [&](auto... elems) {
              return flatten_index<All_Strides...>(0, elems...);
            },
            index_tuple_new);

        bool in_range = std::apply(
            [&](auto... elems) {
              bool truth = true;
              static_assert(sizeof...(elems) == sizeof...(All_Strides));
              ((truth = truth && elems >= 0 && elems < All_Strides), ...);
              return truth;
            },
            index_tuple_new); // Computing whether the index is in the range of
                              // the Domain;

        if (!in_range)
          continue; // Do not add value if the domain is out of range

        matrix[get_index<problem_size>(row_index, flat_index_column)] =
            values[j];
      }
    }

  } else {
    for (int i = 0; i < FirstStride; i++) {
      create_matrix_from_stencil<problem_size>(
          matrix, values, offsets, std::index_sequence<Strides...>{},
          std::index_sequence<All_Strides...>{}, indices..., i);
    }
  }
}

template <std::size_t problem_size, typename DataType, typename OffsetType,
          std::size_t stencil_length, std::size_t... Strides>
void create_matrix_from_stencil(
    std::array<DataType, problem_size * problem_size> &matrix,
    std::array<DataType, stencil_length> &values,
    std::array<OffsetType, stencil_length> &offsets,
    std::index_sequence<Strides...>) {

  // std::size_t current_position = 0;

  create_matrix_from_stencil<problem_size>(matrix, values, offsets,
                                           std::index_sequence<Strides...>{},
                                           std::index_sequence<Strides...>{});
};
