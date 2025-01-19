#include "concepts.h"
#include "matrix_utilities.h"
#include "predefinitions.h"
#include "utils.h"
#include <array>
#include <cmath>
#include <format>
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>
#include <tuple>
#include <utility>

#ifndef FULLILU_H
#define FULLILU_H

template <std::size_t stride> int get_index(std::size_t i, std::size_t j) {
  return i * stride + j;
}

template <std::size_t problem_size, typename DataType>
void print_matrix(DataType *matrix, sycl::queue &q) {
  std::cout << "Matrix: " << std::endl;
  for (int i = 0; i < problem_size; i++) {
    for (int j = 0; j < problem_size; j++) {
      DataType value;
      q.memcpy(&value, matrix + get_index<problem_size>(i, j), sizeof(DataType))
          .wait();
      std::cout << std::format("{:6.3f} ", value);
    }
    std::cout << std::endl;
  }
}

template <std::size_t problem_size, typename DataType,
          subscriptable<std::size_t> Vector_Type>
void matrix_vector_multiply(DataType *matrix, Vector_Type vector,
                            DataType *result) {
  for (int i = 0; i < problem_size; i++) {
    result[i] = 0;
    for (int j = 0; j < problem_size; j++) {
      result[i] += matrix[get_index<problem_size>(i, j)] * vector[j];
    }
  }
}

template <std::size_t problem_size, typename DataType,
          subscriptable<std::size_t> Vector_Type>
void vector_norm(Vector_Type vector, DataType *result) {
  result = 0;
  for (int i = 0; i < problem_size; i++) {
    result += vector[i] * vector[i];
  }
  result = std::sqrt(result);
}

template <std::size_t problem_size, typename DataType,
          subscriptable<std::size_t> Vector_Type>
void vector_subtract(Vector_Type vector1, Vector_Type vector2,
                     DataType *result) {
  for (int i = 0; i < problem_size; i++) {
    result[i] = vector1[i] - vector2[i];
  }
}

template <std::size_t problem_size, typename DataType>
void Factorize_ILU(DataType *matrix) {
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

template <std::size_t problem_size, typename DataType,
          subscriptable<std::size_t> Vector_Type>
void solve_ILU(DataType *matrix, Vector_Type &vector) {

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

template <std::size_t Dim, std::size_t problem_size, typename DataType,
          typename OffsetType, std::size_t stencil_length,
          std::size_t... Strides, std::size_t... dims>
void create_matrix_from_stencil(
    DataType *matrix, const std::array<double, stencil_length> &values,
    const std::array<OffsetType, stencil_length> &offsets, sycl::queue &q,
    std::index_sequence<Strides...>, std::index_sequence<dims...>) {

  static_assert(Dim == sizeof...(Strides));
  q.parallel_for(sycl::range<Dim>(Strides...), [=](sycl::id<Dim> I) {
     auto index_tuple = std::make_tuple(I[dims]...);
     auto row_index = std::apply(
         [&](auto... elems) { return flatten_index<Strides...>(0, elems...); },
         index_tuple); // computing the row_index

     // utils::print_tuple(index_tuple);
     for (int j = 0; j < stencil_length; j++) {

       auto index_tuple_new = index_tuple;
       auto flat_index_row = std::apply(
           [&](auto... elems) {
             return flatten_index<Strides...>(0, elems...);
           },
           index_tuple_new);
       utils::add_to_tuple(index_tuple_new, offsets[j]);
       auto flat_index_column = std::apply(
           [&](auto... elems) {
             return flatten_index<Strides...>(0, elems...);
           },
           index_tuple_new);

       bool in_range = std::apply(
           [&](auto... elems) {
             bool truth = true;
             static_assert(sizeof...(elems) == sizeof...(Strides));
             ((truth = truth && elems >= 0 && elems < Strides), ...);
             return truth;
           },
           index_tuple_new); // Computing whether the index is in the range of
                             // the Domain;

       if (!in_range)
         continue; // Do not add value if the domain is out of range

       matrix[get_index<problem_size>(row_index, flat_index_column)] =
           values[j];
     }
   }).wait();
};

template <std::size_t Dim, std::size_t problem_size, typename DataType,
          typename OffsetType, std::size_t stencil_length,
          std::size_t... Strides>
void create_matrix_from_stencil(
    DataType *matrix, const std::array<DataType, stencil_length> &values,
    const std::array<OffsetType, stencil_length> &offsets, sycl::queue &q,
    std::index_sequence<Strides...>) {

  create_matrix_from_stencil<Dim, problem_size>(
      matrix, values, offsets, q, std::index_sequence<Strides...>{},
      std::make_index_sequence<Dim>{});
}

#endif //! FULLILU_H
