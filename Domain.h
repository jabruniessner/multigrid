#include "cuda_reduce.hpp"
#include "hipSYCL/pcuda/cuda_runtime.h"
#include "predefinitions.h"
#include "thrust_adjust.hpp"
#include "utils.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <execution>
#include <format>
#include <iostream>
#include <numeric>
#include <ostream>
#include <tuple>
#include <utility>

#ifndef DOMAIN_H
#define DOMAIN_H

namespace domain {

template <Length FirstStride, Length... RestStrides, typename Padding,
          typename Position, typename... PositionRest>
size_t flatten_index(Padding padding, Position i,
                     PositionRest... rest_positions) {
  // initialize i with the first index value
  int index = i;

  // Use a fold expression to process all Strides and indices
  ((index *= RestStrides + 2 * padding, index += rest_positions), ...);

  return index;
}

template <Length FirstStride, Length... RestStrides, std::size_t... directions>
std::array<Length, sizeof...(RestStrides) + 1>
flat_to_multi_index(Length i, std::index_sequence<directions...>) {
  static_assert(sizeof...(RestStrides) == sizeof...(directions));
  constexpr std::array<Length, sizeof...(RestStrides)> strides{RestStrides...};
  constexpr std::size_t Dim = sizeof...(RestStrides) + 1;

  std::array<Length, sizeof...(RestStrides) + 1> multi_index;

  // for (int i : strides)
  //   std::cout << "Strides: " << i << std::endl;

  ((multi_index[Dim - 1 - directions] = i % strides[Dim - 2 - directions],
    // std::cout << "strides: " << strides[Dim - 2 - directions] << std::endl,
    // std::cout << "position: " << Dim - 1 - directions << std::endl,
    // std::cout << "value: " << i % strides[Dim - 2 - directions] << std::endl,
    i /= strides[Dim - 2 - directions]),
   ...);
  multi_index[0] = i;

  return multi_index;
}

template <Length FirstStride, Length... RestStrides>
std::array<Length, sizeof...(RestStrides) + 1> flat_to_multi_index(Length i) {
  return flat_to_multi_index<FirstStride, RestStrides...>(
      i, std::make_index_sequence<sizeof...(RestStrides)>{});
}

template <typename DataType, Dimension Dim, Length... strides_all> struct Grid {
  template <typename... Length>
  Grid(Paddings padding, int padding_width)
      : strides{strides_all...}, padding(padding), padding_width(padding_width)

  {
    static_assert(sizeof...(strides_all) == Dim);

    num_values = 1;
    ((num_values *= strides_all + 2 * padding_width), ...);

    num_dofs = 1;
    ((num_dofs *= strides_all), ...);

    cudaMallocManaged(&values_buff, sizeof(DataType[num_values]));
    // q.memset(values_buff, 0, num_values * sizeof(DataType)).wait();
  }

  template <typename... Positions>
  DataType &operator()(const Positions &...positions) const {
    static_assert(sizeof...(Positions) == Dim);
    return values_buff[flatten_index<strides_all...>(padding_width,
                                                     positions...)];
  }

  DataType *values_buff;
  Length strides[Dim];
  Length num_values;
  Length num_dofs;
  Length padding_width;
  Paddings padding;

  static constexpr std::array<Length, Dim> length{strides_all...};
};

template <Dimension Dim, Length... strides_all>
struct Domain : Grid<DataType, Dim, strides_all...> {
  Domain(Paddings padding, int padding_width)
      : Grid<DataType, Dim, strides_all...>(padding, padding_width) {

    thrust::fill(this->values_buff, this->values_buff + this->num_values, 0);
    cudaDeviceSynchronize();
  }

  void print_dx_to_stream(std::ostream &out, DataType xmin, DataType ymin,
                          DataType zmin, DataType Box_length) const {

#define format_v(X) std::format("{:<+13e} ", X)

    const DataType delta =
        Box_length / (this->strides[0] + this->padding_width);
    out << "object 1 class gridpositions counts" << " "
        << this->strides[0] + 2 * this->padding_width << " "
        << this->strides[1] + 2 * this->padding_width << " "
        << this->strides[2] + 2 * this->padding_width << std::endl;
    out << "origin " << format_v(xmin) << format_v(ymin) << format_v(zmin)
        << std::endl;
    out << "delta " << format_v(delta) << format_v(0.0) << format_v(0.0)
        << std::endl;
    out << "delta " << format_v(0.0) << format_v(delta) << format_v(0.0)
        << std::endl;
    out << "delta " << format_v(0.0) << format_v(0.0) << format_v(delta)
        << std::endl;

    out << "object 2 class gridconnections count "
        << this->strides[0] + 2 * this->padding_width << " "
        << this->strides[1] + 2 * this->padding_width << " "
        << this->strides[2] + 2 * this->padding_width << std::endl;

    out << "object 3 class array type double rank 0 items " << this->num_values
        << " data follows" << std::endl;

    cudaDeviceSynchronize();

    for (int i = 0; i < this->num_values; i++) {

      if (i % 3 == 0 && i != 0) {
        out << std::endl;
      }
      out << format_v(this->values_buff[i]);
    }
    out << std::endl;

    out << "attribute \"dep\" string \"positions\"" << std::endl;
    out << "object \"regular positions regular connections\" class field"
        << std::endl;
    out << "component \"positions\" value 1" << std::endl;
    out << "component \"connections\" value 2" << std::endl;
    out << "component \"data\" value 3" << std::endl;
  }

  template <typename... Indices> void print_domain(Indices... indices) {

    if constexpr (sizeof...(Indices) == 0)
      cudaDeviceSynchronize();

    if constexpr (sizeof...(Indices) < Dim) {
      for (Position1D i = 0;
           i < this->strides[sizeof...(Indices)] + 2 * this->padding_width; i++)
        print_domain(indices..., i);
      std::cout << std::endl;
    } else {
      std::cout << std::format(
          "{:6.3f} ",
          this->Grid<DataType, Dim, strides_all...>::operator()(indices...));
    }
  };

  template <typename... Indices>
  void print_domain_to_stream(std::ostream &output, Indices... indices) {

    if constexpr (sizeof...(Indices) == 0)
      cudaDeviceSynchronize();

    if constexpr (sizeof...(Indices) < Dim) {
      constexpr auto size = sizeof...(Indices);
      constexpr auto dimension_size =
          utils::get_stack_element<size>(strides_all...);
      for (int i = 0; i < dimension_size + 2 * this->padding_width; i++) {
        print_domain_to_stream(output, indices..., i);
      }
    } else {
      ((std::cout << indices << " "), ...);
      std::cout << std::format(
                       "{:6.3f}",
                       this->Grid<DataType, Dim, strides_all...>::operator()(
                           indices...))
                << std::endl;
    }
  }

  template <std::size_t... Ints>
  void print_header(std::ostream &output, std::index_sequence<Ints...>) {
    output << "Dimension: " << Dim << std::endl;
    output << "Number of points in direction:" << std::endl;
    ((output << "Dir " << Ints << " " << strides_all + 1 << std::endl), ...);
  }

  void print_header(std::ostream &output) {
    print_header(output, std::make_index_sequence<Dim>{});
  }

  void print_to_output(std::ostream &output) {
    print_header(output);
    print_domain_to_stream(output);
  }
};

template <Dimension Dim, Length... strides_all>
int domain_compute_norm_squared(DataType &result,
                                Domain<Dim, strides_all...> &a) {

  int num_required_threads = (a.num_dofs + reduction_kernel::seq_size - 1);

  int num_blocks = (num_required_threads + reduction_kernel::gbs - 1) /
                   reduction_kernel::gbs;

  DataType *results;
  cudaMalloc(&results, sizeof(DataType) * num_blocks);
  cudaMemset(results, 0, sizeof(DataType) * num_blocks);

  reduction_kernel::cudaParallelTransformReduce(
      a.num_values, &result, results, std::plus<>(),
      [=](int i) { return a.values_buff[i] * a.values_buff[i]; });

  return 0;
}

template <Dimension Dim, Length... strides_all>
DataType domain_scalar_product(Domain<Dim, strides_all...> &a,
                               Domain<Dim, strides_all...> &b) {
  return thrust::transform_reduce(
      a.values_buff, a.values_buff + a.num_values, b.values_buff, 0.0,
      std::plus<>(),
      [](const DataType val_a, const DataType val_b) { return val_a * val_b; });
}

template <Dimension Dim, Length... strides_all>
DataType domain_find_ideal_factor(Domain<Dim, strides_all...> &a,
                                  Domain<Dim, strides_all...> &b) {
  DataType numerator = domain_scalar_product(a, b);
  DataType denominator = domain_scalar_product(b, b);

  return numerator / denominator;
}

template <Dimension Dim, Length... strides_all>
int domain_scalar_multiply(Domain<Dim, strides_all...> &dest,
                           Domain<Dim, strides_all...> &a,
                           const DataType &scalar) {

  thrust::transform(a.values_buff, a.values_buff + a.num_values,
                    dest.values_buff,
                    [=](const DataType val) { return scalar * val; });
  return 0;
}

template <Dimension Dim, Length... strides_all>
int divide_domains(Domain<Dim, strides_all...> &dest,
                   Domain<Dim, strides_all...> &a,
                   Domain<Dim, strides_all...> &b) {

  thrust::transform(
      a.values_buff, a.values_buff + a.num_values, b.values_buff,
      dest.values_buff,
      [](const DataType val_a, const DataType val_b) { return val_a / val_b; });
  return 0;
}

template <Dimension Dim, Length... strides_all>
int multiply_domains(Domain<Dim, strides_all...> &dest,
                     Domain<Dim, strides_all...> &a,
                     Domain<Dim, strides_all...> &b) {

  thrust::transform(
      a.values_buff, a.values_buff + a.num_values, b.values_buff,
      dest.values_buff,
      [](const DataType val_a, const DataType val_b) { return val_a * val_b; });

  return 0;
}

template <Dimension Dim, Length... strides_all>
int subtract_domains(Domain<Dim, strides_all...> &dest,
                     Domain<Dim, strides_all...> &a,
                     Domain<Dim, strides_all...> &b) {

  thrust::transform(
      a.values_buff, a.values_buff + a.num_values, b.values_buff,
      dest.values_buff,
      [](const DataType val_a, const DataType val_b) { return val_a - val_b; });

  return 0;
}

template <Dimension Dim, Length... strides_all>
int subtract_and_multiply_domains(Domain<Dim, strides_all...> &dest,
                                  Domain<Dim, strides_all...> &a,
                                  Domain<Dim, strides_all...> &b,
                                  const DataType &val) {

  thrust::transform(a.values_buff, a.values_buff + a.num_values, b.values_buff,
                    dest.values_buff,
                    [=](const DataType val_a, const DataType val_b) {
                      return val * val_a - val_b;
                    });

  return 0;
}

template <Dimension Dim, Length... strides_all>
int add_domains(Domain<Dim, strides_all...> &dest,
                Domain<Dim, strides_all...> &a,
                Domain<Dim, strides_all...> &b) {

  thrust::transform(
      a.values_buff, a.values_buff + a.num_values, b.values_buff,
      dest.values_buff,
      [](const DataType val_a, const DataType val_b) { return val_a + val_b; });

  return 0;
}

template <Dimension Dim, Length... strides_all>
int add_and_multiply_domains(Domain<Dim, strides_all...> &dest,
                             Domain<Dim, strides_all...> &a,
                             Domain<Dim, strides_all...> &b,
                             const DataType &val) {

  thrust::transform(a.values_buff, a.values_buff + a.num_values, b.values_buff,
                    dest.values_buff,
                    [=](const DataType val_a, const DataType val_b) {
                      return val_a + val * val_b;
                    });

  return 0;
}

template <typename Range> struct RangeProps;

template <template <std::size_t, std::size_t> typename Range, std::size_t start,
          std::size_t end>
struct RangeProps<Range<start, end>> {
  constexpr static std::size_t length = end - start;
  constexpr static std::size_t start_v = start;
  constexpr static std::size_t end_v = end;
};

} // namespace domain

#endif
