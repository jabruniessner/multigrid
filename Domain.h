#include "predefinitions.h"
#include "utils.h"
#include <array>
#include <cassert>
#include <cstddef>
#include <format>
#include <iostream>
#include <ostream>
#include <sycl/sycl.hpp>
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
  Grid(Paddings padding, sycl::queue &q, int padding_width)
      : strides{strides_all...}, padding(padding), padding_width(padding_width),
        q(q)

  {
    static_assert(sizeof...(strides_all) == Dim);

    num_values = 1;
    ((num_values *= strides_all + 2 * padding_width), ...);

    num_dofs = 1;
    ((num_dofs *= strides_all), ...);

    values_buff = sycl::malloc_device<DataType>(num_values, q);
    q.wait();
    // q.memset(values_buff, 0, num_values * sizeof(DataType)).wait();
  }

  template <typename... Positions>
  DataType &operator()(const Positions &...positions) const {
    static_assert(sizeof...(Positions) == Dim);
    return values_buff[flatten_index<strides_all...>(padding_width,
                                                     positions...)];
  }

  template <typename... Positions>
  void set_value(DataType val, Positions... position) {
    q.memcpy(&values_buff[flatten_index<strides_all...>(padding_width,
                                                        position...)],
             &val, sizeof(DataType))
        .wait();
  }

  template <typename... Positions> DataType get_value(Positions... position) {
    DataType k;
    std::size_t flat_index =
        flatten_index<strides_all...>(padding_width, position...);
    q.memcpy(&k, &values_buff[flat_index], sizeof(DataType)).wait();

    return k;
  }

  DataType *values_buff;
  Length strides[Dim];
  Length num_values;
  Length num_dofs;
  Length padding_width;
  Paddings padding;
  sycl::queue &q;

  static constexpr std::array<Length, Dim> length{strides_all...};
};

template <Dimension Dim, Length... strides_all>
struct Domain : Grid<DataType, Dim, strides_all...> {
  Domain(Paddings padding, sycl::queue &q, int padding_width)
      : Grid<DataType, Dim, strides_all...>(padding, q, padding_width) {
    q.memset(this->values_buff, 0, this->num_values * sizeof(DataType)).wait();
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

    std::unique_ptr<DataType[]> values{new DataType[this->num_values]};
    this->q
        .memcpy(values.get(), this->values_buff,
                sizeof(DataType) * this->num_values)
        .wait();

    for (int i = 0; i < this->num_values; i++) {

      if (i % 3 == 0 && i != 0) {
        out << std::endl;
      }
      out << format_v(values[i]);
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
    if constexpr (sizeof...(Indices) < Dim) {
      for (Position1D i = 0;
           i < this->strides[sizeof...(Indices)] + 2 * this->padding_width; i++)
        print_domain(indices..., i);
      std::cout << std::endl;
    } else {
      std::cout << std::format("{:6.3f} ", this->get_value(indices...));
    }
  };

  template <typename... Indices>
  void print_domain_to_stream(std::ostream &output, Indices... indices) {
    if constexpr (sizeof...(Indices) < Dim) {
      constexpr auto size = sizeof...(Indices);
      constexpr auto dimension_size =
          utils::get_stack_element<size>(strides_all...);
      for (int i = 0; i < dimension_size + 2 * this->padding_width; i++) {
        print_domain_to_stream(output, indices..., i);
      }
    } else {
      ((std::cout << indices << " "), ...);
      std::cout << std::format("{:6.3f}", this->get_value(indices...))
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

template <Dimension Dim, Length... strides_all, std::size_t... dims>
int domain_compute_norm_squared(DataType &result,
                                Domain<Dim, strides_all...> &a,
                                std::index_sequence<dims...>) {
  DataType *result_device = sycl::malloc_device<DataType>(1, a.q);

  a.q.memset(result_device, 0, sizeof(DataType));

  a.q.parallel_for(sycl::range<Dim>(strides_all...),
                   sycl::reduction(result_device, sycl::plus<>()),
                   [=](sycl::id<Dim> I, auto &acc) {
                     acc += a((I[dims] + a.padding_width)...) *
                            a((I[dims] + a.padding_width)...);
                   });

  a.q.memcpy(&result, result_device, sizeof(DataType)).wait();

  return 0;
}

template <Dimension Dim, Length... strides_all>
int domain_compute_norm_squared(DataType &result,
                                Domain<Dim, strides_all...> &a) {
  return domain_compute_norm_squared(result, a,
                                     std::make_index_sequence<Dim>());
}

template <Dimension Dim, Length... strides_all, std::size_t... dims>
DataType domain_scalar_product(Domain<Dim, strides_all...> &a,
                               Domain<Dim, strides_all...> &b,
                               std::index_sequence<dims...>) {
  DataType *result_device =
      sycl::malloc_device<DataType>(sizeof(DataType), a.q);

  a.q.memset(result_device, 0, sizeof(DataType));

  a.q.parallel_for(sycl::range<Dim>(strides_all...),
                   sycl::reduction(result_device, sycl::plus<>()),
                   [=](sycl::id<Dim> I, auto &acc) {
                     acc += a((I[dims] + a.padding_width)...) *
                            b((I[dims] + b.padding_width)...);
                   })
      .wait();

  DataType result_host;
  a.q.memcpy(&result_host, result_device, sizeof(DataType)).wait();

  return result_host;
}

template <Dimension Dim, Length... strides_all>
DataType domain_scalar_product(Domain<Dim, strides_all...> &a,
                               Domain<Dim, strides_all...> &b) {
  return domain_scalar_product(a, b, std::make_index_sequence<Dim>());
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
  assert(dest.q == a.q);
  assert(dest.num_values == a.num_values);
  assert(dest.padding_width == a.padding_width);

  dest.q
      .parallel_for(sycl::range<1>(a.num_values),
                    [=](sycl::id<1> i) {
                      dest.values_buff[i] = a.values_buff[i] * scalar;
                    })
      .wait();

  return 0;
}

template <Dimension Dim, Length... strides_all>
int divide_domains(Domain<Dim, strides_all...> &dest,
                   Domain<Dim, strides_all...> &a,
                   Domain<Dim, strides_all...> &b) {

  assert(dest.q == a.q && a.q == b.q);
  assert(dest.num_values == a.num_values && b.num_values == a.num_values);
  assert(dest.padding_width == a.padding_width &&
         a.padding_width == b.padding_width);

  dest.q
      .parallel_for(sycl::range<1>(a.num_values),
                    [=](sycl::id<1> i) {
                      dest.values_buff[i] = a.values_buff[i] / b.values_buff[i];
                    })
      .wait();

  return 0;
}

template <Dimension Dim, Length... strides_all>
int multiply_domains(Domain<Dim, strides_all...> &dest,
                     Domain<Dim, strides_all...> &a,
                     Domain<Dim, strides_all...> &b) {

  assert(dest.q == a.q && a.q == b.q);
  assert(dest.num_values == a.num_values && b.num_values == a.num_values);
  assert(dest.padding_width == a.padding_width &&
         a.padding_width == b.padding_width);

  dest.q
      .parallel_for(sycl::range<1>(a.num_values),
                    [=](sycl::id<1> i) {
                      dest.values_buff[i] = a.values_buff[i] * b.values_buff[i];
                    })
      .wait();

  return 0;
}

template <Dimension Dim, Length... strides_all>
int subtract_domains(Domain<Dim, strides_all...> &dest,
                     Domain<Dim, strides_all...> &a,
                     Domain<Dim, strides_all...> &b) {

  assert(dest.q == a.q && a.q == b.q);
  assert(dest.num_values == a.num_values && b.num_values == a.num_values);

  assert(dest.padding_width == a.padding_width &&
         a.padding_width == b.padding_width);

  dest.q
      .parallel_for(sycl::range<1>(a.num_values),
                    [=](sycl::id<1> i) {
                      dest.values_buff[i] = a.values_buff[i] - b.values_buff[i];
                    })
      .wait();

  return 0;
}

template <Dimension Dim, Length... strides_all>
int subtract_and_multiply_domains(Domain<Dim, strides_all...> &dest,
                                  Domain<Dim, strides_all...> &a,
                                  Domain<Dim, strides_all...> &b,
                                  const DataType &val) {

  assert(dest.q == a.q && a.q == b.q);
  assert(dest.num_values == a.num_values && b.num_values == a.num_values);

  assert(dest.padding_width == a.padding_width &&
         a.padding_width == b.padding_width);

  dest.q.parallel_for(sycl::range<1>(a.num_values), [=](sycl::id<1> i) {
    dest.values_buff[i] = val * a.values_buff[i] - b.values_buff[i];
  });

  return 0;
}

template <Dimension Dim, Length... strides_all>
int add_domains(Domain<Dim, strides_all...> &dest,
                Domain<Dim, strides_all...> &a,
                Domain<Dim, strides_all...> &b) {

  assert(dest.q == a.q && a.q == b.q);
  assert(dest.num_values == a.num_values && b.num_values == a.num_values);
  assert(dest.padding_width == a.padding_width &&
         a.padding_width == b.padding_width);

  dest.q
      .parallel_for(sycl::range<1>(a.num_values),
                    [=](sycl::id<1> i) {
                      dest.values_buff[i] = a.values_buff[i] + b.values_buff[i];
                    })
      .wait();

  return 0;
}

template <Dimension Dim, Length... strides_all>
int add_and_multiply_domains(Domain<Dim, strides_all...> &dest,
                             Domain<Dim, strides_all...> &a,
                             Domain<Dim, strides_all...> &b,
                             const DataType &val) {

  assert(dest.q == a.q && a.q == b.q);
  assert(dest.num_values == a.num_values && b.num_values == a.num_values);

  assert(dest.padding_width == a.padding_width &&
         a.padding_width == b.padding_width);

  dest.q
      .parallel_for(sycl::range<1>(a.num_values),
                    [=](sycl::id<1> i) {
                      dest.values_buff[i] =
                          a.values_buff[i] + val * b.values_buff[i];
                    })
      .wait();

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

template <typename Domain, typename... Ranges> struct Subdomain;

template <template <Dimension, Length...> typename Domain, Dimension Dim,
          Length... strides_all, typename... Ranges>
struct Subdomain<Domain<Dim, strides_all...>, Ranges...> {

  Subdomain(Domain<Dim, strides_all...> &domain, Ranges... ranges)
      : parent_domain(domain), values_buff(domain.values_buff),
        padding_width(domain.padding_width) {
    static_assert(sizeof...(Ranges) == Dim);
  }

  template <typename... Positions>
  DataType &operator()(const Positions &...positions) const {
    static_assert(sizeof...(Positions) == Dim);
    return values_buff[flatten_index<strides_all...>(
        padding_width, (positions + RangeProps<Ranges>::start_v)...)];

    // return parent_domain((positions + RangeProps<Ranges>::start_v)...);
  }

  template <typename Position, std::size_t... directions>
  DataType &subscript(Position position,
                      std::index_sequence<directions...>) const {
    static_assert(sizeof...(directions) == sizeof...(Ranges));
    const auto multi_index =
        flat_to_multi_index<RangeProps<Ranges>::length...>(position);
    return this->operator()(multi_index[directions]...);
  }

  template <typename Position> DataType &operator[](Position position) const {
    return subscript(position, std::make_index_sequence<sizeof...(Ranges)>{});
  }

  // template <typename Position> DataType &operator[](Position position) {
  //   const auto multi_index =
  //       flat_to_multi_index<RangeProps<Ranges>::length...>(position);
  //   return std::apply(
  //       [&](const auto &...elems) { return (this->operator()(elems...)); },
  //       multi_index);
  // }

  Domain<Dim, strides_all...> &parent_domain;
  DataType *values_buff;
  DataType padding_width;
};

template <template <Dimension, Length...> typename Domain, Length... length,
          Dimension Dim, typename... Ranges>
Subdomain(Domain<Dim, length...> &, Ranges...)
    -> Subdomain<Domain<Dim, length...>, Ranges...>;

} // namespace domain

#endif
