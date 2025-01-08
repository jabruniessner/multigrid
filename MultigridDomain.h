#include "Domain.h"
#include "level_transition.h"
#include "predefinitions.h"
#include "utils.h"
#include <format>
#include <utility>

#ifndef MULTIGRIDDOMAIN_H
#define MULTIGRIDDOMAIN_H

namespace multigrid_domain {

using namespace domain;

constexpr Length nlev = 5;
constexpr Length values_1D = utils::Power<2u, nlev>::value - 1;

template <Dimension Dim, std::size_t nlev, std::size_t... base_length>
struct Multigrid_domain
    : public Multigrid_domain<Dim, nlev - 1, base_length...> {

  using OffsetType = std::array<int, Dim>;

  Multigrid_domain(sycl::queue &q)
      : domain(Paddings::PERIODIC, q, 1),
        Multigrid_domain<Dim, nlev - 1, base_length...>(q) {};

  template <std::size_t lev = nlev, typename... Position1D>
  DataType get_value(Position1D... i) {
    static_assert(lev <= nlev, "level too large!");
    static_assert(sizeof...(Position1D) == Dim);
    return Multigrid_domain<Dim, lev, base_length...>::domain.get_value(i...);
  }

  template <std::size_t lev = nlev, typename... Position1D>
  void set_value(DataType val, Position1D... i) {
    static_assert(lev <= nlev, "level too large!");
    static_assert(sizeof...(Position1D) == Dim);
    Multigrid_domain<Dim, lev, base_length...>::domain.set_value(val, i...);
  }

  template <std::size_t lev = nlev>
  decltype(Multigrid_domain<Dim, lev, base_length...>::domain) &get_domain() {
    return Multigrid_domain<Dim, lev, base_length...>::domain;
  }

  void print_level() { std::cout << nlev << std::endl; }

  void print_length() { std::cout << std::get<0>(length) << std::endl; }

  template <std::size_t level, std::size_t length>
  void coarsening(std::array<DataType, length> values,
                  std::array<OffsetType, length> offsets) {
    auto &upper_domain = this->template get_domain<level>();
    auto &lower_domain = this->template get_domain<level - 1>();

    level_transition::coarsening(lower_domain, upper_domain,
                                 coarse_filter_values, coarse_filter_offsets);
  }

  template <std::size_t level = nlev> auto get_length() {
    return Multigrid_domain<Dim, level, base_length...>::length;
  }

  template <std::size_t... indices> struct Domain_Type {
    constexpr Domain_Type(std::index_sequence<indices...>) {
      static_assert(sizeof...(indices) == Dim);
    }
    constexpr static std::tuple length =
        std::make_tuple((base_length * utils::Power<2u, nlev>::value - 1)...);
    using domain_t = Domain<Dim, std::get<indices>(length)...>;
  };

  template <std::size_t... indices>
  Domain_Type(std::index_sequence<indices...>) -> Domain_Type<indices...>;

  constexpr static std::array<DataType, 1> coarse_filter_values{1};
  constexpr static std::array<OffsetType, 1> coarse_filter_offsets{{{0, 0}}};
  constexpr static Domain_Type domain_t_v{std::make_index_sequence<Dim>{}};
  constexpr static auto &length = decltype(domain_t_v)::length;

  decltype(domain_t_v)::domain_t domain;
};

template <Dimension Dim, std::size_t... base_length>
struct Multigrid_domain<Dim, 0u, base_length...> {
  Multigrid_domain(sycl::queue &q) : domain(Paddings::PERIODIC, q, 1) {}

  template <typename... Position1D> DataType get_value(Position1D... i) {
    return domain.get_value(i...);
  }

  template <typename... Position1D>
  void set_value(DataType val, Position1D... i) {
    domain.set_value(val, i...);
  }

  void print_level() { std::cout << 0u << std::endl; }

  std::size_t get_base_length() { return utils::get_first(base_length...); }

  template <std::size_t... indices> struct Domain_Type {
    constexpr Domain_Type(std::index_sequence<indices...>) {
      static_assert(sizeof...(indices) == Dim);
    }
    constexpr static std::tuple length =
        std::make_tuple((base_length * utils::Power<2u, nlev>::value - 1)...);
    using domain_t = Domain<Dim, std::get<indices>(length)...>;
  };

  template <std::size_t... indices>
  Domain_Type(std::index_sequence<indices...>) -> Domain_Type<indices...>;

  constexpr static std::array<DataType, 1> coarse_filter_values{1};
  constexpr static std::array<OffsetType, 1> coarse_filter_offsets{{{0, 0}}};

  decltype(Domain_Type(std::make_index_sequence<Dim>{}))::domain_t domain;
};

template <Dimension Dim, std::size_t nlev, std::size_t level = nlev,
          std::size_t... base_length>
void print_multigrid_domain(
    Multigrid_domain<Dim, nlev, base_length...> &MultDomain) {
  if constexpr (level == 0) {
    return;
  } else {

    std::cout << "\n\n\n";
    std::cout << "Level: " << level << std::endl;
    MultDomain.template get_domain<level>().print_domain();
    print_multigrid_domain<Dim, nlev, level - 1, base_length...>(MultDomain);
  }
}

template <Dimension Dim, typename DataType, std::size_t length,
          Length base_length, std::size_t nlev>
struct Multi_Level_operator
    : public Multi_Level_operator<Dim, DataType, length, base_length,
                                  nlev - 1> {
  using OffsetType = std::array<int, Dim>;

  Multi_Level_operator() {};

  Multi_Level_operator(Integer<nlev>, std::array<DataType, length> &values,
                       std::array<OffsetType, length> &offsets,
                       Integer<base_length>)
      : values(values), offsets(offsets),
        Multi_Level_operator<Dim, DataType, length, base_length, nlev - 1>(
            Integer<nlev - 1>{}, values, offsets, Integer<base_length>{}) {};

  Multi_Level_operator(Integer<nlev>, std::array<DataType, length> &values_new,
                       std::array<OffsetType, length> &offsets,
                       DataType Box_Length, Integer<base_length>)
      : offsets(offsets),
        Multi_Level_operator<Dim, DataType, length, base_length, nlev - 1>(
            Integer<nlev - 1>{}, values_new, offsets, Box_Length,
            Integer<base_length>{}) {

    auto num_points =
        std::get<0>(Multigrid_domain<1, nlev, base_length>::length) + 1;

    auto h = Box_Length / num_points;
    for (int i = 0; i < length; i++) {

      this->values[i] = values_new[i] / (h * h);
    }
  };

  Multi_Level_operator(Integer<nlev>, std::array<DataType, length> &&values,
                       std::array<OffsetType, length> &&offsets,
                       Integer<base_length>)
      : values(values), offsets(offsets),
        Multi_Level_operator<Dim, DataType, length, nlev - 1, base_length>(
            Integer<nlev - 1>{}, values, offsets) {};

  template <std::size_t access_level = nlev> auto &get_values() {
    return Multi_Level_operator<Dim, DataType, length, base_length,
                                access_level>::values;
  }

  template <std::size_t access_level = nlev> auto &get_offsets() {
    return Multi_Level_operator<Dim, DataType, length, base_length,
                                access_level>::offsets;
  }

  void print_operator() {
    std::cout << "The level is: " << nlev << std::endl;
    std::cout << std::endl << std::endl;

    for (int i = 0; i < length; i++) {
      for (int j = 0; j < Dim; j++)
        std::cout << " " << std::format("{:>2}", offsets[i][j]);
      std::cout << ":";
      std::cout << " " << values[i] << std::endl;
    }

    Multi_Level_operator<Dim, DataType, length, base_length,
                         nlev - 1>::print_operator();
  }

  std::array<DataType, length> values;
  std::array<OffsetType, length> offsets;
};

template <Dimension Dim, typename DataType, std::size_t length,
          Length base_length>
struct Multi_Level_operator<Dim, DataType, length, base_length, 1u> {
  using OffsetType = std::array<int, Dim>;

  Multi_Level_operator(Integer<1>, std::array<DataType, length> &values,
                       std::array<OffsetType, length> &offsets,
                       Integer<base_length>)
      : values(values), offsets(offsets) {};

  Multi_Level_operator(Integer<1>, std::array<DataType, length> &values_new,
                       std::array<OffsetType, length> &offsets,
                       DataType Box_Length, Integer<base_length>)
      : offsets(offsets) {

    auto num_points =
        std::get<0>(Multigrid_domain<1, 1u, base_length>::length) + 1;
    auto h = Box_Length / num_points;
    for (int i = 0; i < length; i++) {

      this->values[i] = values_new[i] / (h * h);
    }
  }

  Multi_Level_operator(Integer<1>, std::array<DataType, length> &&values,
                       std::array<OffsetType, length> &&offsets,
                       Integer<base_length>)
      : values(values), offsets(offsets) {};

  auto get_values() { return values; }

  auto get_offsets() { return offsets; }

  void print_operator() {
    std::cout << "The level is: " << 1 << std::endl;
    std::cout << std::endl << std::endl;

    for (int i = 0; i < length; i++) {
      for (int j = 0; j < Dim; j++)
        std::cout << " " << std::format("{:>2}", offsets[i][j]);
      std::cout << ":";
      std::cout << " " << values[i] << std::endl;
    }
  }

  std::array<DataType, length> values;
  std::array<OffsetType, length> offsets;
};

} // namespace multigrid_domain

#endif
