#include "Domain.h"
#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/usm.hpp"
#include "level_transition.h"
#include "predefinitions.h"
#include "scientific_quantities.h"
#include "utils.h"
#include <format>
#include <tuple>
#include <type_traits>
#include <utility>

#ifndef MULTIGRIDDOMAIN_H
#define MULTIGRIDDOMAIN_H

namespace multigrid_domain {

using namespace domain;

// constexpr Length nlev = 5;
// constexpr Length values_1D = utils::Power<2u, nlev>::value - 1;

template <typename DataType, Dimension Dim, Dimension Type_dim,
          std::size_t nlev, std::size_t... base_length>
struct Multigrid_domain_t
    : public Multigrid_domain_t<DataType, Dim, Type_dim, nlev - 1,
                                base_length...> {

  template <std::size_t... indices> struct Domain_Type;

  template <std::size_t... indices>
  Domain_Type(std::index_sequence<indices...>) -> Domain_Type<indices...>;

  constexpr static std::array<DataType, 1> coarse_filter_values{1};
  constexpr static std::array<OffsetType, 1> coarse_filter_offsets{{{0, 0}}};
  constexpr static Domain_Type domain_t_v{
      std::make_index_sequence<Dim + Type_dim>{}};

  template <std::size_t lev = nlev>
  using domain_type = decltype(Multigrid_domain_t<DataType, Dim, Type_dim, lev,
                                                  base_length...>::domain_t_v);

  constexpr static auto &length = decltype(domain_t_v)::length;

  using ValueType = DataType;

  Multigrid_domain_t(sycl::queue &q)
      : domain(Paddings::PERIODIC, q, 1),
        Multigrid_domain_t<DataType, Dim, Type_dim, nlev - 1, base_length...>(
            q) {};

  template <std::size_t lev = nlev, typename... Position1D>
  DataType get_value(Position1D... i) {
    static_assert(lev <= nlev, "level too large!");
    static_assert(sizeof...(Position1D) == Dim + Type_dim);
    return Multigrid_domain_t<DataType, Dim, Type_dim, lev,
                              base_length...>::domain.get_value(i...);
  }

  template <std::size_t lev = nlev, typename... Position1D>
  void set_value(DataType val, Position1D... i) {
    static_assert(lev <= nlev, "level too large!");
    static_assert(sizeof...(Position1D) == Dim + Type_dim);
    Multigrid_domain_t<DataType, Dim, Type_dim, lev, base_length...>::domain
        .set_value(val, i...);
  }

  template <std::size_t lev = nlev, std::size_t dummy = 0>
  domain_type<lev>::domain_t &get_domain() {
    return Multigrid_domain_t<DataType, Dim, Type_dim, lev,
                              base_length...>::domain;
  }

  template <std::size_t lev = nlev>
  const domain_type<lev>::domain_t &get_domain() const {
    return Multigrid_domain_t<DataType, Dim, Type_dim, lev,
                              base_length...>::domain;
  }

  template <std::size_t lev = nlev, std::size_t dummy = 0>
  domain_type<lev>::domain_t_wrapper &get_domain_wrapper() {
    return Multigrid_domain_t<DataType, Dim, Type_dim, lev,
                              base_length...>::domain;
  }

  template <std::size_t lev = nlev>
  const domain_type<lev>::domain_t_wrapper &get_domain_wrapper() const {
    return Multigrid_domain_t<DataType, Dim, Type_dim, lev,
                              base_length...>::domain;
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
    return Multigrid_domain_t<DataType, Dim, Type_dim, level,
                              base_length...>::length;
  }

  template <typename DataType2>
  std::enable_if_t<std::is_arithmetic_v<DataType> &&
                       std::is_same_v<DataType, DataType2>,
                   void>
  set_zero() {
    this->get_domain().template set_zero<DataType2>();
    this->Multigrid_domain_t<DataType, Dim, Type_dim, nlev - 1,
                             base_length...>::template set_zero<DataType2>();
  }

  template <std::size_t... indices> struct Domain_Type {
    constexpr Domain_Type(std::index_sequence<indices...>) {
      static_assert(sizeof...(indices) == Dim + Type_dim);
    }

    // Making tuple for multigrid dimensions

    constexpr static auto base_lengths = std::make_tuple(base_length...);
    constexpr static auto base_length_mg =
        utils::extract_tuple(base_lengths, std::make_index_sequence<Dim>{});

    constexpr static auto base_length_type = utils::extract_tuple(
        base_lengths,
        utils::add_offset<Dim>(std::make_index_sequence<Type_dim>{}));

    constexpr static auto length_func = [](auto length) {
      return length * utils::power_off(2, nlev) - 1;
    };

    constexpr static std::tuple length =
        utils::tuple_map(base_length_mg, length_func);

    constexpr static std::tuple length_all =
        std::tuple_cat(length, base_length_type);

    using domain_t =
        Grid<DataType, Dim + Type_dim, std::get<indices>(length_all)...>;

    using domain_t_wrapper = Grid_wrapper<DataType, Dim + Type_dim,
                                          std::get<indices>(length_all)...>;
  };

  // template <std::size_t... indices>
  // Domain_Type(std::index_sequence<indices...>) -> Domain_Type<indices...>;

  decltype(domain_t_v)::domain_t_wrapper domain;
};

template <typename DataType, Dimension Dim, Dimension Type_dim,
          std::size_t... base_length>
struct Multigrid_domain_t<DataType, Dim, Type_dim, 0u, base_length...> {

  using ValueType = DataType;

  template <std::size_t... indices> struct Domain_Type;

  template <std::size_t... indices>
  Domain_Type(std::index_sequence<indices...>) -> Domain_Type<indices...>;

  constexpr static std::array<DataType, 1> coarse_filter_values{1};
  constexpr static std::array<OffsetType, 1> coarse_filter_offsets{{{0, 0}}};
  constexpr static Domain_Type domain_t_v{
      std::make_index_sequence<Dim + Type_dim>{}};

  template <std::size_t lev = 0>
  using domain_type = decltype(Multigrid_domain_t<DataType, Dim, Type_dim, lev,
                                                  base_length...>::domain_t_v);

  Multigrid_domain_t(sycl::queue &q) {}

  template <typename... Position1D> DataType get_value(Position1D... i) {
    return 0;
  }

  template <typename... Position1D>
  void set_value(DataType val, Position1D... i) {}

  void print_level() { std::cout << 0u << std::endl; }

  std::size_t get_base_length() { return utils::get_first(base_length...); }

  template <std::size_t... indices> struct Domain_Type {
    constexpr Domain_Type(std::index_sequence<indices...>) {
      static_assert(sizeof...(indices) == Dim + Type_dim);
    }

    // Making tuple for multigrid dimensions

    constexpr static auto base_lengths = std::make_tuple(base_length...);
    constexpr static auto base_length_mg =
        utils::extract_tuple(base_lengths, std::make_index_sequence<Dim>{});

    constexpr static auto base_length_type = utils::extract_tuple(
        base_lengths,
        utils::add_offset<Dim>(std::make_index_sequence<Type_dim>{}));

    constexpr static auto length_func = [](auto length) {
      return length * utils::power_off(2, 0u) - 1;
    };

    constexpr static std::tuple length =
        utils::tuple_map(base_length_mg, length_func);

    constexpr static std::tuple length_all =
        std::tuple_cat(length, base_length_type);

    using domain_t =
        Grid<DataType, Dim + Type_dim, std::get<indices>(length_all)...>;

    using domain_t_wrapper = Grid_wrapper<DataType, Dim + Type_dim,
                                          std::get<indices>(length_all)...>;
  };

  template <typename DataType2>
  std::enable_if_t<std::is_arithmetic_v<DataType> &&
                       std::is_same_v<DataType, DataType2>,
                   void>
  set_zero() {}

  // decltype(domain_t_v)::domain_t domain;
};

template <Dimension Dim, std::size_t nlev, std::size_t... base_length>
using Multigrid_domain =
    Multigrid_domain_t<DataType, Dim, 0u, nlev, base_length...>;

template <Dimension Dim, std::size_t nlev, std::size_t level = nlev,
          std::size_t... base_length>
void print_multigrid_domain(
    Multigrid_domain_t<DataType, Dim, nlev, base_length...> &MultDomain) {
  if constexpr (level == 0) {
    return;
  } else {

    std::cout << "\n\n\n";
    std::cout << "Level: " << level << std::endl;
    MultDomain.template get_domain<level>().print_domain();
    print_multigrid_domain<Dim, nlev, level - 1, base_length...>(MultDomain);
  }
}

template <Dimension Dim, typename DataType,
          template <Dimension, std::size_t> typename Func_type,
          std::size_t nlev>
struct Multi_Level_map
    : public Multi_Level_map<Dim, DataType, Func_type, nlev - 1> {

  Multi_Level_map(auto... map_init_args)
      : map(map_init_args...),
        Multi_Level_map<Dim, DataType, Func_type, nlev - 1>(map_init_args...) {
        };

  Multi_Level_map(Integer<nlev>, Integer<Dim>)
      : Multi_Level_map<Dim, DataType, Func_type, nlev - 1>(Integer<nlev - 1>{},
                                                            Integer<Dim>{}) {};

  template <std::size_t access_level = nlev>
  Func_type<Dim, access_level> get_map() {
    return Multi_Level_map<Dim, DataType, Func_type, access_level>::map;
  }

  Func_type<Dim, nlev> map;
};

template <Dimension Dim, typename DataType,
          template <Dimension, std::size_t> typename FuncType>
struct Multi_Level_map<Dim, DataType, FuncType, 1u> {
  using OffsetType = std::array<int, Dim>;

  Multi_Level_map(auto... map_init_args) : map(map_init_args...) {}
  Multi_Level_map(Integer<1>, Integer<Dim>) {};

  template <std::size_t access_level = 1> FuncType<Dim, 1u> get_map() {
    return map;
  }

  FuncType<Dim, 1u> map;
};

template <Dimension Dim, typename Functype, Length... strides_all,
          std::size_t... dirs>
DataType get_ideal_omega(Domain<Dim, strides_all...> x,
                         Domain<Dim, strides_all...> r,
                         Domain<Dim, strides_all...> b, Functype map,
                         std::index_sequence<dirs...>) {

  sycl::queue &q = x.q;

  DataType *upper_lower =
      sycl::malloc_device<DataType>(sizeof(DataType) * 2, q);

  q.parallel_for(sycl::range{strides_all...},
                 sycl::reduction(upper_lower, sycl::plus<>()),
                 sycl::reduction(upper_lower + 1, sycl::plus<>()),
                 [=](sycl::id<Dim> I, auto &acc_upper, auto &acc_lower) {
                   ((I[dirs] += x.padding_width), ...);
                   acc_upper += r(I[dirs]...) * b(I[dirs]...) -
                                r(I[dirs]...) * map(x, I);

                   acc_lower += r(I[dirs]...) * map(r, I);
                 })
      .wait();

  std::array<DataType, 2> upper_lower_host;
  q.memcpy(upper_lower_host.data(), upper_lower, sizeof(DataType) * 2).wait();

  return upper_lower_host[0] / upper_lower_host[1];
}

template <Dimension Dim, typename FuncType, Length... strides_all>
DataType get_ideal_omega(Domain<Dim, strides_all...> x,
                         Domain<Dim, strides_all...> r,
                         Domain<Dim, strides_all...> b, FuncType map) {
  return get_ideal_omega(x, r, b, map, std::make_index_sequence<Dim>{});
}

} // namespace multigrid_domain

#endif
