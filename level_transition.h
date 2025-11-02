#include "Convolution.h"
#include "utils.h"
#include <type_traits>
#include <unistd.h>

#ifdef __CUDACC__
#include <thrust/iterator/counting_iterator.h>
#else
#include <boost/iterator/counting_iterator.hpp>
#endif

#ifndef LEVEL_TRANSITION
#define LEVEL_TRANSITION

namespace level_transition {

using namespace domain;

template <typename dtype_dest, typename dtype_src, typename Offsets,
          std::size_t size, std::size_t... dims>
struct coarsening_helper {

  __host__ __device__ void operator()(std::size_t idx) const {
    constexpr auto strides = dtype_dest::length;
    auto I = domain::flat_to_multi_index<strides[dims]...>(idx);
    ((I[dims] += dest.padding_width), ...);
    decltype(I) I_fine;
    ((I_fine[dims] = 2 * I[dims]), ...);

    DataType result = 0;

    for (int k = 0; k < size; k++) {
      result += src((I_fine[dims] + offsets[k][dims])...) * values[k];
    }

    dest(I[dims]...) = result;
  }

  const domain::array<Offsets, size> offsets;
  const domain::array<DataType, size> values;
  dtype_dest dest;
  dtype_src src;
};

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all, std::size_t... dims>
void coarsening(Domain<Dim, ((strides_all + 1) / 2 - 1)...> &dest,
                Domain<Dim, strides_all...> &src,
                const domain::array<DataType, size> values,
                const domain::array<Offsets, size> offsets,
                std::index_sequence<dims...>) {

#ifdef __CUDACC__
  using iterator = thrust::counting_iterator<int>;
#else
  using iterator = boost::iterators::counting_iterator<int>
#endif
  iterator start(0);
  iterator end(dest.num_dofs);

  using dtype_src = std::remove_reference_t<decltype(src)>;
  using dtype_dest = std::remove_reference_t<decltype(dest)>;

  coarsening_helper<dtype_dest, dtype_src, Offsets, size, dims...> ch{
      offsets, values, dest, src};

  thrust::for_each(start, end, ch);
}

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all>
void coarsening(Domain<Dim, ((strides_all + 1) / 2 - 1)...> &dest,
                Domain<Dim, strides_all...> &src,
                const domain::array<DataType, size> values,
                const domain::array<Offsets, size> offsets) {
  coarsening(dest, src, values, offsets, std::make_index_sequence<Dim>());
}

template <typename dtype_dest, typename dtype_src, std::size_t... dims>
struct coarseing_inject_helper {
  __host__ __device__ void operator()(std::size_t idx) const {
    constexpr auto strides = dtype_src::length;
    auto I = domain::flat_to_multi_index<((strides[dims] + 1) / 2 - 1)...>(idx);

    ((I[dims] += dest.padding_width), ...);
    decltype(I) I_fine{2 * I[dims]...};
    dest(I[dims]...) = src(I_fine[dims]...);
  }

  dtype_dest dest;
  dtype_src src;
};

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all, std::size_t... dims>
void coarsening_inject(Domain<Dim, ((strides_all + 1) / 2 - 1)...> &dest,
                       Domain<Dim, strides_all...> &src,
                       const domain::array<DataType, size> values,
                       const domain::array<Offsets, size> offsets,
                       std::index_sequence<dims...>) {

#ifdef __CUDACC__
  using iterator = thrust::counting_iterator<int>;
#else
  using iterator = boost::iterators::counting_iterator<int>
#endif

  iterator start(0);
  iterator end(dest.num_dofs);

  using dtype_dest = std::remove_reference_t<decltype(dest)>;
  using dtype_src = std::remove_reference_t<decltype(src)>;

  coarseing_inject_helper<dtype_dest, dtype_src, dims...> cih{dest, src};

  thrust::for_each(start, end, cih);
}

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all>
void coarsening_inject(Domain<Dim, ((strides_all + 1) / 2 - 1)...> &dest,
                       Domain<Dim, strides_all...> &src,
                       const std::array<DataType, size> values,
                       const std::array<Offsets, size> offsets) {
  coarsening_inject(dest, src, values, offsets,
                    std::make_index_sequence<Dim>());
}

template <typename DataType, size_t size, std::size_t... strides_all>
void coarsening_inject_sequential(
    Domain<3, ((strides_all + 1) / 2 - 1)...> &dest,
    Domain<3, strides_all...> &src) {
  assert(dest.q == src.q);
  // assert(dest.padding_width == src.padding_width);
  constexpr std::array<const std::size_t, 3> a{strides_all...};

  int i_coarse = 1;
  for (int i = 1; i <= a[0]; i += 2) {
    int j_coarse = 1;
    for (int j = 1; j <= a[1]; j += 2) {
      int k_coarse = 1;
      for (int k = 1; k <= a[2]; k += 2) {
        dest(i_coarse++, j_coarse++, k_coarse++) = src(i, j, k);
      }
    }
  }

  // dest.q.wait();
}

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all, std::size_t... dims>
void coarsening_and_copy(Domain<Dim, ((strides_all + 1) / 2 - 1)...> &dest1,
                         Domain<Dim, ((strides_all + 1) / 2 - 1)...> &dest2,
                         Domain<Dim, strides_all...> &src,
                         const std::array<DataType, size> values,
                         const std::array<Offsets, size> offsets,
                         std::index_sequence<dims...>) {

#ifdef __CUDACC__
  using iterator = thrust::counting_iterator<int>;
#else
  using iterator = boost::iterators::counting_iterator<int>
#endif

  iterator start(0);
  iterator end(dest1.num_dofs);

  thrust::for_each(start, end, [=](int idx) {
    auto I = domain::flat_to_multi_index<((strides_all + 1) / 2 - 1)...>(idx);
    ((I[dims] += dest1.padding_width), ...);
    decltype(I) I_fine;
    ((I_fine[dims] = 2 * I[dims]), ...);

    DataType result = 0;

    for (int k = 0; k < size; k++) {
      result += src((I_fine[dims] + offsets[k][dims])...) * values[k];
    }

    dest1(I[dims]...) = dest2(I[dims]...) = result;
  });
}

template <typename DataType, typename Offsets, size_t size, Dimension Dim,
          Length... strides_all>
void coarsening_and_copy(Domain<Dim, ((strides_all + 1) / 2 - 1)...> &dest1,
                         Domain<Dim, ((strides_all + 1) / 2 - 1)...> &dest2,
                         Domain<Dim, strides_all...> &src,
                         const std::array<DataType, size> values,
                         const std::array<Offsets, size> offsets) {
  coarsening_and_copy(dest1, dest2, src, values, offsets,
                      std::make_index_sequence<Dim>());
}

template <Dimension Dim, Length... strides_all, typename... Index,
          typename... Rest_indices>
__host__ __device__ DataType domain_refinement_helper(
    const Domain<Dim, strides_all...> &dom, std::tuple<Index...> &index_tuple,
    Rest_indices &...rest_indices) {
  if constexpr (sizeof...(Index) == Dim) {
    std::apply([&](auto &&...args) { ((args /= 2), ...); }, index_tuple);

    return std::apply(dom, index_tuple);
  } else {
    auto first = utils::get_first(rest_indices...);
    auto rest = utils::get_tail(rest_indices...);

    if (first % 2 == 0) {

      auto new_index_tuple =
          std::tuple_cat(index_tuple, std::make_tuple(first));
      auto first_arg_tuple = std::forward_as_tuple(dom, new_index_tuple);
      auto arg_tuple = std::tuple_cat(
          std::forward<decltype(first_arg_tuple)>(first_arg_tuple), rest);

      return std::apply(
          [](auto &&...args) { return domain_refinement_helper(args...); },
          arg_tuple);

    } else {
      auto new_index_tuple_1 =
          std::tuple_cat(index_tuple, std::make_tuple(first + 1));

      auto new_index_tuple_2 =
          std::tuple_cat(index_tuple, std::make_tuple(first - 1));

      auto first_arg_tuple_1 = std::forward_as_tuple(dom, new_index_tuple_1);

      auto first_arg_tuple_2 = std::forward_as_tuple(dom, new_index_tuple_2);

      auto arg_tuple_1 = std::tuple_cat(
          std::forward<decltype(first_arg_tuple_1)>(first_arg_tuple_1), rest);

      auto arg_tuple_2 = std::tuple_cat(
          std::forward<decltype(first_arg_tuple_2)>(first_arg_tuple_2), rest);

      auto return_value = std::apply(
          [](auto &&...args) { return domain_refinement_helper(args...); },
          arg_tuple_1);

      return_value += std::apply(
          [](auto &&...args) { return domain_refinement_helper(args...); },
          arg_tuple_2);

      return return_value / 2;
    }
  }

  return 0.;
}

template <typename dtype_dest, typename dtype_src, std::size_t... dims>
struct refinement_helper_struct {

  __host__ __device__ void operator()(std::size_t idx) const {
    constexpr auto strides = dtype_dest::length;
    auto I = domain::flat_to_multi_index<strides[dims]...>(idx);
    ((I[dims] += dest.padding_width), ...);
    std::tuple<> empty_index_tuple;
    dest(I[dims]...) =
        domain_refinement_helper(src, empty_index_tuple, I[dims]...);
  }

  dtype_dest dest;
  dtype_src src;
};

template <Dimension Dim, Length... strides_all, std::size_t... dims>
void refinement(Domain<Dim, strides_all...> &dest,
                Domain<Dim, ((strides_all + 1) / 2 - 1)...> &src,
                std::index_sequence<dims...>) {

#ifdef __CUDACC__
  using iterator = thrust::counting_iterator<int>;
#else
  using iterator = boost::iterators::counting_iterator<int>
#endif

  iterator start(0);
  iterator end(dest.num_dofs);

  using dtype_dest = std::remove_reference_t<decltype(dest)>;
  using dtype_src = std::remove_reference_t<decltype(src)>;
  refinement_helper_struct<dtype_dest, dtype_src, dims...> rhs{dest, src};

  thrust::for_each(start, end, rhs);
}

template <Dimension Dim, Length... strides_all>
void refinement(Domain<Dim, strides_all...> &dest,
                Domain<Dim, ((strides_all + 1) / 2 - 1)...> &src) {
  refinement(dest, src, std::make_index_sequence<Dim>());
}

template <Dimension Dim, Length... strides_all, std::size_t... dims>
void refinement_and_copy(Domain<Dim, strides_all...> &dest1,
                         Domain<Dim, strides_all...> &dest2,
                         Domain<Dim, ((strides_all + 1) / 2 - 1)...> &src,
                         std::index_sequence<dims...>) {
  assert(dest1.q == src.q && dest2.q == src.q);
  // assert(dest.padding_width == src.padding_width);

#ifdef __CUDACC__
  using iterator = thrust::counting_iterator<int>;
#else
  using iterator = boost::iterators::counting_iterator<int>
#endif

  iterator start(0);
  iterator end(dest1.num_dofs);

  thrust::for_each(start, end, [=](int idx) {
    auto I = domain::flat_to_multi_index<strides_all...>(idx);
    ((I[dims] += dest1.padding_width), ...);
    std::tuple<> empty_index_tuple;
    dest1(I[dims]...) = dest2(I[dims]...) =
        domain_refinement_helper(src, empty_index_tuple, I[dims]...);
  });
}

template <Dimension Dim, Length... strides_all>
void refinement_and_copy(Domain<Dim, strides_all...> &dest1,
                         Domain<Dim, strides_all...> &dest2,
                         Domain<Dim, ((strides_all + 1) / 2 - 1)...> &src) {
  refinement_and_copy(dest1, dest2, src, std::make_index_sequence<Dim>());
}
}; // namespace level_transition

#endif
