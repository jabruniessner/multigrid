#ifndef BITSHIFT_LIB_H
#define BITSHIFT_LIB_H

#include "blas.h"
#include "concepts.h"
#include "utils.h"
#include <cstdint>
#include <type_traits>
#include <utility>

namespace bitshift {

template <UnsignedIntegral num_type, Dimension Dim>
using num_vector = blas::vector<num_type, Dim>;

template <std::size_t Length, std::size_t steps = 1,
          UnsignedIntegral num_type = std::uint8_t>
inline num_type cycle_right(num_type i) {

  static_assert(
      Length <= sizeof(num_type) * 8,
      "The length need to be smaller than the number of bits in the num type");

  for (int j = 1; j <= steps; j++)
    i = (i >> 1 | i << (Length - 1)) & (utils::Power<2, Length>::value - 1);
  return i;
}

template <std::size_t Length, std::size_t steps = 1,
          UnsignedIntegral num_type = std::uint8_t>
inline num_type cycle_left(num_type i) {

  static_assert(
      Length <= sizeof(num_type) * 8,
      "The length need to be smaller than the number of bits in the num type");

  for (int j = 1; j <= steps; j++)
    i = (i << 1 | i >> (Length - 1)) & (utils::Power<2, Length>::value - 1);
  return i;
}

template <UnsignedIntegral num_type, UnsignedIntegral num_type2, Dimension Dim,
          std::size_t... dims>
auto convert_byte_to_vec(num_type2 number, std::index_sequence<dims...>) {
  return num_vector<num_type, Dim>{
      static_cast<num_type>(number >> (Dim - dims - 1) & 1)...};
}

template <UnsignedIntegral num_type, UnsignedIntegral num_type2, Dimension Dim>
auto convert_byte_to_vec(num_type2 num) {
  return convert_byte_to_vec<num_type, num_type2, Dim>(
      num, std::make_index_sequence<Dim>{});
}
}; // namespace bitshift

#endif
