#ifndef BITSHIFT_LIB_H
#define BITSHIFT_LIB_H

#include "blas.h"
#include "concepts.h"
#include "utils.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace bitshift {

template <UnsignedIntegral num_type, Length len>
num_type find_first_set_bit_index(num_type num) {
  if (num == 0)
    return (num_type)utils::power_off(2, len);
  int bit_num = 0;
  while (((num >> (bit_num)) & 1) == 0) {
    bit_num++;
  };
  return bit_num;
}

template <UnsignedIntegral num_type, Length len>
inline num_type get_one_or_two(num_type num1, num_type num2) {
  num_type mask = num1 & ~(num1 - 1) & (utils::power_off(2, len) - 1);

  return (num2 & mask) ? 0 : 1;
}

template <UnsignedIntegral num_type> inline num_type lsb(num_type num) {
  return num & (~num + 1);
}

template <UnsignedIntegral num_type, Length len>
inline num_type get_one_index_from_two(num_type num1, num_type num2) {
  num_type other = ~num1;
  return find_first_set_bit_index<num_type, len>(num1) * 2 +
         get_one_or_two<num_type, len>(other, num2);
}

template <UnsignedIntegral num_type>
inline std::tuple<num_type, num_type> get_two_indices_from_one(num_type num) {

  num_type k = (1 << (num / 2));
  return {k, k | (num % 2 ? lsb((~k) & (~k - 1)) : lsb(~k))};
}

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
