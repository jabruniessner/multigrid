#include <type_traits>

#ifndef CONCEPTS_H
#define CONCEPTS_H

template <typename T, typename Index>
concept subscriptable = requires(T t, Index i) {
  { t[i] };
};

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template <typename num_type>
concept UnsignedIntegral =
    std::is_integral_v<num_type> && !std::is_signed_v<num_type>;

template <typename stuff>
concept has_initialize_residual_dependancy =
    requires(stuff t, double residual) {
      { t.initialize_residual_dependancy(residual) };
    };

#endif // !CONCEPTS_H
