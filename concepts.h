// This is a header function where all the concepts are defined
#include <concepts>
#include <type_traits>

#ifndef CONCEPTS_H
#define CONCEPTS_H

template <typename T, typename Index>
concept subscriptable = requires(T t, Index i) {
  { t[i] };
};

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template <typename T>
concept Integral = std::is_integral_v<T>;

#endif // !CONCEPTS_H
