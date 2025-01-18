// This is a header function where all the concepts are defined
#include <concepts>

#ifndef CONCEPTS_H
#define CONCEPTS_H

template <typename T, typename Index>
concept subscriptable = requires(T t, Index i) {
  { t[i] };
};

#endif // !CONCEPTS_H
