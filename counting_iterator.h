#include <iterator>

#ifndef COUNTING_ITERATOR_H
#define COUNTING_ITERATOR_H

template <class T> class counting_iterator {
public:
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = const T *;
  using reference = T;
  using iterator_category = std::random_access_iterator_tag;

  constexpr counting_iterator() noexcept : value_() {}
  constexpr explicit counting_iterator(T value) noexcept : value_(value) {}

  // Dereference
  constexpr T operator*() const noexcept { return value_; }

  // Arithmetic
  constexpr counting_iterator &operator++() noexcept {
    ++value_;
    return *this;
  }
  constexpr counting_iterator operator++(int) noexcept {
    counting_iterator tmp = *this;
    ++(*this);
    return tmp;
  }
  constexpr counting_iterator &operator--() noexcept {
    --value_;
    return *this;
  }
  constexpr counting_iterator operator--(int) noexcept {
    counting_iterator tmp = *this;
    --(*this);
    return tmp;
  }

  constexpr counting_iterator &operator+=(difference_type n) noexcept {
    value_ += n;
    return *this;
  }
  constexpr counting_iterator &operator-=(difference_type n) noexcept {
    value_ -= n;
    return *this;
  }

  friend constexpr counting_iterator operator+(counting_iterator it,
                                               difference_type n) noexcept {
    return counting_iterator(it.value_ + n);
  }
  friend constexpr counting_iterator operator+(difference_type n,
                                               counting_iterator it) noexcept {
    return counting_iterator(it.value_ + n);
  }
  friend constexpr counting_iterator operator-(counting_iterator it,
                                               difference_type n) noexcept {
    return counting_iterator(it.value_ - n);
  }
  friend constexpr difference_type
  operator-(const counting_iterator &a, const counting_iterator &b) noexcept {
    return a.value_ - b.value_;
  }

  // Comparison
  friend constexpr bool operator==(const counting_iterator &a,
                                   const counting_iterator &b) noexcept {
    return a.value_ == b.value_;
  }
  friend constexpr bool operator!=(const counting_iterator &a,
                                   const counting_iterator &b) noexcept {
    return !(a == b);
  }
  friend constexpr bool operator<(const counting_iterator &a,
                                  const counting_iterator &b) noexcept {
    return a.value_ < b.value_;
  }
  friend constexpr bool operator>(const counting_iterator &a,
                                  const counting_iterator &b) noexcept {
    return b < a;
  }
  friend constexpr bool operator<=(const counting_iterator &a,
                                   const counting_iterator &b) noexcept {
    return !(b < a);
  }
  friend constexpr bool operator>=(const counting_iterator &a,
                                   const counting_iterator &b) noexcept {
    return !(a < b);
  }

  constexpr T operator[](std::ptrdiff_t n) const noexcept { return value_ + n; }

private:
  T value_;
};

#endif
