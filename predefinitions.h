#include <array>
#include <cstddef>
#include <iostream>
#include <sycl/sycl.hpp>

#ifndef PREDEFINITIONS_H
#define PREDEFINITIONS_H

template <std::size_t Int> struct Integer {};

class LinearAlgebraDouble {
public:
  using DataType = double;
  using Length = std::size_t;
  using Position1D = std::size_t;
  // using PositionType = sycl::id<2>;
  using OffsetType = std::array<int, 2>;
  using Dimension = std::size_t;
};

using LAD = LinearAlgebraDouble;
using DataType = LAD::DataType;
using Length = LAD::Length;
using Position1D = LAD::Position1D;
using OffsetType = LAD::OffsetType;
// using PositionType = LAD::PositionType;
using Dimension = LAD::Dimension;

template <DataType val> struct Float {};

template <typename T> void gpu_print(T *val, sycl::queue &q) {
  T val_host;
  q.memcpy(&val_host, val, sizeof(T));
  q.wait();
  std::cout << val_host << std::endl;
}

enum class Paddings {
  PERIODIC = 0,
  MIRROR = 1,
  ZERO = 2,
};

#endif
