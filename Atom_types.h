#include "predefinitions.h"

#ifndef ATOM_TYPE_H
#define ATOM_TYPE_H

template <typename DataType, Dimension Dim> struct Sphere {
  std::array<DataType, Dim> Position;
  DataType radius;
};

template <typename DataType> struct Atom : Sphere<DataType, 3> {
  DataType charge;
};

#endif
