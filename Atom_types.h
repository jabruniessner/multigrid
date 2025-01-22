#include "predefinitions.h"

#ifndef ATOM_TYPE_H
#define ATOM_TYPE_H

template <typename DataType, Dimension Dim> struct Sphere {
  std::array<DataType, Dim> Position;
  DataType radius;
};

template <typename DataType> struct Atom : Sphere<DataType, 3> {
  DataType charge;

  void print() {
    std::cout << "x: " << this->Position[0] << " y: " << this->Position[1]
              << " z: " << this->Position[2] << " r: " << this->radius
              << " c:  " << this->charge << std::endl;
  }
};

#endif
