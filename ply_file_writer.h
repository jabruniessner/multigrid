#pragma once

#include <boost/container/static_vector.hpp>
#include <iostream>
#include <list>

#include "blas.h"

namespace ply {

using vector3d = blas::vector<DataType, 3>;
using Face = boost::container::static_vector<vector3d, 3>;

// template <template <typename> typename container>
template <typename Container>
void print_faces_to_ply(std::ostream &stream, Container &faces) {
  stream << "ply\n";
  stream << "format ascii 1.0\n";
  stream << "element vertex " << faces.size() * 3 << "\n";
  stream << "property float x\n";
  stream << "property float y\n";
  stream << "property float z\n";
  stream << "element face " << faces.size() << "\n";
  stream << "property list uchar int vertex_index\n";
  stream << "end_header\n";

  for (auto face : faces) {
    for (auto point : face) {
      stream << point[0] << " " << point[1] << " " << point[2] << "\n";
    }
  }

  int index = 0;
  for (int i = 0; i < faces.size(); i++) {
    stream << "3 " << index << " " << index + 1 << " " << index + 2 << "\n";
    index += 3;
  }

  stream << std::endl;
}

} // namespace ply
