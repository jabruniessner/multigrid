#include "iterate_tets.h"
#include <cstdint>

int main() {
  std::cout << "2D:" << std::endl;
  iterate_over_tets<2, std::uint8_t>();
  std::cout << "3D:" << std::endl;
  iterate_over_tets<3, std::uint8_t>();
}
