#include "bitshift_lib.h"
#include "iterate_tets.h"
#include <array>
#include <cstdint>

template <std::size_t Dim, UnsignedIntegral num_type>
void func(std::array<num_type, Dim + 1> a, std::size_t j) {
  for (auto i : a) {
    std::cout << static_cast<int>(i) << " ";
  }

  std::cout << j << " ";

  std::cout << std::endl;
}

int main() {
  std::cout << "2D:" << std::endl;
  iterate_over_tets<2, std::uint8_t>(func<2, std::uint8_t>);
  std::cout << "3D:" << std::endl;
  iterate_over_tets<3, std::uint8_t>(func<3, std::uint8_t>);
  std::cout << "4D" << std::endl;
  iterate_over_tets<4, std::uint8_t>(func<4, std::uint8_t>);

  for (std::uint8_t i = 0; i < 16; i++) {
    auto vector =
        bitshift::convert_byte_to_vec<std::uint8_t, std::uint8_t, 4>(i);

    for (auto i : vector) {
      std::cout << static_cast<std::size_t>(i) << " ";
    }
    std::cout << std::endl;
  }
}
