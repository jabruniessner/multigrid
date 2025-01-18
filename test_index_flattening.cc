#include "Domain.h"
#include <cassert>
#include <iostream>

int main() {

  constexpr std::size_t stride_1 = 50;
  constexpr std::size_t stride_2 = 150;
  constexpr std::size_t stride_3 = 300;

  // for (int i = 0; i < stride_1; i++)
  // for (int j = 0; j < stride_2; j++)
  for (int k = 0; k < stride_3; k++) {
    auto flat_i = domain::flatten_index<stride_3>(0, k);
    auto mult_i = domain::flat_to_multi_index<stride_3>(flat_i);

    //   std::cout << "Original: " << i << " " << j << " " << k <<
    //   std::endl; std::cout << "Flat index: " << flat_i << std::endl;
    //   std::cout << "Computed: " << mult_i[0] << " " << mult_i[1] << " "
    //             << mult_i[2] << std::endl;

    assert(k == mult_i[0]);
  }

  //  auto flat_i = domain::flatten_index<stride_1, stride_2, stride_3>(0, 0, 1,
  //  0); std::cout << "The flat index is: " << flat_i << std::endl;
  //
  //  auto mult_i =
  //      domain::flat_to_multi_index<stride_1, stride_2, stride_3>(flat_i);
  //
  //  std::cout << "The multi index is: " << mult_i[0] << " " << mult_i[1] << "
  //  "
  //            << mult_i[2] << std::endl;

  std::cout << "The test was successful!" << std::endl;

  // int index = 1;
  // auto mult_i = domain::flat_to_multi_index<2, 2>(index);
  // auto flat_i = domain::flatten_index<2, 2>(0, mult_i[0], mult_i[1]);

  // std::cout << "The returned flat index is: " << flat_i << std::endl;
}
