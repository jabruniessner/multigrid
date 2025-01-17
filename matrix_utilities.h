#include "utils.h"

using Length = std::size_t;

// Flatten index function
template <Length FirstStride, Length... RestStrides, typename Padding,
          typename Position, typename... PositionRest>
std::size_t flatten_index(Padding padding, Position &i,
                          PositionRest &...rest_positions) {
  std::size_t index = i;
  ((index *= RestStrides + 2 * padding, index += rest_positions), ...);
  return index;
}
