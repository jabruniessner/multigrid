#include "Domain.h"
#include "predefinitions.h"
#include "utils.h"
#include <array>
#include <cstddef>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <utility>

template <typename T> inline T square(T x) { return x * x; }

template <Dimension Dim> struct Sphere {
  std::array<DataType, Dim> Position;
  DataType radius;
};

template <Dimension Dim, Length... strides, typename... Positions,
          Length... directions>
void find_dots_in_sphere_helper(Sphere<Dim> &sphere,
                                domain::Domain<Dim, strides...> &domain,
                                DataType grid_step,
                                std::index_sequence<directions...>,
                                Positions... positions) {

  static_assert(sizeof...(directions) <= Dim,
                "Break condition never satisfied");

  if constexpr (sizeof...(directions) == Dim) {
    domain(positions...) = 1;

    //  } else if constexpr (sizeof...(directions) == 0) {
    //
    //    const int lower_bound =
    //        static_cast<std::size_t>(sphere.Position[0] - sphere.radius) + 1;
    //    const int upper_bound =
    //        static_cast<std::size_t>(sphere.Position[0] + sphere.radius);
    //
    //    for (int i = lower_bound; i <= upper_bound; ++i) {
    //      find_dots_in_sphere_helper(sphere, domain,
    //                                 std::index_sequence<directions...>{}, i);
    //    }

  } else {
    const auto &Position = sphere.Position;
    const auto &current_value = Position[sizeof...(directions)];
    const auto &radius = sphere.radius;
    const std::size_t lower_bound =
        static_cast<int>(
            (current_value -
             std::sqrt(square(radius) -
                       (square(Position[directions] - positions) + ...)) /
                 grid_step)) +
        1;
    const std::size_t upper_bound = static_cast<int>(
        (current_value +
         std::sqrt(square(radius) -
                   (square(Position[directions] - positions) + ...))) /
        grid_step);

    for (int i = lower_bound; i <= upper_bound; i++) {
      find_dots_in_sphere_helper(
          sphere, domain, std::make_index_sequence<sizeof...(directions) + 1>{},
          positions..., i);
    }
  }
}

int main() {

  sycl::gpu_selector selector;
  sycl::queue q(selector);

  constexpr std::size_t Dim = 3;

  domain::Domain<Dim, 10, 10, 10> domain(Paddings::PERIODIC, q, 1);

  std::cout << "Hello World!" << std::endl;
  std::random_device rd;

  std::mt19937 gen(rd());
  double lower_bound = 0.0;
  double upper_bound = 1.0;

  auto *Spheres = new Sphere<3u>;
  Spheres->Position[0] = 50;
  Spheres->Position[1] = 50;
  Spheres->Position[2] = 50;
  Spheres->radius = 3.0;
  // Sphere->charge = 0.5;
}
