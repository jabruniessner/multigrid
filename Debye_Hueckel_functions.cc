#include "Debye_Hueckel_functions.h"
#include "scientific_quantities.h"
#include <cmath>
#include <numbers>

constexpr double pi = std::numbers::pi;

double DH_Sphere(double radius, double charge, double distance, double kappa) {
  return charge / (4 * pi * (epsilon * epsilon_r) * (1 + kappa * radius)) *
         std::exp(-kappa * (distance - radius)) / distance;
}
