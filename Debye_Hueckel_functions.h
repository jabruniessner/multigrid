#include "scientific_quantities.h"
#include <cmath>
#include <numbers>

#ifndef DEBYE_HUCKEL_FUNCTIONS_H
#define DEBYE_HUCKEL_FUNCTIONS_H

constexpr double pi = std::numbers::pi;

double DH_Sphere(double radius, double charge, double distance, double kappa) {
  return charge / (4 * pi * (epsilon * epsilon_r) * (1 + kappa * radius)) *
         std::exp(-kappa * (distance - radius)) / distance;
}

#endif // !DEBYE_HUCKEL_FUNCTIONS_H
