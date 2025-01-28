#include "Debye_Hueckel_functions.h"
#include "scientific_quantities.h"
#include <cmath>

double DH_Sphere(double radius, double charge, double distance, double kappa) {
  return charge / (epsilon * epsilon_r * (1 + kappa * radius)) *
         std::exp(-kappa * (distance - radius)) / radius;
}
