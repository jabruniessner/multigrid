#include "predefinitions.h"
#include <cmath>

#ifndef SCIENTFIC_CONSTANTS_AND_FUNCTIONS_H
#define SCIENTFIC_CONSTANTS_AND_FUNCTIONS_H

constexpr double k_b = 1.380649e-23;
constexpr double e = 1.60217663e-19;
constexpr double T = 298.15;
constexpr double beta = 1 / (k_b * T);
constexpr double potential_unit = k_b * T / e;
constexpr double epsilon = 8.8541878128e-12;
constexpr double epsilon_r = 78.5;
constexpr double epsilon_p = 4.0;
constexpr double n_a = 6.0221408e23;

constexpr double Debye_length_inverse_squared(double IS) {
  return e * e * beta * 2 * IS * n_a / (epsilon_r * epsilon);
}

constexpr double Debye_length(double IS) {
  return (1 / std::sqrt(Debye_length_inverse_squared(1000 * IS))) * 1e10;
};

constexpr double Kappa(double IS) {
  return std::sqrt(Debye_length_inverse_squared(1000 * IS));
}

constexpr double KappaA(double IS) { return Kappa(IS) * 1e-10; }

#endif // !SCIENTFIC_CONSTANTS_AND_FUNCTIONS_H
