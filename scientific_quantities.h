#include "predefinitions.h"
#include <cmath>

#ifndef SCIENTFIC_CONSTANTS_AND_FUNCTIONS_H
#define SCIENTFIC_CONSTANTS_AND_FUNCTIONS_H

constexpr double k_b = 1.380649e-23;
constexpr double e = 1.60217663e-19;
constexpr double T = 298.15;
constexpr double beta = 1 / (k_b * T);
constexpr double potential_unit = k_b * T / e;
constexpr double epsilon_SI = 8.8541878128e-12;
constexpr double epsilon = epsilon_SI * 1e-10 / (e * e) * k_b * T;
constexpr DataType epsilon_r = 78.4;
constexpr DataType epsilon_p = 4.0; // 4.0;
constexpr double n_a = 6.0221408e23;

constexpr double const_sqrt(double x, double guess = 1.0) {
  return std::abs(guess * guess - x) < x * 1e-13
             ? guess
             : const_sqrt(x, (guess + x / guess) / 2);
}

constexpr double sqrt2 = const_sqrt(2, 1.414);

inline double harmonic_average(double x, double y) {
  return 2 * x * y / (x + y);
}

template <typename... Elems> inline double arithmetic_average(Elems... elems) {
  return (elems + ...) / (sizeof...(Elems));
}

// This function is made to work in SI units, so it takes the IS in units of
// Mole/m^3
constexpr double Debye_length_inverse_squared(double IS) {
  return e * e * beta * 2 * IS * n_a / (epsilon_r * epsilon_SI);
}

// This funnctions automatically converts from Mole/ltr to Mole/m^3 and return
// the Debye Length in A
constexpr double Debye_length(double IS) {
  return (1 / const_sqrt(Debye_length_inverse_squared(1000 * IS))) * 1e10;
};

// This function computes Kappa, (the inverse Debye length) in units of 1/m
constexpr double Kappa(double IS) {
  return const_sqrt(Debye_length_inverse_squared(1000 * IS));
}

// This function computes Kappa, (the inverse Debye length) in units of 1/A
constexpr double KappaA(double IS) { return Kappa(IS) * 1e-10; }

constexpr double KappaA2(double IS) { return KappaA(IS) * KappaA(IS); }

template <typename Num_Type>
inline DataType get_local_epsilon(Num_Type weight1, Num_Type weight2) {
  return (weight1 * epsilon_r + weight2 * (epsilon_p)) / (weight1 + weight2);
}

#endif // !SCIENTFIC_CONSTANTS_AND_FUNCTIONS_H
