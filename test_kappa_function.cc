#include "scientific_quantities.h"
#include <iostream>

int main(int argc, char *argv[]) {

  constexpr double ionic_strength = 0.005;
  constexpr double dl = Debye_length(ionic_strength);

  std::cout << "The debye_length is: " << dl << std::endl;

  std::cout << "The electric permittivity is in units of k_b T is: " << epsilon
            << std::endl;
  return 0;
}
