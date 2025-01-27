#include "scientific_quantities.h"
#include <iostream>

int main(int argc, char *argv[]) {

  constexpr double ionic_strength = 0.005;

  std::cout << "The ionic strength is: " << Debye_length(ionic_strength)
            << std::endl;
  return 0;
}
