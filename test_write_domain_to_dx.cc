#include "Domain.h"
#include <fstream>
#include <iostream>

int main(int argc, char *argv[]) {

  std::cout << "Hello World!" << std::endl;
  std::ofstream dx_file{"dx_trial.dx"};
  domain::Domain<3, 10, 10, 10> domain(Paddings::PERIODIC, 1);

  if (dx_file.is_open()) {
    domain.print_dx_to_stream(dx_file, -1, -1, -1, 11);
  }
  return 0;
}
