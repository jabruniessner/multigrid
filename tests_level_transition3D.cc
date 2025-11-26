#include "CG_Solver.h"
#include "Convolution.h"
#include "level_transition.h"
#include <chrono>

int main() {
  using namespace level_transition;
  using OffsetType = std::array<int, 3>;

  constexpr std::size_t num_coarse = 256;
  constexpr std::size_t num_fine = 2 * num_coarse + 1;

  Domain<3, num_coarse, num_coarse, num_coarse> domain_dest(Paddings::PERIODIC,
                                                            1);
  Domain<3, num_fine, num_fine, num_fine> domain_src(Paddings::PERIODIC, 1);

  std::array<DataType, 27> values;
  std::fill(values.begin(), values.end(), 1 / (27.));

  std::array<OffsetType, 27> offsets;
  for (int i = -1; i < 2; i++)
    for (int k = -1; k < 2; k++)
      for (int j = -1; j < 2; j++)
        offsets[(i + 1) * 9 + (k + 1) * 3 + j + 1] = {i, k, j};

  std::cout << "The input matrix is given by: " << std::endl;
  for (Position1D i = 1; i < num_fine + 1; i++) {
    for (Position1D j = 1; j < num_fine + 1; j++) {
      for (Position1D k = 1; k < num_fine + 1; k++) {
        domain_src(k, j, i) = ((i + j + k) % 2 == 0) * 1;
        //  std::cout << domain_src(k, j, i) << " ";
      }
      //  std::cout << std::endl;
    }
    //  std::cout << std::endl;
    //  std::cout << std::endl;
  }

  coarsening(domain_dest, domain_src, values, offsets);

  //  std::cout << "The output is given by: " << std::endl;
  //  for (Position1D i = 1; i < num_coarse + 1; i++) {
  //    for (Position1D j = 1; j < num_coarse + 1; j++) {
  //      for (Position1D k = 1; k < num_coarse + 1; k++) {
  //        std::cout << domain_dest(k, j, i) << " ";
  //      }
  //      std::cout << std::endl;
  //    }
  //    std::cout << std::endl;
  //    std::cout << std::endl;
  //  }

  auto start = std::chrono::high_resolution_clock::now();
  refinement(domain_src, domain_dest);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double>(end - start);

  std::cout << "The required time was: " << duration.count() << std::endl;

  //  for (Position1D i = 0; i < 7 + 2; i++) {
  //    for (Position1D j = 0; j < 7 + 2; j++) {
  //      for (Position1D k = 0; k < 7 + 2; k++)
  //        std::cout << domain_src(k, j, i) << " ";
  //      std::cout << std::endl;
  //    }
  //    std::cout << std::endl;
  //    std::cout << std::endl;
  //  }
}
