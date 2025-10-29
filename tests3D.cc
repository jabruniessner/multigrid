#include "CG_Solver.h"
#include "Convolution.h"
#include "hipSYCL/cuda/cuda_runtime.hpp"
#include <iostream>

int main() {

  using OffsetType = std::array<int, 3>;

  using namespace convolution;
  using namespace domain;

  Domain<3, 3u, 3u, 3u> domain_src(Paddings::PERIODIC, 1);
  Domain<3, 3u, 3u, 3u> domain_dest(Paddings::PERIODIC, 1);

  constexpr std::array<DataType, 7> values = {6, -1, -1, -1, -1, -1, -1};
  constexpr std::array<OffsetType, 7> offsets{{{0, 0, 0},
                                               {1, 0, 0},
                                               {-1, 0, 0},
                                               {0, 1, 0},
                                               {0, -1, 0},
                                               {0, 0, 1},
                                               {0, 0, -1}}};

  //====== First Trial ====//

  std::cout << "The input matrix is given by: " << std::endl;
  for (Position1D i = 1; i < 3 + 1; i++) {
    for (Position1D j = 1; j < 3 + 1; j++) {
      for (Position1D k = 1; k < 3 + 1; k++) {
        domain_src(i, j, k) = 1;
      }
    }
  }

  for (Position1D k = 0; k < 3 + 2; k++) {
    for (Position1D j = 0; j < 3 + 2; j++) {
      for (Position1D i = 0; i < 3 + 2; i++) {
        std::cout << domain_src(i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  std::cout << "The norm squared is given by: " << std::endl;
  DataType result = 1;

  domain_compute_norm_squared(result, domain_src);
  cudaDeviceSynchronize();

  std::cout << "The result is given by: " << result << std::endl;

  Convolve(domain_dest, domain_src, values, offsets);
  cudaDeviceSynchronize();

  std::cout << "The output Matrix is given by: " << std::endl;

  for (Position1D k = 0; k < 3 + 2; k++) {
    for (Position1D j = 0; j < 3 + 2; j++) {
      for (Position1D i = 0; i < 3 + 2; i++) {
        std::cout << domain_dest(i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  //======== Second trial ========//

  std::cout << "The input matrix is given by: " << std::endl;
  for (Position1D j = 0; j < 3 + 2; j++)
    for (Position1D i = 0; i < 3 + 2; i++) {
      domain_src(i, j, 4u) = 1;
      domain_src(i, j, 0u) = 1;

      domain_src(i, 4u, j) = 1;
      domain_src(i, 0u, j) = 1;

      domain_src(4u, i, j) = 1;
      domain_src(0u, i, j) = 1;
    }

  std::cout << "(Boundary set to 1)" << std::endl;

  for (Position1D k = 1; k < 3 + 1; k++) {
    for (Position1D j = 1; j < 3 + 1; j++) {
      for (Position1D i = 1; i < 3 + 1; i++) {
        std::cout << domain_src(i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  Convolve(domain_dest, domain_src, values, offsets);
  cudaDeviceSynchronize();

  std::cout << "The output matrix is given by: " << std::endl;

  for (Position1D k = 1; k < 3 + 1; k++) {
    for (Position1D j = 1; j < 3 + 1; j++) {
      for (Position1D i = 1; i < 3 + 1; i++) {
        std::cout << domain_dest(i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  //============== Trial scalar multiplication ===============
  std::cout << "The initial matrix is: " << std::endl;
  DataType count = 0;

  for (Position1D k = 1; k < 3 + 1; k++) {
    for (Position1D j = 1; j < 3 + 1; j++) {
      for (Position1D i = 1; i < 3 + 1; i++) {
        domain_src(i, j, k) = ++count;
        std::cout << count << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  std::cout << "The result is: " << std::endl;

  std::cout << "Scalar multiplication with factor 3" << std::endl;

  domain_scalar_multiply(domain_dest, domain_src, 3);
  cudaDeviceSynchronize();

  for (Position1D k = 1; k < 3 + 1; k++) {
    for (Position1D j = 1; j < 3 + 1; j++) {
      for (Position1D i = 1; i < 3 + 1; i++) {
        std::cout << domain_dest(i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  //======== Trial Addition ======//

  std::cout << "Component-wise addition Domain with itself" << std::endl;
  add_domains(domain_dest, domain_src, domain_src);
  cudaDeviceSynchronize();

  for (Position1D k = 1; k < 3 + 1; k++) {
    for (Position1D j = 1; j < 3 + 1; j++) {
      for (Position1D i = 1; i < 3 + 1; i++) {
        std::cout << domain_dest(i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  //======= Trial Subtraction =======/

  std::cout << "Component-wise subtraction Domain with itself" << std::endl;
  subtract_domains(domain_dest, domain_src, domain_src);
  cudaDeviceSynchronize();

  for (Position1D k = 1; k < 3 + 1; k++) {
    for (Position1D j = 1; j < 3 + 1; j++) {
      for (Position1D i = 1; i < 3 + 1; i++) {
        std::cout << domain_dest(i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  //===== Trial Multiplication ======/

  std::cout << "Component-wise multiplication Domain with itself" << std::endl;
  multiply_domains(domain_dest, domain_src, domain_src);
  cudaDeviceSynchronize();

  for (Position1D k = 1; k < 3 + 1; k++) {
    for (Position1D j = 1; j < 3 + 1; j++) {
      for (Position1D i = 1; i < 3 + 1; i++) {
        std::cout << domain_dest(i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  //===== Trial Division ======/
  std::cout << "Component-wise division Domain with itself" << std::endl;
  divide_domains(domain_dest, domain_src, domain_src);
  cudaDeviceSynchronize();

  for (Position1D k = 1; k < 3 + 1; k++) {
    for (Position1D j = 1; j < 3 + 1; j++) {
      for (Position1D i = 1; i < 3 + 1; i++) {
        std::cout << domain_dest(i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}
