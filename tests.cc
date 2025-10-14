// #include "CG_Solver.h"
#include "Convolution.h"
#include "hipSYCL/pcuda/pcuda_runtime.hpp"

int main() {

  using namespace convolution;
  using namespace domain;

  Domain<2, 3u, 3u> domain_src(Paddings::PERIODIC, 1);
  Domain<2, 3u, 3u> domain_dest(Paddings::PERIODIC, 1);

  //===== First Trial ======//

  std::cout << "The input matrix is given by: " << std::endl;
  for (Position1D i = 1; i < 3 + 1; i++) {
    for (Position1D j = 1; j < 3 + 1; j++) {
      // domain_src.values_buff[flatten_index<3u, 3u>(1, i, j)] = 1;
      domain_src(i, j) = 1;
    }

    std::cout << std::endl;
  }

  pcudaDeviceSynchronize();

  domain_src(1, 1) = 1;

  for (Position1D i = 0; i < 3 + 2; i++) {
    for (Position1D j = 0; j < 3 + 2; j++) {
      // domain_src.set_value(1,  i, j);
      std::cout << domain_src(i, j) << " ";
    }

    std::cout << std::endl;
  }

  std::cout << "The norm squared is given by: " << std::endl;
  DataType result = 1;

  domain_compute_norm_squared(result, domain_src);

  std::cout << "The result is given by: " << result << std::endl;

  Convolve(domain_dest, domain_src, vec_val, vec_offsets);
  pcudaDeviceSynchronize();

  std::cout << "The output Matrix is given by: " << std::endl;

  for (Position1D i = 1; i < 3 + 1; i++) {
    for (Position1D j = 1; j < 3 + 1; j++) {
      std::cout << domain_dest(i, j) << " ";
    }
    std::cout << std::endl;
  }

  //===== Second trial ======//

  std::cout << "The input matrix is given by: " << std::endl;
  for (Position1D i = 0; i < 3 + 2; i++) {
    domain_src(i, 4u) = 1;
    domain_src(i, 0u) = 1;

    domain_src(4u, i) = 1;
    domain_src(0u, i) = 1;
  }

  std::cout << "(Boundary set to 1)" << std::endl;
  for (Position1D i = 1; i < 3 + 1; i++) {
    for (Position1D j = 1; j < 3 + 1; j++) {
      domain_src(i, j) = 0;
      std::cout << domain_src(i, j) << " ";
    }
    std::cout << std::endl;
  }

  Convolve(domain_dest, domain_src, vec_val, vec_offsets);
  pcudaDeviceSynchronize();

  std::cout << "The output Matrix is given by: " << std::endl;

  for (Position1D i = 1; i < 3 + 1; i++) {
    for (Position1D j = 1; j < 3 + 1; j++) {
      std::cout << domain_dest(i, j) << " ";
    }
    std::cout << std::endl;
  }

  //======= Trial scalar multiplication =======
  std::cout << "The initial matrix is: " << std::endl;
  DataType count = 0;
  for (Position1D i = 1; i < 3 + 1; i++) {
    for (Position1D j = 1; j < 3 + 1; j++) {
      domain_src(i, j) = ++count;
      std::cout << count << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "The result is: " << std::endl;

  std::cout << "Scalar multiplication with factor 3" << std::endl;

  domain_scalar_multiply(domain_dest, domain_src, 3);
  pcudaDeviceSynchronize();

  for (Position1D i = 1; i < 3 + 1; i++) {
    for (Position1D j = 1; j < 3 + 1; j++)
      std::cout << domain_dest(i, j) << " ";

    std::cout << std::endl;
  }

  //===== Trial Addition ======/

  std::cout << "Component-wise addition Domain with itself" << std::endl;
  add_domains(domain_dest, domain_src, domain_src);
  pcudaDeviceSynchronize();

  for (Position1D i = 1; i < 3 + 1; i++) {
    for (Position1D j = 1; j < 3 + 1; j++)
      std::cout << domain_dest(i, j) << " ";

    std::cout << std::endl;
  }

  //===== Trial Subtraction ======/

  std::cout << "Component-wise subtraction Domain with itself" << std::endl;
  subtract_domains(domain_dest, domain_src, domain_src);
  pcudaDeviceSynchronize();

  for (Position1D i = 1; i < 3 + 1; i++) {
    for (Position1D j = 1; j < 3 + 1; j++)
      std::cout << domain_dest(i, j) << " ";

    std::cout << std::endl;
  }

  //===== Trial Multiplication ======/

  std::cout << "Component-wise multiplication Domain with itself" << std::endl;
  multiply_domains(domain_dest, domain_src, domain_src);
  pcudaDeviceSynchronize();

  for (Position1D i = 1; i < 3 + 1; i++) {
    for (Position1D j = 1; j < 3 + 1; j++)
      std::cout << domain_dest(i, j) << " ";

    std::cout << std::endl;
  }

  //===== Trial Division ======/
  std::cout << "Component-wise division Domain with itself" << std::endl;
  divide_domains(domain_dest, domain_src, domain_src);
  pcudaDeviceSynchronize();

  for (Position1D i = 1; i < 3 + 1; i++) {
    for (Position1D j = 1; j < 3 + 1; j++)
      std::cout << domain_dest(i, j) << " ";

    std::cout << std::endl;
  }
}
