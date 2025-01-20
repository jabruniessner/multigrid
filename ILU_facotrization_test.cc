#include "FullILU.h"
#include <array>
#include <chrono>
#include <iostream>
#include <random>

// using DataType = double;
constexpr std::size_t problem_size = 4;

int main(int argc, char const *argv[]) {
  std::cout << "Hello World!" << std::endl;
  std::random_device rd;
  std::mt19937 gen(rd());
  double lower_bound = 0.0;
  double upper_bound = 1.0;
  std::uniform_real_distribution<double> dist(lower_bound, upper_bound);

  std::array<DataType, problem_size * problem_size>
      matrix; //{
              // -1. / 2., 9, -2, 1, -3. / 2., 30, -12, 0, 1, -15, 0, -4, 0, -6,
              // 18, 8};
  // generate random matrix
  for (int i = 0; i < problem_size * problem_size; ++i) {
    matrix[i] = dist(gen);
  }

  std::array<DataType, problem_size> vector;
  std::array<DataType, problem_size> result, difference;
  for (int i = 0; i < problem_size; i++) {
    vector[i] = dist(gen);
  }

  matrix_vector_multiply(matrix, vector, result);

  std::cout << "The vector is: " << std::endl;
  for (int i = 0; i < problem_size; i++) {
    std::cout << vector[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "The result of the matrix vector multiplication is: "
            << std::endl;

  for (int i = 0; i < problem_size; i++) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;

  Factorize_ILU<problem_size>(matrix);
  solve_ILU<problem_size>(matrix, result);

  std::cout << "The result of the ILU factorization is: " << std::endl;
  for (int i = 0; i < problem_size; i++) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;

  vector_subtract(difference, result, vector);

  DataType norm;
  vector_norm(difference, norm);

  std::cout << "The norm of the difference is: " << norm << std::endl;

  return 0;
}
