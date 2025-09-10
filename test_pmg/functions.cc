#include "functions.h"

void initialize_random(DataType *first, DataType *second, int num) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(0.0, 1.0);

  for (int i = 0; i < num; i++)
    first[i] = second[i] = distrib(gen);

  for (int i = 0; i < num; i++) {
    assert(std::abs(first[i] - second[i]) < 1e-12);
  }
}

void compare(DataType *first, DataType *second, int num) {
  for (int i = 0; i < num; i++)
    assert(std::abs(first[i] - second[i]) < 1e-12);
}
