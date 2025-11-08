#include "profiling_library.h"
#include "predefinitions.h"
#include <iostream>

namespace profiling {

int stack_depth_counter = 0;

stack_depth::stack_depth() { stack_depth_counter++; }
stack_depth::~stack_depth() { stack_depth_counter--; }

Values::~Values() {

#define PRINT_TIMING(type)                                                     \
  std::cout << #type << " time: " << type                                      \
            << "s: " << (overall_time == 0 ? 0 : type / overall_time * 100)    \
            << "%" << std::endl;

  // #define PRINT_TIMING(type)

  PRINT_TIMING(overall_time);
  PRINT_TIMING(pre_smoothing);
  PRINT_TIMING(defect_computation);
  PRINT_TIMING(restriction);
  PRINT_TIMING(refinement);
  PRINT_TIMING(post_smoothing);
  PRINT_TIMING(residual_computation);
  PRINT_TIMING(coarser_grids);
  PRINT_TIMING(coarse_grid_solver);
}

Values values;

} // namespace profiling
