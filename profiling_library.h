#include <iostream>

#ifndef PROFILING_LIBRARY_H
#define PROFILING_LIBRARY_H

namespace profiling {

extern int stack_depth_counter;

struct stack_depth {
  stack_depth();
  ~stack_depth();
};

struct Values {
  double pre_smoothing = 0;
  double refinement = 0;
  double defect_computation = 0;
  double restriction = 0;
  double post_smoothing = 0;
  double residual_computation = 0;
  double overall_time = 0;
  ~Values();
};

extern Values values;

} // namespace profiling

#ifdef PROFILING
#define PROFILE_ADD profiling::stack_depth plus_on;
#else
#define PROFILE_ADD
#endif

#ifdef PROFILING
#define PROFILE_START(type)                                                    \
  auto start_##type = std::chrono::high_resolution_clock::now();
#else
#define PROFILE_START(type)
#endif

#ifdef PROFILING
#define PROFILE_END(type)                                                      \
  if (profiling::stack_depth_counter == 1) {                                   \
    current.get_domain().q.wait();                                             \
    auto end_##type = std::chrono::high_resolution_clock::now();               \
    std::chrono::duration<double> duration = end_##type - start_##type;        \
    profiling::values.type += duration.count();                                \
  }
#else
#define PROFILE_END(type)
#endif

#endif
