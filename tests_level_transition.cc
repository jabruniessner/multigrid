#include "CG_Solver.h"
#include "Convolution.h"
#include "level_transition.h"

int main() {

#ifdef __ACPP__
  hipsycl::stdpar::unified_shared_memory::pop_disabled();
#endif

  using namespace level_transition;

  Domain<2, 3u, 3u> domain_dest(Paddings::PERIODIC, 1);
  Domain<2, 7u, 7u> domain_src(Paddings::PERIODIC, 1);

  printf("At the very beginning the coarse domain has a pointer: %p\n",
         domain_dest.values_buff);

  printf("At the very beginning the fine domain has a pointer: %p\n",
         domain_src.values_buff);

  constexpr std::array<DataType, 9> values{
      1. / 9, 1. / 9, 1. / 9., 1. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9};

  constexpr std::array<OffsetType, 9> offsets{{{0, 0},
                                               {1, 0},
                                               {-1, 0},
                                               {0, 1},
                                               {0, -1},
                                               {1, 1},
                                               {-1, 1},
                                               {1, -1},
                                               {-1, -1}}};

  //====== First Trial ======//
  std::cout << "The input matrix is given by: " << std::endl;
  for (Position1D i = 1; i < 7 + 1; i++) {
    for (Position1D j = 1; j < 7 + 1; j++) {
      domain_src(i, j) = ((i + j) % 2 == 0) * 1;
    }
  }

  for (Position1D i = 0; i < 7 + 2; i++) {
    for (Position1D j = 0; j < 7 + 2; j++) {
      // domain_src.set_value(1,  i, j);
      std::cout << domain_src(i, j) << " ";
    }

    std::cout << std::endl;
  }

  coarsening(domain_dest, domain_src, values, offsets);

  std::cout << std::endl;

  for (Position1D i = 0; i < 3 + 2; i++) {
    for (Position1D j = 0; j < 3 + 2; j++) {
      // domain_src.set_value(1,  i, j);
      std::cout << domain_dest(i, j) << " ";
    }

    std::cout << std::endl;
  }

  refinement(domain_src, domain_dest);

  std::cout << "The output after the refinement is:" << std::endl;

  for (Position1D i = 0; i < 7 + 2; i++) {
    for (Position1D j = 0; j < 7 + 2; j++) {
      // domain_src.set_value(1,  i, j);
      std::cout << domain_src(i, j) << " ";
    }

    std::cout << std::endl;
  }
}
