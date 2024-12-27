#include "Convolution.h"
#include "MultigridDomain.h"
#include "cycles.h"
#include "level_transition.h"
#include <string>
// #include <utility>

using namespace cycles;
using namespace convolution;

template <Dimension Dim, std::size_t base_length, std::size_t nlev,
          std::size_t level = nlev>
void Initializing_all_rhs(
    Multigrid_domain<Dim, base_length, nlev> &MultDomain) {
  if constexpr (level == 1) {
    return;
  } else {

    constexpr std::array<OffsetType, 1> offsets_coarsening{{{0, 0}}};
    constexpr std::array<DataType, 1> values_coarsening{1};

    coarsening(MultDomain.template get_domain<level - 1>(),
               MultDomain.template get_domain<level>(), values_coarsening,
               offsets_coarsening);
    Initializing_all_rhs<Dim, base_length, nlev, level - 1>(MultDomain);
  }
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cout << "Usage: ./this_program num_iters" << std::endl;
    return 0;
  }
  int num_iter = std::stoi(argv[1]);

#ifdef DEBUGMODE
  sycl::cpu_selector selector;
#else
  sycl::gpu_selector selector;
#endif

  sycl::queue q(selector,
                sycl::property_list{sycl::property::queue::in_order{}});

  constexpr std::size_t nlev = 2u;

  Multigrid_domain<2, 1, nlev> lhs_domain1(q);
  Multigrid_domain<2, 1, nlev> lhs_domain2(q);
  Multigrid_domain<2, 1, nlev> rhs_domain(q);
  Multigrid_domain<2, 1, nlev> boundary_values(q);

  constexpr auto length = Multigrid_domain<2, 1, nlev>::length;

  for (int i = 0; i < length + 2; i++) {
    boundary_values.set_value(0., 0, i);
    boundary_values.set_value(1., length + 1, i);
    boundary_values.set_value((double)i / (double)(length + 1), i, 0);
    boundary_values.set_value((double)i / (double)(length + 1), i, length + 1);
  }

  print_multigrid_domain(boundary_values);

  std::array<OffsetType, 5> offsets_op{
      {{-1, 0}, {1, 0}, {0, 0}, {0, -1}, {0, 1}}};
  std::array<DataType, 5> values_op{
      -1. / 4., -1. / 4., 1, -1. / 4.,
      -1. / 4.}; // Dividing the original operator by the Diagonal
                 // as it is only applied to the right hand side anyways

  Multi_Level_operator diff_operator(Integer<nlev>{}, values_op, offsets_op);

  std::array<OffsetType, 4u> offsets{
      {{-1, 0}, {1, 0}, {0, 1}, {0, -1}}}; // Smoothing operator
  std::array<DataType, 4u> values{-1. / 4., -1. / 4., -1. / 4.,
                                  -1. / 4.}; // Formula S = 1 - D^(-1) L,

  Multi_Level_operator mult_level(Integer<nlev>{}, values, offsets);

  std::array<OffsetType, 1u> offsets_coarse{
      {{0, 0}}}; // Coarsening operator single poit for now
  std::array<DataType, 1u> values_coarse{1};

  Multi_Level_operator coarser(Integer<nlev>{}, values_coarse, offsets_coarse);

  Convolve(rhs_domain.domain, boundary_values.domain, values_op, offsets_op);

  Jacobi_Smoother j_smoother(Integer<3>{}, rhs_domain, values, offsets);

  cg_solver::Solver_CG solver(Integer<1>{}, rhs_domain.template get_domain<1>(),
                              values_op, offsets_op);

  mult_level.print_operator();

  V_Cycle_base v_cycle(j_smoother, j_smoother, solver, lhs_domain1, mult_level,
                       diff_operator, coarser);

  auto *current = &lhs_domain1;
  auto *next = &lhs_domain2;

  Domain<2, length, length> helper(Paddings::PERIODIC, q, 1);

  for (int num = 0; num < num_iter; num++) {
    v_cycle.iteration(*current, *next, rhs_domain, mult_level, diff_operator,
                      coarser);

    DataType const residual = compute_residual(
        rhs_domain.template get_domain<nlev>(),
        current->template get_domain<nlev>(), helper, values_op, offsets_op);

    std::cout << "The residual after " << num + 1 << " iterations is "
              << residual << std::endl;
    // std::swap(current, next);
  }

  //  std::cout << "lhs_domain_1:" << std::endl;
  //  print_multigrid_domain(lhs_domain1);
  //
  //  std::cout << "lhs_domain_2:" << std::endl;
  //  print_multigrid_domain(lhs_domain2);
  //
  //  std::cout << "rhs_domain:" << std::endl;
  //  print_multigrid_domain(rhs_domain);
}
