#include "Convolution.h"
#include "MultigridDomain.h"
#include "cycles.h"
#include "level_transition.h"
#include <chrono>
#include <string>
#include <utility>
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
  constexpr std::size_t base_length = 22u;
  constexpr DataType omega = 4. / 5.;

  Multigrid_domain<2, base_length, nlev> lhs_domain1(q);
  Multigrid_domain<2, base_length, nlev> lhs_domain2(q);
  Multigrid_domain<2, base_length, nlev> rhs_domain(q);
  Multigrid_domain<2, base_length, nlev> boundary_values(q);

  constexpr auto length = Multigrid_domain<2, base_length, nlev>::length;

  std::cout << "The length is: " << length << std::endl;

  // std::cout << "After initialization we get:" << std::endl;
  // print_multigrid_domain(lhs_domain1);

  for (int i = 0; i < length + 2; i++) {
    boundary_values.set_value(0., 0, i);
    boundary_values.set_value(1., length + 1, i);
    boundary_values.set_value((double)i / (double)(length + 1), i, 0);
    boundary_values.set_value((double)i / (double)(length + 1), i, length + 1);
  }

  // print_multigrid_domain(boundary_values);

  std::array<OffsetType, 5> offsets_op{
      {{-1, 0}, {1, 0}, {0, 0}, {0, -1}, {0, 1}}};
  std::array<DataType, 5> values_op{
      -1., -1, 4, -1.,
      -1.}; // Dividing the original operator by the Diagonal
            // as it is only applied to the right hand side anyways

  Multi_Level_operator diff_operator(Integer<nlev>{}, values_op, offsets_op, 1.,
                                     Integer<base_length>{});

  std::array<OffsetType, 5u> offsets{
      {{-1, 0}, {1, 0}, {0, 1}, {0, -1}, {0, 0}}}; // Smoothing operator
  std::array<DataType, 5u> values{-omega * 1. / 4., -omega * 1. / 4.,
                                  -omega * 1. / 4., -omega * 1. / 4.,
                                  -1. + omega}; // Formula S = 1 - D^(-1) L,

  Multi_Level_operator mult_level(Integer<nlev>{}, values, offsets,
                                  Integer<base_length>{});

  std::array<OffsetType, 9u> offsets_coarse{
      {{-1, -1},
       {0, -1},
       {1, -1},
       {-1, 0},
       {0, 0},
       {1, 0},
       {-1, 1},
       {0, 1},
       {1, 1}}}; // Coarsening operator single point for now
  std::array<DataType, 9u> values_coarse{1. / 16., 2. / 16., 1. / 16.,
                                         2. / 16., 4. / 16., 2. / 16.,
                                         1. / 16., 2. / 16., 1. / 16.};

  Multi_Level_operator coarser(Integer<nlev>{}, values_coarse, offsets_coarse,
                               Integer<base_length>{});

  Convolve(rhs_domain.domain, boundary_values.domain,
           diff_operator.get_values(), diff_operator.get_offsets());

  // rhs_domain.domain.print_domain();

  std::index_sequence<5> smoother_sequence{};
  Jacobi_Smoother j_smoother(smoother_sequence, rhs_domain, values, offsets);

  cg_solver::Solver_CG solver(Float<1e-9>{},
                              rhs_domain.template get_domain<1>(),
                              diff_operator.template get_values<1>(),
                              diff_operator.template get_offsets<1>());

  //  // mult_level.print_operator();
  std::index_sequence<1> num_iters_level{};
  V_Cycle_base v_cycle(j_smoother, j_smoother, solver, lhs_domain1, mult_level,
                       diff_operator, coarser, num_iters_level);

  auto *current = &lhs_domain1;
  auto *next = &lhs_domain2;

  Domain<2, length, length> helper(Paddings::PERIODIC, q, 1);

  auto start = std::chrono::high_resolution_clock::now();
  for (int num = 0; num < num_iter; num++) {
    v_cycle.iteration(*current, *next, rhs_domain, mult_level, diff_operator,
                      coarser, 1., omega);

    DataType const residual = compute_residual(
        rhs_domain.template get_domain<nlev>(),
        current->template get_domain<nlev>(), helper,
        diff_operator.get_values(), diff_operator.get_offsets());

    std::cout << "The residual after " << num + 1 << " iterations is "
              << residual << std::endl;
    // std::swap(current, next);
  }

  // Convolve(helper, current->template get_domain<nlev>(),
  //          diff_operator.get_values(), diff_operator.get_offsets());

  // helper.print_domain();

  // current->template get_domain<nlev>().print_domain();

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cout << "The required time was: " << duration.count() << " seconds"
            << std::endl;

  // current->template get_domain<nlev>().print_to_output(std::cout);

  // std::cout << "lhs_domain_1:" << std::endl;
  // print_multigrid_domain(lhs_domain1);

  // std::cout << "lhs_domain_2:" << std::endl;
  // print_multigrid_domain(lhs_domain2);

  // std::cout << "rhs_domain:" << std::endl;
  // print_multigrid_domain(rhs_domain);
}
