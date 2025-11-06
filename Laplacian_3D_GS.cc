#include "Convolution.h"
#include "MultigridDomain.h"
#include "cycles.h"
#include "level_transition.h"
#include "profiling_library.h"
#include <algorithm>
#include <chrono>
#include <execution>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

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

DataType f(DataType a, DataType b, DataType c, DataType d, DataType e,
           DataType f, DataType g, DataType i, DataType l, DataType m,
           DataType n, DataType h) {
  return (296.0 / 45.0) * std::pow(h, 7) -
         1.0 / 90.0 * std::pow(h, 5) *
             (56 * b - 32 * c + 21 * d - 24 * e + 28 * f - 49 * g) +
         18 * std::pow(h, 4) * std::pow(l, 3) +
         32 * std::pow(h, 4) * std::pow(m, 3) +
         98 * std::pow(h, 4) * std::pow(n, 3) +
         (1.0 / 10.0) * std::pow(h, 3) * std::pow(i, 2) +
         (1.0 / 30.0) * std::pow(h, 3) * i * (3 * a + b + c + d + e + f + g) +
         9 * std::pow(h, 3) * std::pow(l, 4) +
         16 * std::pow(h, 3) * std::pow(m, 4) +
         49 * std::pow(h, 3) * std::pow(n, 4) +
         (1.0 / 60.0) * std::pow(h, 3) *
             (6 * std::pow(a, 2) + 2 * a * b + 2 * a * c + 2 * a * e +
              2 * std::pow(b, 2) + 2 * std::pow(c, 2) + 2 * std::pow(d, 2) +
              d * (2 * a + b + c) + 2 * std::pow(e, 2) + 2 * std::pow(f, 2) +
              f * (2 * a + b + e) + 2 * std::pow(g, 2) + g * (2 * a + c + e)) +
         (1.0 / 2.0) * std::pow(l, 2) *
             (24 * std::pow(h, 5) + 3 * std::pow(h, 3) * i +
              std::pow(h, 3) * (3 * a + b + c + d + e + f + g)) +
         (1.0 / 10.0) * l *
             (30 * std::pow(h, 6) + 18 * std::pow(h, 4) * i +
              std::pow(h, 4) *
                  (12 * a + 3 * b + 3 * c + 2 * d + 8 * e + 7 * f + 7 * g)) +
         (2.0 / 3.0) * std::pow(m, 2) *
             (32 * std::pow(h, 5) + 36 * std::pow(h, 4) * l +
              3 * std::pow(h, 3) * i + 36 * std::pow(h, 3) * std::pow(l, 2) +
              std::pow(h, 3) * (3 * a + b + c + d + e + f + g)) +
         (2.0 / 15.0) * m *
             (40 * std::pow(h, 6) + 180 * std::pow(h, 5) * l +
              18 * std::pow(h, 4) * i + 180 * std::pow(h, 4) * std::pow(l, 2) +
              std::pow(h, 4) *
                  (12 * a + 3 * b + 8 * c + 7 * d + 3 * e + 2 * f + 7 * g)) +
         (7.0 / 6.0) * std::pow(n, 2) *
             (56 * std::pow(h, 5) - 36 * std::pow(h, 4) * l -
              48 * std::pow(h, 4) * m - 3 * std::pow(h, 3) * i -
              36 * std::pow(h, 3) * std::pow(l, 2) -
              48 * std::pow(h, 3) * std::pow(m, 2) -
              std::pow(h, 3) * (3 * a + b + c + d + e + f + g)) +
         (7.0 / 30.0) * n *
             (70 * std::pow(h, 6) - 180 * std::pow(h, 5) * l -
              240 * std::pow(h, 5) * m - 18 * std::pow(h, 4) * i -
              180 * std::pow(h, 4) * std::pow(l, 2) -
              240 * std::pow(h, 4) * std::pow(m, 2) -
              std::pow(h, 4) *
                  (12 * a + 8 * b + 3 * c + 7 * d + 3 * e + 7 * f + 2 * g));
}

template <std::size_t Dim, std::size_t... strides_all>
DataType compute_deviation(domain::Domain<Dim, strides_all...> sol_domain,
                           DataType h) {
  // DataType *result = sycl::malloc_device<DataType>(1, sol_domain.q);

  std::array<int, Dim> length{(strides_all + 1)...};
  iterator start(0);
  iterator end(((strides_all + 1) * ...));

  DataType result = std::transform_reduce(
      std::execution::par_unseq, start, end, 0.0, std::plus<>{}, [=](int idx) {
        auto I = domain::flat_to_multi_index<(strides_all + 1)...>(idx);
        return f(sol_domain(I[0], I[1], I[2]), sol_domain(I[0], I[1], I[2] + 1),
                 sol_domain(I[0], I[1] + 1, I[2]),
                 sol_domain(I[0], I[1] + 1, I[2] + 1),
                 sol_domain(I[0] + 1, I[1], I[2]),
                 sol_domain(I[0] + 1, I[1], I[2] + 1),
                 sol_domain(I[0] + 1, I[1] + 1, I[2]),
                 sol_domain(I[0] + 1, I[1] + 1, I[2] + 1),

                 (DataType)I[0] / (length[0]), (DataType)I[1] / (length[1]),
                 (DataType)I[2] / (length[2]), h);
      });

  return std::sqrt(result);
}

DataType f_grad(DataType a, DataType b, DataType c, DataType d, DataType e,
                DataType f, DataType g, DataType i, DataType l, DataType m,
                DataType n, DataType h) {
  return (296.0 / 3.0) * std::pow(h, 5) + 36 * std::pow(h, 3) * std::pow(l, 2) +
         64 * std::pow(h, 3) * std::pow(m, 2) +
         196 * std::pow(h, 3) * std::pow(n, 2) -
         2.0 / 3.0 * std::pow(h, 3) *
             (14 * b - 8 * c + 3 * d - 6 * e + 4 * f - 7 * g) +
         h * std::pow(i, 2) - 2.0 / 3.0 * h * i * (d + f + g) +
         (1.0 / 3.0) * h *
             (3 * std::pow(a, 2) - 2 * a * b - 2 * a * c - 2 * a * e +
              2 * std::pow(b, 2) + 2 * std::pow(c, 2) + 2 * std::pow(d, 2) -
              d * (b + c) + 2 * std::pow(e, 2) + 2 * std::pow(f, 2) -
              f * (b + e) + 2 * std::pow(g, 2) - g * (c + e)) +
         2 * l *
             (18 * std::pow(h, 4) + 2 * std::pow(h, 2) * i -
              std::pow(h, 2) * (2 * a + b + c + 2 * d - 2 * e - f - g)) +
         (8.0 / 3.0) * m *
             (24 * std::pow(h, 4) + 2 * std::pow(h, 2) * i -
              std::pow(h, 2) * (2 * a + b - 2 * c - d + e + 2 * f - g)) +
         (14.0 / 3.0) * n *
             (42 * std::pow(h, 4) - 2 * std::pow(h, 2) * i +
              std::pow(h, 2) * (2 * a - 2 * b + c - d + e - f + 2 * g));
}

template <std::size_t Dim, std::size_t... strides_all>
DataType compute_energy_norm(domain::Domain<Dim, strides_all...> sol_domain,
                             DataType h) {

  std::array<int, Dim> length{(strides_all + 1)...};

  iterator start(0);
  iterator end(((strides_all + 1) * ...));

  DataType result = std::transform_reduce(
      std::execution::par_unseq, start, end, 0.0, std::plus<>{}, [=](int idx) {
        auto I = domain::flat_to_multi_index<(strides_all + 1)...>(idx);
        return f_grad(
            sol_domain(I[0], I[1], I[2]), sol_domain(I[0], I[1], I[2] + 1),
            sol_domain(I[0], I[1] + 1, I[2]),
            sol_domain(I[0], I[1] + 1, I[2] + 1),
            sol_domain(I[0] + 1, I[1], I[2]),
            sol_domain(I[0] + 1, I[1], I[2] + 1),
            sol_domain(I[0] + 1, I[1] + 1, I[2]),
            sol_domain(I[0] + 1, I[1] + 1, I[2] + 1),

            (DataType)I[0] / (length[0]), (DataType)I[1] / (length[1]),
            (DataType)I[2] / (length[2]), h);
      });

  return std::sqrt(result);
}

int main(int argc, char *argv[]) {
  using OffsetType = std::array<int, 3>;

  if (argc < 2) {
    std::cout << "Usage: ./this_program num_iters" << std::endl;
    return 0;
  }
  int num_iter = std::stoi(argv[1]);

  constexpr std::size_t nlev = 5u;
  constexpr std::size_t base_length = 16u;
  constexpr DataType omega = 1.;
  // constexpr DataType box_length = 12;
  constexpr DataType upper_grid_step = 1;

  constexpr auto &length =
      Multigrid_domain<3, nlev, base_length, base_length, base_length>::length;

  Multigrid_domain<3, nlev, base_length, base_length, base_length>
      lhs_domain1{};
  Multigrid_domain<3, nlev, base_length, base_length, base_length>
      lhs_domain2{};
  Multigrid_domain<3, nlev, base_length, base_length, base_length>
      lhs_domain3{};
  Multigrid_domain<3, nlev, base_length, base_length, base_length> rhs_domain{};
  Multigrid_domain<3, nlev, base_length, base_length, base_length>
      boundary_values{};
  Multigrid_domain<3, nlev, base_length, base_length, base_length>
      defect_domain{};

  Domain<3, std::get<0>(length), std::get<1>(length), std::get<2>(length)>
      u_domain(Paddings::PERIODIC, 1), convolved(Paddings::PERIODIC, 1),
      helper(Paddings::PERIODIC, 1);

  std::cout << "The number of dofs is: " << u_domain.num_dofs << std::endl;

  std::cout << "The length are: " << std::get<0>(length) << " "
            << std::get<1>(length) << " " << std::get<2>(length) << std::endl;

  constexpr DataType h = (DataType)1 / (std::get<0>(length) + 1);

  std::cout << "The value of h is: " << h << std::endl;

  auto boundary_conditions = [=](DataType x, DataType y, DataType z) {
    return -3 * x * x - 4 * y * y + 7 * z * z;
  };

  iterator start(0);
  iterator end(u_domain.num_values);

  std::for_each(std::execution::par_unseq, start, end, [=](int idx) {
    auto I = domain::flat_to_multi_index<std::get<0>(length) + 2,
                                         std::get<1>(length) + 2,
                                         std::get<2>(length) + 2>(idx);
    //  I[0] += 1;
    //  I[1] += 1;
    //  I[2] += 1;
    u_domain(I[0], I[1], I[2]) =
        boundary_conditions((DataType)I[0] / (std::get<0>(length) + 1),
                            (DataType)I[1] / (std::get<1>(length) + 1),
                            (DataType)I[2] / (std::get<2>(length) + 1));
  });

  // std::ofstream u_file("u_file_test.dx");
  // u_domain.print_dx_to_stream(u_file, 0, 0, 0, 1);

  // std::cout << "lhs_domain1: " << std::endl;
  // print_multigrid_domain(lhs_domain1);

  // std::cout << "The length is: " << std::get<0>(length) << std::endl;

  // std::cout << "After initialization we get:" << std::endl;
  // print_multigrid_domain(lhs_domain1);

  {

    iterator start(0);
    iterator end((std::get<1>(length) + 2) * (std::get<2>(length) + 2));

    auto &boundary_domain = lhs_domain1.template get_domain<nlev>();
    std::for_each(std::execution::par_unseq, start, end, [=](int idx) {
      auto I = domain::flat_to_multi_index<(std::get<1>(length) + 2),
                                           (std::get<2>(length) + 2)>(idx);
      boundary_domain(I[0], I[1], 0) =
          boundary_conditions((DataType)I[0] / (std::get<1>(length) + 1),
                              (DataType)I[1] / (std::get<2>(length) + 1), 0);

      boundary_domain(I[0], I[1], std::get<2>(length) + 1) =
          boundary_conditions((DataType)I[0] / (std::get<1>(length) + 1),
                              (DataType)I[1] / (std::get<2>(length) + 1), 1);

      boundary_domain(0, I[0], I[1]) =
          boundary_conditions(0, (DataType)I[0] / (std::get<1>(length) + 1),
                              (DataType)I[1] / (std::get<2>(length) + 1));

      boundary_domain(std::get<0>(length) + 1, I[0], I[1]) =
          boundary_conditions(1, (DataType)I[0] / (std::get<1>(length) + 1),
                              (DataType)I[1] / (std::get<2>(length) + 1));

      boundary_domain(I[0], 0, I[1]) =
          boundary_conditions((DataType)I[0] / (std::get<1>(length) + 1), 0,
                              (DataType)I[1] / (std::get<2>(length) + 1));

      boundary_domain(I[0], std::get<1>(length) + 1, I[1]) =
          boundary_conditions((DataType)I[0] / (std::get<1>(length) + 1), 1,
                              (DataType)I[1] / (std::get<2>(length) + 1));
    });

    //  std::cout << "The boundary domain is: " << std::endl;
    //  boundary_domain.print_domain();
  }

  //  {
  //    auto &boundary_domain = lhs_domain2.template get_domain<nlev>();
  //    q.parallel_for(
  //         sycl::range<2>(std::get<1>(length) + 2, std::get<2>(length) +
  // 2),
  //         [=](sycl::id<2> I) {
  //           boundary_domain(I[0], I[1], 0) = boundary_conditions(
  //               (DataType)I[0] / (std::get<1>(length) + 2),
  //               (DataType)I[1] / (std::get<2>(length) + 2), 0);
  //
  //           boundary_domain(I[0], I[1], std::get<2>(length) + 1) =
  //               boundary_conditions((DataType)I[0] / (std::get<1>(length)
  //+
  //               2),
  //                                   (DataType)I[1] / (std::get<2>(length)
  //+
  //                                   2), 1);
  //
  //           boundary_domain(0, I[0], I[1]) = boundary_conditions(
  //               0, (DataType)I[0] / (std::get<1>(length) + 2),
  //               (DataType)I[1] / (std::get<2>(length) + 2));
  //
  //           boundary_domain(std::get<0>(length) + 1, I[0], I[1]) =
  //               boundary_conditions(1,
  //                                   (DataType)I[0] / (std::get<1>(length)
  //+
  //                                   2), (DataType)I[1] /
  //                                   (std::get<2>(length)
  //                                   + 2));
  //
  //           boundary_domain(I[0], 0, I[1]) = boundary_conditions(
  //               (DataType)I[0] / (std::get<1>(length) + 2), 0,
  //               (DataType)I[1] / (std::get<2>(length) + 2));
  //
  //           boundary_domain(I[0], std::get<1>(length) + 1, I[1]) =
  //               boundary_conditions((DataType)I[0] / (std::get<1>(length)
  //+
  //               2),
  //                                   1,
  //                                   (DataType)I[1] / (std::get<2>(length)
  //+
  //                                   2));
  //         })
  //        .wait();
  //  }
  //
  std::array<OffsetType, 7> offsets_op{{{-1, 0, 0},
                                        {1, 0, 0},
                                        {0, 0, 0},
                                        {0, -1, 0},
                                        {0, 1, 0},
                                        {0, 0, -1},
                                        {0, 0, 1}}};
  std::array<DataType, 7> values_op{
      -1., -1,  6,  -1.,
      -1., -1., -1.}; // Dividing the original operator by the Diagonal
                      // as it is only applied to the right hand side
                      //   anyways
  Multi_Level_operator diff_operator(Integer<nlev>{}, values_op, offsets_op,
                                     upper_grid_step, Integer<base_length>{});

  diff_operator.print_operator();

  // convolution::Convolve(convolved, u_domain, diff_operator.get_values(),
  //                       diff_operator.get_offsets());

  // std::ofstream file("convolved.dx");
  // convolved.print_dx_to_stream(file, 0, 0, 0, 1);

  // Here I am testing that ty u is indeed harmonic

  std::array<OffsetType, 7u> offsets{{{-1, 0, 0},
                                      {1, 0, 0},
                                      {0, 0, 0},
                                      {0, -1, 0},
                                      {0, 1, 0},
                                      {0, 0, -1},
                                      {0, 0, 1}}};
  // Smoothing operator
  std::array<DataType, 7u> values{-omega * 1. / 6., -omega * 1. / 6.,
                                  -1. + omega,      -omega * 1. / 6.,
                                  -omega * 1. / 6., -omega * 1 / 6.,
                                  -omega * 1 / 6.}; // Formula S = 1 - D^(-1) L,

  Multi_Level_operator mult_level(
      Integer<nlev>{}, values, offsets,
      Integer<base_length>{}); // Smoothing diff_operator
                               //
                               //  // mult_level.print_operator();
                               //
  //  //  // //  std::array<OffsetType, 9u> offsets_coarse{
  //  //  // //      {{-1, -1},
  //  //  // //       {0, -1},
  //  //  // //       {1, -1},
  //  //  // //       {-1, 0},
  //  //  // //       {0, 0},
  //  //  // //       {1, 0},
  //  //  // //       {-1, 1},
  //  //  // //       {0, 1},
  //  //  // //       {1, 1}}}; // Coarsening operator single point for now
  //  //  // //  std::array<DataType, 9u> values_coarse{1. / 16., 2. / 16., 1.
  //  //  / 16.,
  //  //  // //                                         2. / 16., 4. / 16., 2.
  //  //  / 16.,
  //  //  // //                                         1. / 16., 2. / 16., 1.
  //  //  / 16.};
  //  //
  //  //  Lambda expression for computing deviation

  std::array<OffsetType, 1u> offsets_coarse{{{0, 0, 0}}};
  std::array<DataType, 1u> values_coarse{1.};

  Multi_Level_operator coarser(Integer<nlev>{}, values_coarse, offsets_coarse,
                               Integer<base_length>{});
  //  // coarser.print_operator();

  //  Convolve(rhs_domain.domain, boundary_values.domain,
  //           diff_operator.get_values(), diff_operator.get_offsets());

  // rhs_domain.domain.print_domain();

  std::index_sequence<2> smoother_sequence_pre{};
  std::index_sequence<2> smoother_sequence_post{};
  GS_Smoother g_smoother(rhs_domain);

  // g_smoother(Integer<2>{}, std::index_sequence<2>{}, lhs_domain2,
  // lhs_domain1,
  //            rhs_domain, (DataType)1.0, (DataType)1.0);

  // q.wait();

  // lhs_domain2.get_domain().print_domain();

  cg_solver::Solver_CG solver(Float<static_cast<DataType>(1e-8)>{},
                              rhs_domain.template get_domain<1>(),
                              diff_operator.template get_values<1>(),
                              diff_operator.template get_offsets<1>());

  // mult_level.print_operator();
  std::index_sequence<1> num_iters_level{};
  V_Cycle_base v_cycle(g_smoother, g_smoother, solver, lhs_domain1, mult_level,
                       diff_operator, coarser);

  auto *current = &lhs_domain2;
  auto *next = &lhs_domain3;
  Domain<3, std::get<0>(length), std::get<1>(length), std::get<2>(length)>
      helper2(Paddings::PERIODIC, 1);

  //  std::ofstream previous("previous_domain.dx");
  //  next->get_domain().print_dx_to_stream(previous, 0, 0, 0, 1);
  //  previous.close();

  // std::cout << "The right hand side domain is: " << std::endl;
  // rhs_domain.get_domain().print_domain();
  //  std::stringstream filenames;
  //
  //  filenames << "deviations" << base_length << ".txt";
  //
  //  std::ofstream out_file_devation(filenames.str());

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int num = 0; num < num_iter; num++) {

    //  DataType const deviation = compute_truth_deviation(
    //      u_domain, lhs_domain1.get_domain(), helper,
    //      diff_operator.get_values(), diff_operator.get_offsets());

    //  DataType const deviation_grad = compute_truth_deviaton_gradient(
    //      u_domain, lhs_domain1.get_domain(), helper, helper2,
    //      diff_operator.get_values(), diff_operator.get_offsets(), 1 /
    // 511.);

    //  out_file_devation << "The deviation after " << num << " iterations
    //  is "
    //                    << deviation << " The deviation_grad is "
    //                    << deviation_grad << std::endl;

    // Computing the defect
    convolution::Subtract_Convolve(
        defect_domain.get_domain(), lhs_domain1.get_domain(),
        rhs_domain.get_domain(), diff_operator.get_values(),
        diff_operator.get_offsets());

    v_cycle.iteration(*next, *current, defect_domain, mult_level, diff_operator,
                      coarser, upper_grid_step, omega, num_iters_level,
                      smoother_sequence_pre, smoother_sequence_post, true);

    add_domains(lhs_domain1.get_domain(), next->get_domain(),
                lhs_domain1.get_domain());
  }

  //  auto &sol_domain = current->get_domain();
  //
  //  // std::cout << "The sol domain is given by: " << std::endl;
  //  // sol_domain.print_domain();
  //
  //  DataType result = compute_deviation(lhs_domain1.get_domain(), h);
  //
  //  std::cout << "The deviation is given by " << result << std::endl;
  //
  //  DataType e_norm = compute_energy_norm(lhs_domain1.get_domain(), h);
  //
  //  std::cout << "The gradient is given by " << e_norm << std::endl;

  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end_time - start_time;
  std::cout << "The required time was: " << duration.count() << " seconds"
            << std::endl;
}
