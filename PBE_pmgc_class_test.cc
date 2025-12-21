#include "PBE_pmgc_class.h"
#include "fileio.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <type_traits>

constexpr std::size_t base_length = 2;
constexpr std::size_t nlev = 3;
constexpr DataType box_length = 16;

template <typename T> struct TD;
template <std::size_t nlev> struct TD2;

// struct dummy_struct {
//   ~dummy_struct() {
//     std::cout << "The final value of i is: " << domain::i << std::endl;
//   }
// };
//
// dummy_struct d{};

int main(int argc, char *argv[]) {

  DataType grid_step =
      pmgc_solver::PBE_linear_problem<base_length, nlev, box_length>::grid_step;

  std::cout << "The value of grid_step is: " << grid_step << std::endl;

  if (argc < 3) {
    std::cout
        << "Usage: ./this_program in_file out_file x_min y_min z_min num_iters"
        << std::endl;
    return 0;
  }

#ifdef DEBUGMODE
  sycl::cpu_selector selector;
#else
  sycl::gpu_selector selector;
#endif

  sycl::queue q{selector,
                sycl::property_list{sycl::property::queue::in_order{}}};

  auto dev = q.get_device();

  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << std::endl;

  std::string filename_in{argv[1]};
  std::string filename_out{argv[2]};
  std::list<Atom<DataType>> atom_list;
  read_pqr_file(filename_in, atom_list);

  const auto num_atoms = atom_list.size();

  std::vector<Atom<DataType>> atoms_vector;
  atoms_vector.reserve(num_atoms);

  auto x_min = std::stod(argv[3]);
  auto y_min = std::stod(argv[4]);
  auto z_min = std::stod(argv[5]);
  int num_iters = std::stoi(argv[6]);

  for (auto &atom : atom_list) {
    atom.Position[0] -= x_min;
    atom.Position[1] -= y_min;
    atom.Position[2] -= z_min;
    // atom.radius += ionradius;
    atoms_vector.push_back(atom);
  }

  {
    pmgc_solver::PBE_linear_problem<base_length, nlev, box_length> mg_solver(
        (DataType)0, DataType(0.15), q);

    using d_type = std::remove_reference_t<
        decltype(mg_solver.sol.template get_domain<nlev>())>;

    {

      auto start = std::chrono::high_resolution_clock::now();

      mg_solver.initialize_boundary(atoms_vector);
      mg_solver.initialize_epsilons(atoms_vector);
      mg_solver.set_up_rhs(atoms_vector);
      mg_solver.buildmultilevelops<std::true_type>();

      auto end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> duration = end - start;

      std::cout << "The required time for the problem setup was: "
                << duration.count() << std::endl;
    }

    // std::cout << "The initial residual is:" << mg_solver.compute_residual()
    //           << std::endl;

    const auto initial_res_2 = mg_solver.compute_residual_2();
    const auto initial_res_1 = mg_solver.compute_residual_1();

    //  mg_solver.solve_by_cg<nlev>(Float<1e-5>{});
    //
    //  std::cout << "After the cg_method on top level, the residual is: "
    //            << mg_solver.compute_residual() << std::endl;

    q.wait();

    // for (int i = 0; i < num_iters; i++) {
    //   mg_solver.smooth_domain_sol(2);
    //   mg_solver.compute_defect_sol_2_sol2();
    //   mg_solver.restrict_domain_sol2_2_rhs();
    //   mg_solver.solve_by_cg(Float<1e-5>{});
    //   mg_solver.prolong_sol_2_sol2();
    //   mg_solver.add_domain_sol_sol_sol2();
    //   mg_solver.smooth_domain_sol(2);

    //   std::cout << "The residual after " << i + 1 << " iterations is "
    //             << mg_solver.compute_residual() << std::endl;
    // }

    //  std::cout << "oE map" << std::endl;
    //  mg_solver.epsilon_oE_map.get_domain().print_domain();
    //  std::cout << "oN map" << std::endl;
    //  mg_solver.epsilon_oN_map.get_domain().print_domain();
    //  std::cout << "uC map" << std::endl;
    //  mg_solver.epsilon_uC_map.get_domain().print_domain();
    //
    //

    //  mg_solver.sol.set_zero<DataType>();
    //  q.wait();

    // mg_solver.rhs_domain.get_domain().print_domain();
    //   mg_solver.smooth_domain_sol(2);
    //   mg_solver.compute_defect_sol_2_sol2();
    //   mg_solver.restrict_domain_sol2_2_rhs();
    // mg_solver.rhs_domain.template get_domain<nlev - 1>().print_domain();
    //  mg_solver.template smooth_domain_sol<std::true_type, nlev - 1>(2, 0,
    //  true); mg_solver.compute_defect_sol_2_sol2<nlev - 1>();
    //  mg_solver.sol2.get_domain().print_domain();
    // mg_solver.restrict_domain_sol2_2_rhs<1>();
    //   mg_solver.rhs_domain.template get_domain<nlev - 1>().print_domain();
    // mg_solver.template solve_by_cg<1>(Float<1e-8>{});
    // mg_solver.template prolong_sol_2_sol2<nlev - 1>();
    // mg_solver.template add_domain_sol_sol_sol2<nlev - 1>();
    // mg_solver.template smooth_domain_sol<std::true_type, nlev - 1>(2, 1);
    // mg_solver.template prolong_sol_2_sol2<nlev>();
    // mg_solver.template add_domain_sol_sol_sol2<nlev>();
    // mg_solver.smooth_domain_sol(2, 1);

    // This is where the second iteration starts
    //  mg_solver.smooth_domain_sol(2);
    //  mg_solver.compute_defect_sol_2_sol2();
    // mg_solver.sol2.get_domain().print_domain();
    //  mg_solver.restrict_domain_sol2_2_rhs();
    //  mg_solver.template smooth_domain_sol<std::true_type, nlev - 1>(2, 0,
    //  true);

    // mg_solver.rhs_domain.template get_domain<nlev - 1>().print_domain();
    // mg_solver.sol.template get_domain<nlev - 1>().print_domain();

    DataType current_res_1 = 0;
    DataType previous_res_1 = initial_res_1;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iters; i++) {
      mg_solver.v_cycle<std::true_type>();

      current_res_1 = mg_solver.compute_residual_1();

      std::cout << "The residual after " << i + 1 << " iterations is "
                << current_res_1 / initial_res_1 << std::endl;

      std::cout << "contraction number " << current_res_1 / previous_res_1
                << std::endl;

      previous_res_1 = current_res_1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "The required time for the solution was: " << duration.count()
              << std::endl;

    // mg_solver.sol.get_domain().print_domain();

    std::ofstream out_file{"output_potential.dx"};

    //    mg_solver.sol.get_domain().print_dx_to_stream(out_file, x_min, y_min,
    //    z_min,
    //                                                  box_length);
  }

  return 0;
}
