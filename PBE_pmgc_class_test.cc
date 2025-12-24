#include "PBE_pmgc_class.h"
#include "fileio.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <type_traits>

constexpr std::size_t base_length = 22;
constexpr std::size_t nlev = 4;
constexpr DataType box_length = 96;

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

    d_type init_guess(Paddings::PERIODIC, q, 1),
        rhs_domain(Paddings::PERIODIC, q, 1);

    {

      auto start = std::chrono::high_resolution_clock::now();

      mg_solver.initialize_boundary(atoms_vector);
      mg_solver.initialize_epsilons(atoms_vector);
      mg_solver.set_up_rhs_extern(atoms_vector, rhs_domain);
      mg_solver.buildmultilevelops<std::false_type>();

      auto end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> duration = end - start;

      std::cout << "The required time for the problem setup was: "
                << duration.count() << std::endl;
    }

    const auto initial_res_2 = mg_solver.compute_residual_2();
    const auto initial_res_1 = mg_solver.compute_residual_1();

    q.wait();

    DataType current_res_1 = 0;
    DataType previous_res_1 = initial_res_1;

    auto main_solver =
        cg_solver::make_general_solver<DataType>(mg_solver.sol.get_domain());
    cg_solver::counting_criterion counter(num_iters);

    auto start_time = std::chrono::high_resolution_clock::now();

    auto map = mg_solver.get_map();

    main_solver(init_guess, rhs_domain, map, counter, mg_solver, true);

    //  main_solver();

    //  for (int i = 0; i < num_iters; i++) {
    //    mg_solver.v_cycle<std::false_type>();

    //    current_res_1 = mg_solver.compute_residual_1();

    //    std::cout << "The residual after " << i + 1 << " iterations is "
    //              << current_res_1 / initial_res_1 << std::endl;

    //    std::cout << "contraction number " << current_res_1 / previous_res_1
    //              << std::endl;

    //    previous_res_1 = current_res_1;
    //  }

    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "The required time for the solution was: " << duration.count()
              << std::endl;

    std::ofstream out_file{"output_potential.dx"};
  }

  return 0;
}
