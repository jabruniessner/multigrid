#include "PBE_pmgc_class.h"
#include "fileio.h"

constexpr std::size_t base_length = 1;
constexpr std::size_t nlev = 2;
constexpr DataType box_length = 16;

template <typename T> struct TD;
template <std::size_t nlev> struct TD2;

int main(int argc, char *argv[]) {

  std::size_t grid_step =
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

  pmgc_solver::PBE_linear_problem<base_length, nlev, box_length> mg_solver(
      (DataType)0, DataType(0.15), q);

  mg_solver.initialize_boundary(atoms_vector);
  mg_solver.initialize_epsilons(atoms_vector);
  mg_solver.set_up_rhs(atoms_vector);
  mg_solver.buildmultilevelops();

  std::cout << "The initial residual is:" << mg_solver.compute_residual()
            << std::endl;

  //  mg_solver.solve_by_cg<nlev>(Float<1e-5>{});
  //
  //  std::cout << "After the cg_method on top level, the residual is: "
  //            << mg_solver.compute_residual() << std::endl;

  q.wait();

  mg_solver.smooth_domain_sol(2);
  mg_solver.smooth_domain_sol2(2);

  std::cout << "After 2 iterations the sol domain is: " << std::endl;
  mg_solver.sol2.get_domain().print_domain();

  //  TD<decltype(mg_solver.epsilon_oNE_map)> td;
  //  TD2<nlev> td2;

  // mg_solver.rhs_domain.get_domain().print_domain();

  //  std::cout << "oC epsilon" << std::endl;
  //  mg_solver.epsilon_oC_map.get_domain().print_domain();
  //  std::cout << "oE epsilon" << std::endl;
  //  mg_solver.epsilon_oE_map.get_domain().print_domain();
  //  std::cout << "oN epsilon" << std::endl;
  //  mg_solver.epsilon_oN_map.get_domain().print_domain();
  //  std::cout << "uC epsilon" << std::endl;
  //  mg_solver.epsilon_uC_map.get_domain().print_domain();

  // mg_solver.sol.get_domain().print_domain();

  return 0;
}
