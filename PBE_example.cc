#include "Convolution.h"
#include "Debye_Hueckel_functions.h"
#include "MultigridDomain.h"
#include "cycles.h"
#include "fileio.h"
#include "hipSYCL/sycl/device_selector.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include "hipSYCL/sycl/usm.hpp"
#include "level_transition.h"
#include "scientific_quantities.h"
#include <array>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace cycles;
using namespace convolution;

constexpr Dimension Dim = 3;
constexpr std::size_t nlev = 4u;
constexpr std::size_t base_length = 6;
constexpr DataType omega = 4. / 5.;
constexpr DataType box_length = 96;
constexpr double ionic_strength = 0.005;
constexpr DataType kappa = KappaA(ionic_strength);
constexpr DataType ionradius = 1.5;

using Domain_Type =
    Multigrid_domain<Dim, nlev, base_length, base_length, base_length>;

using Domain_Type_upper = decltype(Domain_Type::domain_t_v)::domain_t;

DataType sqr(double val) { return val * val; }

template <std::size_t level = nlev>
void Set_boundary_conditions(Atom<DataType> *atoms, std::size_t num_atoms,
                             const Domain_Type_upper &domain, std::size_t x,
                             std::size_t y, std::size_t z) {
  for (int i = 0; i < num_atoms; i++) {
    const DataType distance = std::sqrt(sqr(x - atoms[i].Position[0]) +
                                        sqr(y - atoms[i].Position[1]) +
                                        sqr(z - atoms[i].Position[2]));

    domain(x, y, z) =
        DH_Sphere(atoms[i].radius, atoms[i].charge, distance, kappa);
  }
}

int main(int argc, char *argv[]) {

  using OffsetType = std::array<int, Dim>;

  if (argc < 3) {
    std::cout
        << "Usage: ./this_program in_file out_file x_min y_min z_min num_iters"
        << std::endl;
  }
  // int num_iters = std::stoi(argv[3]);

#ifdef DEBUGMODE
  sycl::cpu_selector selector;
#else
  sycl::gpu_selector selector;
#endif

  sycl::queue q{selector,
                sycl::property_list{sycl::property::queue::in_order{}}};

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
  auto num_iters = std::stod(argv[6]);

  for (auto &atom : atom_list) {
    atom.Position[0] -= x_min;
    atom.Position[1] -= y_min;
    atom.Position[2] -= z_min;
    atom.radius += ionradius;
    atoms_vector.push_back(atom);
  }

  Atom<DataType> *atoms_device =
      sycl::malloc_device<Atom<DataType>>(atom_list.size(), q);

  q.memcpy(atoms_device, atoms_vector.data(),
           atoms_vector.size() * sizeof(Atom<DataType>))
      .wait();

  Domain_Type lhs_domain1(q), lhs_domain2(q), rhs_domain(q), boundary_values(q),
      epsilon(q), kappa_(q);

  constexpr auto &length = Domain_Type::length;

  {
    const auto &boundary_domain = boundary_values.template get_domain<nlev>();
    q.parallel_for(
         sycl::range<2>(std::get<1>(length) + 2, std::get<2>(length) + 2),
         [=](sycl::id<2> I) {
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, I[0], I[1], 0);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, I[0], I[1],
                                         std::get<2>(length) + 1);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, 0, I[0], I[1]);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain,
                                         std::get<0>(length) + 1, I[0], I[1]);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, I[0], 0, I[1]);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, I[0],
                                         std::get<1>(length) + 1, I[1]);
         })
        .wait();

    std::ofstream out_file{filename_out};
    boundary_domain.print_dx_to_stream(out_file, x_min, y_min, z_min,
                                       box_length);
  }

  std::cout << "The length is: " << std::get<0>(length) << std::endl;

  return 0;
}
