#include "create_charge_distribution.h"
#include "Atom_types.h"
#include "Domain.h"
#include "dot_finder.h"
#include "fileio.h"
#include "hipSYCL/sycl/queue.hpp"
#include "hipSYCL/sycl/usm.hpp"
#include "predefinitions.h"
#include <fstream>
#include <iostream>
#include <sycl/sycl.hpp>

int main(int argc, char *argv[]) {
  if (argc < 6) {
    std::cout << "Usage: ./program in_file out_file origin_x origin_y origin_z "
                 "ion_radius"
              << std::endl;
  }

  constexpr std::size_t length = 95;

  sycl::cpu_selector selector;
  sycl::queue q{selector,
                sycl::property_list{sycl::property::queue::in_order{}}};

  std::string pqr_file{argv[1]};
  std::ofstream dx_file{argv[2]};

  std::array<double, 3> a{std::stod(argv[3]), std::stod(argv[4]),
                          std::stod(argv[5])};

  double radius = std::stod(argv[6]);

  std::list<Atom<DataType>> atoms;
  read_pqr_file(pqr_file, atoms);

  std::vector<Atom<DataType>> atoms_vector;
  atoms_vector.reserve(atoms.size());

  for (auto &atom : atoms) {
    atom.Position[0] -= a[0];
    atom.Position[1] -= a[1];
    atom.Position[2] -= a[2];
    atom.radius += radius;
    atoms_vector.push_back(atom);
  }

  Atom<DataType> *atoms_device =
      sycl::malloc_device<Atom<DataType>>(atoms.size(), q);

  q.memcpy(atoms_device, atoms_vector.data(),
           atoms_vector.size() * sizeof(Atom<DataType>));

  domain::Domain<3, length, length, length> domain(Paddings::PERIODIC, q, 1);
  q.wait();

  // q.parallel_for(sycl::range<1>(atoms.size()), [=](sycl::id<1> I) {
  // q.submit([=](sycl::handler &h) {
  //   h.single_task([=]() {
  for (int I = 0; I < atoms.size(); I++)
    add_charges_to_distribution(domain, atoms_device[I].Position,
                                atoms_device[I].charge,
                                spacing<DataType, 1.>{});
  //   });
  // }).wait();
  //}).wait();

  domain.print_dx_to_stream(dx_file, a[0], a[1], a[2], 96.);
}
