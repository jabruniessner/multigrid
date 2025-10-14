#include "Atom_types.h"
#include "Domain.h"
#include "dot_finder.h"
#include "fileio.h"
#include <algorithm>
#include <boost/iterator/counting_iterator.hpp>
#include <execution>
#include <fstream>
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc < 6) {
    std::cout << "Usage: ./program in_file out_file origin_x origin_y origin_z "
                 "ion_radius"
              << std::endl;
    return 0;
  }

  constexpr std::size_t length = 95;

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

  Atom<DataType> *atoms_device = atoms_vector.data();

  domain::Domain<3, length, length, length> domain(Paddings::PERIODIC, 1);

  boost::iterators::counting_iterator<int> start(0);

  {
    boost::iterators::counting_iterator<int> end(atoms.size());

    std::for_each(std::execution::par_unseq, start, end, [=](int I) {
      find_dots_in_sphere(atoms_device[I], domain, static_cast<DataType>(1.));
    });
  }

  {
    boost::iterators::counting_iterator<int> end((length + 2) * (length + 2) *
                                                 (length + 2));

    std::for_each(std::execution::par_unseq, start, end, [=](int idx) {
      auto I =
          domain::flat_to_multi_index<length + 2, length + 2, length + 2>(idx);

      domain(I[0], I[1], I[2]) = (domain(I[0], I[1], I[2]) < .9) * 1.f;
    });
  }

  domain.print_dx_to_stream(dx_file, a[0], a[1], a[2], 96.f);
}
