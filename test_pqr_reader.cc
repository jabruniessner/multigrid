#include "Atom_types.h"
#include "fileio.h"
#include <chrono>
#include <list>
#include <string>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "No filename given" << std::endl;
    std::cout << "Usage: ./program filename" << std::endl;
    return -1;
  }
  std::string filename{argv[1]};
  std::list<Atom<DataType>> Atom_list;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 400; i++) {
    read_pqr_file(filename, Atom_list);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;

  std::cout << "The required time was: " << duration.count() << std::endl;

  // for (auto &atom : Atom_list)
  //   atom.print();
}
