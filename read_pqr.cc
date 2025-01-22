#include "Atom_types.h"
#include "predefinitions.h"
#include <assert.h>
#include <cassert>
#include <fstream>
#include <list>
#include <sstream>
#include <string>

bool read_pqr_file(std::string &filename,
                   std::list<Atom<DataType>> &atom_list) {

  std::ifstream file{filename};
  std::string line;

  if (file.is_open()) {
    while (std::getline(file, line)) {
      std::vector<std::string> li;
      std::istringstream iss(line);

      for (std::string s; iss >> s;) {
        li.push_back(s);
      }

      if (li[0] == "REMARK") {
        continue;
      }
      if (li.size() == 10) {
        const DataType x = stod(li[5]);
        const DataType y = stod(li[6]);
        const DataType z = stod(li[7]);
        const DataType c = stod(li[8]);
        const DataType r = stod(li[9]);

        if (r < 1e-6)
          continue;

        // atom_list.push_back();

        Atom<DataType> a{.charge = c};
        a.Position = {x, y, z};
        a.radius = r;

        atom_list.push_back(a);
      }
    }
  } else {
    std::cout << "Unable to open file" << std::endl;
    assert(false);
  }

  return true;
}
