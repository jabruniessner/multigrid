#include "Atom_types.h"
#include <list>
#include <string>

#ifndef READ_PQR_FILE_H
#define READ_PQR_FILE_H

bool read_pqr_file(std::string &filename, std::list<Atom<DataType>> &atom_list);

#endif
