

#ifndef PRECISION_H
#define PRECISION_H

namespace pmgc {

#ifdef USE_DOUBLE
using DataType = double;
#else
using DataType = float;
#endif

} // namespace pmgc

#endif
