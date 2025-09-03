#include "functions.h"
#include "hipSYCL/sycl/device_selector.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include <gsd.h>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <vmatrix.h>

using DataType = double;

extern "C" void gsrb7x_(int *nx, int *ny, int *nz, int *ipc, double *rpc,
                        double *oC, double *cc, double *fc, double *oE,
                        double *oN, double *uC, double *x, double *w1,
                        double *w2, double *r, int *itmax, int *iter,
                        int *errtol, double *omega, int *iresid, int *iadjoint);

int nxf = 5;
int nyf = 5;
int nzf = 5;
int nxc = 3;
int nyc = 3;
int nzc = 3;

int main() {
  sycl::cpu_selector selector;
  sycl::queue q{selector,
                sycl::property_list{sycl::property::queue::in_order{}}};

  DataType *oC = new DataType[nxf * nyf * nzf]();
  DataType *oE = new DataType[nxf * nyf * nzf]();
  DataType *oN = new DataType[nxf * nyf * nzf]();
  DataType *uC = new DataType[nxf * nyf * nzf]();
  DataType *oNE = new DataType[nxf * nyf * nzf]();
  DataType *oNW = new DataType[nxf * nyf * nzf]();
  DataType *uE = new DataType[nxf * nyf * nzf]();
  DataType *uW = new DataType[nxf * nyf * nzf]();
  DataType *uN = new DataType[nxf * nyf * nzf]();
  DataType *uS = new DataType[nxf * nyf * nzf]();
  DataType *uNE = new DataType[nxf * nyf * nzf]();
  DataType *uNW = new DataType[nxf * nyf * nzf]();
  DataType *uSE = new DataType[nxf * nyf * nzf]();
  DataType *uSW = new DataType[nxf * nyf * nzf]();
  DataType *fc = new DataType[nxf * nyf * nzf]();
  DataType *cc = new DataType[nxf * nyf * nzf]();
  DataType *xc = new DataType[nxf * nyf * nzf]();
  DataType *xf = new DataType[nxf * nyf * nzf]();
  DataType *r = new DataType[nxf * nyf * nzf]();
  DataType *rpc = new DataType[nxf * nyf * nzf]();

  int *itermax = new int;
  *itermax = 1;
  int *iters = new int;
  *iters = 0;
  int *iadjoint = new int;
  *iadjoint = 0;
  int *ipc = new int[nxf * nyf * nzf]();

  int *iresid = new int;
  *iresid = 0;

  initialize_random(oC, oC, nxf * nyf * nzf);
  initialize_random(oE, oE, nxf * nyf * nzf);
  initialize_random(oN, oN, nxf * nyf * nzf);
  initialize_random(uC, uC, nxf * nyf * nzf);
  initialize_random(oNE, oNE, nxf * nyf * nzf);
  initialize_random(oNW, oNW, nxf * nyf * nzf);
  initialize_random(uE, uE, nxf * nyf * nzf);
  initialize_random(uW, uW, nxf * nyf * nzf);
  initialize_random(uN, uN, nxf * nyf * nzf);
  initialize_random(uS, uS, nxf * nyf * nzf);
  initialize_random(uNE, uNE, nxf * nyf * nzf);
  initialize_random(uNW, uNW, nxf * nyf * nzf);
  initialize_random(uSE, uSE, nxf * nyf * nzf);
  initialize_random(uSW, uSW, nxf * nyf * nzf);
  initialize_random(xc, xf, nxf * nyf * nzf);

  gsrb7x_(&nxf, &nyf, &nzf, ipc, rpc, oC, cc, fc, oE, oN, uC, xf, nullptr,
          nullptr, r, itermax, iters, nullptr, nullptr, iresid, iadjoint);

  Vgsrb7x(&nxf, &nyf, &nzf, ipc, rpc, oC, cc, fc, oE, oN, uC, xc, nullptr,
          nullptr, r, itermax, iters, nullptr, nullptr, iresid, iadjoint, q);

  q.wait();

  compare(xc, xf, nxf * nyf * nzf);
}
