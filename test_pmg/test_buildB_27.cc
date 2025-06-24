#include "functions.h"
#include "hipSYCL/sycl/queue.hpp"
#include <buildBd.h>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <vmatrix.h>

using DataType = double;

extern "C" void buildband1_27_(int *nx, int *ny, int *nz, int *ipc, double *rpc,
                               double *oC, double *oE, double *oN, double *uC,
                               double *oNE, double *oNW, double *uE, double *uW,
                               double *uN, double *uS, double *uNE, double *uNW,
                               double *uSE, double *uSW, int *ipcB,
                               double *rpcB, double *acB, int *n, int *m,
                               int *lda);

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

  int *ipcB_f = new int[4]();
  int *ipcB_c = new int[4]();
  auto *n = new int;
  *n = (nxf - 2) * (nyf - 2) * (nzf - 2);
  auto *m = new int;
  *m = (nxf - 2) * (nyf - 2) + (nxf - 2) + 1;
  auto *lda = new int;
  *lda = *m + 1;
  int *ipc = nullptr;
  DataType *rpc = nullptr;
  DataType *rpcB = nullptr;
  DataType *acB_f = new DataType[*lda * *n]();
  DataType *acB_c = new DataType[*lda * *n]();

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

  buildband1_27_(&nxf, &nyf, &nzf, ipc, rpc, oC, oE, oN, uC, oNE, oNW, uE, uW,
                 uN, uS, uNE, uNW, uSE, uSW, ipcB_f, rpcB, acB_f, n, m, lda);

  Vbuildband1_27(&nxf, &nyf, &nzf, ipc, rpc, oC, oE, oN, uC, oNE, oNW, uE, uW,
                 uN, uS, uNE, uNW, uSE, uSW, ipcB_c, rpcB, acB_c, n, m, lda, q);

  q.wait();

  MAT3(uC, nxf, nyf, nzf);
  MAT2(acB_c, *lda, *n);
  MAT2(acB_f, *lda, *n);

  std::cout << "The value of uC at the critical position is: ";
  std::cout << VAT3(uC, 2, 2, 1) << std::endl;

  std::cout << "The value of acB_c at the critical position is: ";
  std::cout << VAT2(acB_c, 1, 1) << std::endl;

  std::cout << "The value of acB_f at the critical position is: ";
  std::cout << VAT2(acB_f, 1, 1) << std::endl;

  std::cout << "acB_f[0] " << acB_f[0] << std::endl;
  std::cout << "acB_c[0] " << acB_c[0] << std::endl;

  compare((acB_f), (acB_c), *lda * *n);
}
