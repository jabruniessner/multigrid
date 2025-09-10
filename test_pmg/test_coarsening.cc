#include "functions.h"
#include <abps_macros.h>
#include <buildPd.h>
#include <iostream>
#include <matvecd.h>
#include <random>
#include <sycl/sycl.hpp>

using DataType = double;

extern "C" void restrc2_(int *nxf, int *nyf, int *nzf, int *nxc, int *nyc,
                         int *nzc, DataType *xin, DataType *xout, DataType *oPC,
                         DataType *oPN, DataType *oPS, DataType *oPE,
                         DataType *oPW, DataType *oPNE, DataType *oPNW,
                         DataType *oPSE, DataType *oPSW, DataType *uPC,
                         DataType *uPN, DataType *uPS, DataType *uPE,
                         DataType *uPW, DataType *uPNE, DataType *uPNW,
                         DataType *uPSE, DataType *uPSW, DataType *dPC,
                         DataType *dPN, DataType *dPS, DataType *dPE,
                         DataType *dPW, DataType *dPNE, DataType *dPNW,
                         DataType *dPSE, DataType *dPSW);

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

  DataType *oPC = new DataType[nxf * nyf * nzf]();
  DataType *oPN = new DataType[nxf * nyf * nzf]();
  DataType *oPS = new DataType[nxf * nyf * nzf]();
  DataType *oPE = new DataType[nxf * nyf * nzf]();
  DataType *oPW = new DataType[nxf * nyf * nzf]();
  DataType *oPNE = new DataType[nxf * nyf * nzf]();
  DataType *oPNW = new DataType[nxf * nyf * nzf]();
  DataType *oPSE = new DataType[nxf * nyf * nzf]();
  DataType *oPSW = new DataType[nxf * nyf * nzf]();
  DataType *uPC = new DataType[nxf * nyf * nzf]();
  DataType *uPN = new DataType[nxf * nyf * nzf]();
  DataType *uPS = new DataType[nxf * nyf * nzf]();
  DataType *uPE = new DataType[nxf * nyf * nzf]();
  DataType *uPW = new DataType[nxf * nyf * nzf]();
  DataType *uPNE = new DataType[nxf * nyf * nzf]();
  DataType *uPNW = new DataType[nxf * nyf * nzf]();
  DataType *uPSE = new DataType[nxf * nyf * nzf]();
  DataType *uPSW = new DataType[nxf * nyf * nzf]();
  DataType *dPC = new DataType[nxf * nyf * nzf]();
  DataType *dPN = new DataType[nxf * nyf * nzf]();
  DataType *dPS = new DataType[nxf * nyf * nzf]();
  DataType *dPE = new DataType[nxf * nyf * nzf]();
  DataType *dPW = new DataType[nxf * nyf * nzf]();
  DataType *dPNE = new DataType[nxf * nyf * nzf]();
  DataType *dPNW = new DataType[nxf * nyf * nzf]();
  DataType *dPSE = new DataType[nxf * nyf * nzf]();
  DataType *dPSW = new DataType[nxf * nyf * nzf]();

  DataType *xin = new DataType[nxf * nyf * nzf]();
  DataType *xout_c = new DataType[nxc * nyc * nzc]();
  DataType *xout_f = new DataType[nxc * nyc * nzc]();

  initialize_random(xin, xin, nxf * nyf * nzf);

  initialize_random(oPC, oPC, nxf * nyf * nzf);
  initialize_random(oPN, oPN, nxf * nyf * nzf);
  initialize_random(oPS, oPS, nxf * nyf * nzf);
  initialize_random(oPE, oPE, nxf * nyf * nzf);
  initialize_random(oPW, oPW, nxf * nyf * nzf);
  initialize_random(oPNE, oPNE, nxf * nyf * nzf);
  initialize_random(oPNW, oPNW, nxf * nyf * nzf);
  initialize_random(oPSE, oPSE, nxf * nyf * nzf);
  initialize_random(oPSW, oPSW, nxf * nyf * nzf);
  initialize_random(uPC, uPC, nxf * nyf * nzf);
  initialize_random(uPN, uPN, nxf * nyf * nzf);
  initialize_random(uPS, uPS, nxf * nyf * nzf);
  initialize_random(uPE, uPE, nxf * nyf * nzf);
  initialize_random(uPW, uPW, nxf * nyf * nzf);
  initialize_random(uPNE, uPNE, nxf * nyf * nzf);
  initialize_random(uPNW, uPNW, nxf * nyf * nzf);
  initialize_random(uPSE, uPSE, nxf * nyf * nzf);
  initialize_random(uPSW, uPSW, nxf * nyf * nzf);
  initialize_random(dPC, dPC, nxf * nyf * nzf);
  initialize_random(dPN, dPN, nxf * nyf * nzf);
  initialize_random(dPS, dPS, nxf * nyf * nzf);
  initialize_random(dPE, dPE, nxf * nyf * nzf);
  initialize_random(dPW, dPW, nxf * nyf * nzf);
  initialize_random(dPNE, dPNE, nxf * nyf * nzf);
  initialize_random(dPNW, dPNW, nxf * nyf * nzf);
  initialize_random(dPSE, dPSE, nxf * nyf * nzf);
  initialize_random(dPSW, dPSW, nxf * nyf * nzf);

  restrc2_(&nxf, &nyf, &nzf, &nxc, &nyc, &nzc, xin, xout_f, oPC, oPN, oPS, oPE,
           oPW, oPNE, oPNW, oPSE, oPSW, uPC, uPN, uPS, uPE, uPW, uPNE, uPNW,
           uPSE, uPSW, dPC, dPN, dPS, dPE, dPW, dPNE, dPNW, dPSE, dPSW);

  Vrestrc2(&nxf, &nyf, &nzf, &nxc, &nyc, &nzc, xin, xout_c, oPC, oPN, oPS, oPE,
           oPW, oPNE, oPNW, oPSE, oPSW, uPC, uPN, uPS, uPE, uPW, uPNE, uPNW,
           uPSE, uPSW, dPC, dPN, dPS, dPE, dPW, dPNE, dPNW, dPSE, dPSW, q);

  q.wait();

  compare(xout_c, xout_f, nxc * nyc * nzc);

  //  for (int i = 0; i < 27; i++) {
  //    std::cout << "xout_c: " << xout_c[i] << " xout_f: " << xout_f[i]
  //              << std::endl;
  //  }
}
