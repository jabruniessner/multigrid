#include "functions.h"
#include <abps_macros.h>
#include <buildPd.h>
#include <iostream>
#include <matvecd.h>
#include <random>
#include <sycl/sycl.hpp>

using DataType = double;

extern "C" void interp2_(int *nxc, int *nyc, int *nzc, int *nxf, int *nyf,
                         int *nzf, DataType *xin, DataType *xout, DataType *oPC,
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

  DataType *oPC = new DataType[nxc * nyc * nzc]();
  DataType *oPN = new DataType[nxc * nyc * nzc]();
  DataType *oPS = new DataType[nxc * nyc * nzc]();
  DataType *oPE = new DataType[nxc * nyc * nzc]();
  DataType *oPW = new DataType[nxc * nyc * nzc]();
  DataType *oPNE = new DataType[nxc * nyc * nzc]();
  DataType *oPNW = new DataType[nxc * nyc * nzc]();
  DataType *oPSE = new DataType[nxc * nyc * nzc]();
  DataType *oPSW = new DataType[nxc * nyc * nzc]();
  DataType *uPC = new DataType[nxc * nyc * nzc]();
  DataType *uPN = new DataType[nxc * nyc * nzc]();
  DataType *uPS = new DataType[nxc * nyc * nzc]();
  DataType *uPE = new DataType[nxc * nyc * nzc]();
  DataType *uPW = new DataType[nxc * nyc * nzc]();
  DataType *uPNE = new DataType[nxc * nyc * nzc]();
  DataType *uPNW = new DataType[nxc * nyc * nzc]();
  DataType *uPSE = new DataType[nxc * nyc * nzc]();
  DataType *uPSW = new DataType[nxc * nyc * nzc]();
  DataType *dPC = new DataType[nxc * nyc * nzc]();
  DataType *dPN = new DataType[nxc * nyc * nzc]();
  DataType *dPS = new DataType[nxc * nyc * nzc]();
  DataType *dPE = new DataType[nxc * nyc * nzc]();
  DataType *dPW = new DataType[nxc * nyc * nzc]();
  DataType *dPNE = new DataType[nxc * nyc * nzc]();
  DataType *dPNW = new DataType[nxc * nyc * nzc]();
  DataType *dPSE = new DataType[nxc * nyc * nzc]();
  DataType *dPSW = new DataType[nxc * nyc * nzc]();

  DataType *xin = new DataType[nxc * nyc * nzc]();
  DataType *xout_c = new DataType[nxf * nyf * nzf]();
  DataType *xout_f = new DataType[nxf * nyf * nzf]();

  initialize_random(xin, xin, nxc * nyc * nzc);

  initialize_random(oPC, oPC, nxc * nyc * nzc);
  initialize_random(oPN, oPN, nxc * nyc * nzc);
  initialize_random(oPS, oPS, nxc * nyc * nzc);
  initialize_random(oPE, oPE, nxc * nyc * nzc);
  initialize_random(oPW, oPW, nxc * nyc * nzc);
  initialize_random(oPNE, oPNE, nxc * nyc * nzc);
  initialize_random(oPNW, oPNW, nxc * nyc * nzc);
  initialize_random(oPSE, oPSE, nxc * nyc * nzc);
  initialize_random(oPSW, oPSW, nxc * nyc * nzc);
  initialize_random(uPC, uPC, nxc * nyc * nzc);
  initialize_random(uPN, uPN, nxc * nyc * nzc);
  initialize_random(uPS, uPS, nxc * nyc * nzc);
  initialize_random(uPE, uPE, nxc * nyc * nzc);
  initialize_random(uPW, uPW, nxc * nyc * nzc);
  initialize_random(uPNE, uPNE, nxc * nyc * nzc);
  initialize_random(uPNW, uPNW, nxc * nyc * nzc);
  initialize_random(uPSE, uPSE, nxc * nyc * nzc);
  initialize_random(uPSW, uPSW, nxc * nyc * nzc);
  initialize_random(dPC, dPC, nxc * nyc * nzc);
  initialize_random(dPN, dPN, nxc * nyc * nzc);
  initialize_random(dPS, dPS, nxc * nyc * nzc);
  initialize_random(dPE, dPE, nxc * nyc * nzc);
  initialize_random(dPW, dPW, nxc * nyc * nzc);
  initialize_random(dPNE, dPNE, nxc * nyc * nzc);
  initialize_random(dPNW, dPNW, nxc * nyc * nzc);
  initialize_random(dPSE, dPSE, nxc * nyc * nzc);
  initialize_random(dPSW, dPSW, nxc * nyc * nzc);

  interp2_(&nxc, &nyc, &nzc, &nxf, &nyf, &nzf, xin, xout_f, oPC, oPN, oPS, oPE,
           oPW, oPNE, oPNW, oPSE, oPSW, uPC, uPN, uPS, uPE, uPW, uPNE, uPNW,
           uPSE, uPSW, dPC, dPN, dPS, dPE, dPW, dPNE, dPNW, dPSE, dPSW);

  VinterpPMG2(&nxc, &nyc, &nzc, &nxf, &nyf, &nzf, xin, xout_c, oPC, oPN, oPS,
              oPE, oPW, oPNE, oPNW, oPSE, oPSW, uPC, uPN, uPS, uPE, uPW, uPNE,
              uPNW, uPSE, uPSW, dPC, dPN, dPS, dPE, dPW, dPNE, dPNW, dPSE, dPSW,
              q);

  q.wait();

  compare(xout_f, xout_c, nxf * nyf * nzf);
}
