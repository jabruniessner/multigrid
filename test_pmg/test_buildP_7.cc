#include <buildPd.h>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

using DataType = double;

extern "C" void
buildpb_op7_(int *nxf, int *nyf, int *nzf, int *nxc, int *nyc, int *nzc,
             int *ipc, DataType *rpc, DataType *oC, DataType *oE, DataType *oN,
             DataType *uC, DataType *oPC, DataType *oPN, DataType *oPS,
             DataType *oPE, DataType *oPW, DataType *oPNE, DataType *oPNW,
             DataType *oPSE, DataType *oPSW, DataType *uPC, DataType *uPN,
             DataType *uPS, DataType *uPE, DataType *uPW, DataType *uPNE,
             DataType *uPNW, DataType *uPSE, DataType *uPSW, DataType *dPC,
             DataType *dPN, DataType *dPS, DataType *dPE, DataType *dPW,
             DataType *dPNE, DataType *dPNW, DataType *dPSE, DataType *dPSW);

int nxf = 5;
int nyf = 5;
int nzf = 5;
int nxc = 3;
int nyc = 3;
int nzc = 3;

void initialize_random(DataType *first, DataType *second, int num) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(0.0, 1.0);

  for (int i = 0; i < num; i++)
    first[i] = second[i] = distrib(gen);

  for (int i = 0; i < num; i++) {
    assert(first[i] == second[i]);
  }
}

void compare(DataType *first, DataType *second, int num) {
  for (int i = 0; i < num; i++)
    assert(std::abs(first[i] - second[i]) < 1e-12);
}

int main() {

  sycl::cpu_selector selector;
  sycl::queue q{selector,
                sycl::property_list{sycl::property::queue::in_order{}}};

  // Input DataType arguments
  DataType *oE = new DataType[nxf * nyf * nzf]();
  DataType *oN = new DataType[nxf * nyf * nzf]();
  DataType *oC = new DataType[nxf * nyf * nzf]();
  DataType *uC = new DataType[nxf * nyf * nzf]();

  DataType *oPC_f = new DataType[nxc * nyc * nzc]();
  DataType *oPN_f = new DataType[nxc * nyc * nzc]();
  DataType *oPS_f = new DataType[nxc * nyc * nzc]();
  DataType *oPE_f = new DataType[nxc * nyc * nzc]();
  DataType *oPW_f = new DataType[nxc * nyc * nzc]();
  DataType *oPNE_f = new DataType[nxc * nyc * nzc]();
  DataType *oPNW_f = new DataType[nxc * nyc * nzc]();
  DataType *oPSE_f = new DataType[nxc * nyc * nzc]();
  DataType *oPSW_f = new DataType[nxc * nyc * nzc]();
  DataType *uPC_f = new DataType[nxc * nyc * nzc]();
  DataType *uPN_f = new DataType[nxc * nyc * nzc]();
  DataType *uPS_f = new DataType[nxc * nyc * nzc]();
  DataType *uPE_f = new DataType[nxc * nyc * nzc]();
  DataType *uPW_f = new DataType[nxc * nyc * nzc]();
  DataType *uPNE_f = new DataType[nxc * nyc * nzc]();
  DataType *uPNW_f = new DataType[nxc * nyc * nzc]();
  DataType *uPSE_f = new DataType[nxc * nyc * nzc]();
  DataType *uPSW_f = new DataType[nxc * nyc * nzc]();
  DataType *dPC_f = new DataType[nxc * nyc * nzc]();
  DataType *dPN_f = new DataType[nxc * nyc * nzc]();
  DataType *dPS_f = new DataType[nxc * nyc * nzc]();
  DataType *dPE_f = new DataType[nxc * nyc * nzc]();
  DataType *dPW_f = new DataType[nxc * nyc * nzc]();
  DataType *dPNE_f = new DataType[nxc * nyc * nzc]();
  DataType *dPNW_f = new DataType[nxc * nyc * nzc]();
  DataType *dPSE_f = new DataType[nxc * nyc * nzc]();
  DataType *dPSW_f = new DataType[nxc * nyc * nzc]();

  DataType *oPC_c = new DataType[nxc * nyc * nzc]();
  DataType *oPN_c = new DataType[nxc * nyc * nzc]();
  DataType *oPS_c = new DataType[nxc * nyc * nzc]();
  DataType *oPE_c = new DataType[nxc * nyc * nzc]();
  DataType *oPW_c = new DataType[nxc * nyc * nzc]();
  DataType *oPNE_c = new DataType[nxc * nyc * nzc]();
  DataType *oPNW_c = new DataType[nxc * nyc * nzc]();
  DataType *oPSE_c = new DataType[nxc * nyc * nzc]();
  DataType *oPSW_c = new DataType[nxc * nyc * nzc]();
  DataType *uPC_c = new DataType[nxc * nyc * nzc]();
  DataType *uPN_c = new DataType[nxc * nyc * nzc]();
  DataType *uPS_c = new DataType[nxc * nyc * nzc]();
  DataType *uPE_c = new DataType[nxc * nyc * nzc]();
  DataType *uPW_c = new DataType[nxc * nyc * nzc]();
  DataType *uPNE_c = new DataType[nxc * nyc * nzc]();
  DataType *uPNW_c = new DataType[nxc * nyc * nzc]();
  DataType *uPSE_c = new DataType[nxc * nyc * nzc]();
  DataType *uPSW_c = new DataType[nxc * nyc * nzc]();
  DataType *dPC_c = new DataType[nxc * nyc * nzc]();
  DataType *dPN_c = new DataType[nxc * nyc * nzc]();
  DataType *dPS_c = new DataType[nxc * nyc * nzc]();
  DataType *dPE_c = new DataType[nxc * nyc * nzc]();
  DataType *dPW_c = new DataType[nxc * nyc * nzc]();
  DataType *dPNE_c = new DataType[nxc * nyc * nzc]();
  DataType *dPNW_c = new DataType[nxc * nyc * nzc]();
  DataType *dPSE_c = new DataType[nxc * nyc * nzc]();
  DataType *dPSW_c = new DataType[nxc * nyc * nzc]();

  DataType *xf = new DataType[nxf]();
  DataType *yf = new DataType[nyf]();
  DataType *zf = new DataType[nzf]();

  initialize_random(oC, oC, nxf * nyf * nzf);
  initialize_random(oE, oE, nxf * nyf * nzf);
  initialize_random(oN, oN, nxf * nyf * nzf);
  initialize_random(uC, uC, nxf * nyf * nzf);

  buildpb_op7_(&nxf, &nyf, &nzf, &nxc, &nyc, &nzc, nullptr, nullptr, oC, oE, oN,
               uC, oPC_f, oPN_f, oPS_f, oPE_f, oPW_f, oPNE_f, oPNW_f, oPSE_f,
               oPSW_f, uPC_f, uPN_f, uPS_f, uPE_f, uPW_f, uPNE_f, uPNW_f,
               uPSE_f, uPSW_f, dPC_f, dPN_f, dPS_f, dPE_f, dPW_f, dPNE_f,
               dPNW_f, dPSE_f, dPSW_f);

  VbuildPb_op7(&nxf, &nyf, &nzf, &nxc, &nyc, &nzc, nullptr, nullptr, oC, oE, oN,
               uC, oPC_c, oPN_c, oPS_c, oPE_c, oPW_c, oPNE_c, oPNW_c, oPSE_c,
               oPSW_c, uPC_c, uPN_c, uPS_c, uPE_c, uPW_c, uPNE_c, uPNW_c,
               uPSE_c, uPSW_c, dPC_c, dPN_c, dPS_c, dPE_c, dPW_c, dPNE_c,
               dPNW_c, dPSE_c, dPSW_c, q);

  q.wait();

  compare(oPC_c, oPC_f, nxc * nyc * nzc);
  compare(oPN_c, oPN_f, nxc * nyc * nzc);
  compare(oPS_c, oPS_f, nxc * nyc * nzc);
  compare(oPE_c, oPE_f, nxc * nyc * nzc);
  compare(oPW_c, oPW_f, nxc * nyc * nzc);
  compare(oPNE_c, oPNE_f, nxc * nyc * nzc);
  compare(oPNW_c, oPNW_f, nxc * nyc * nzc);
  compare(oPSE_c, oPSE_f, nxc * nyc * nzc);
  compare(oPSW_c, oPSW_f, nxc * nyc * nzc);
  compare(uPC_c, uPC_f, nxc * nyc * nzc);
  compare(uPN_c, uPN_f, nxc * nyc * nzc);
  compare(uPS_c, uPS_f, nxc * nyc * nzc);
  compare(uPE_c, uPE_f, nxc * nyc * nzc);
  compare(uPW_c, uPW_f, nxc * nyc * nzc);
  compare(uPNE_c, uPNE_f, nxc * nyc * nzc);
  compare(uPNW_c, uPNW_f, nxc * nyc * nzc);
  compare(uPSE_c, uPSE_f, nxc * nyc * nzc);
  compare(uPSW_c, uPSW_f, nxc * nyc * nzc);
  compare(dPC_c, dPC_f, nxc * nyc * nzc);
  compare(dPN_c, dPN_f, nxc * nyc * nzc);
  compare(dPS_c, dPS_f, nxc * nyc * nzc);
  compare(dPE_c, dPE_f, nxc * nyc * nzc);
  compare(dPW_c, dPW_f, nxc * nyc * nzc);
  compare(dPNE_c, dPNE_f, nxc * nyc * nzc);
  compare(dPNW_c, dPNW_f, nxc * nyc * nzc);
  compare(dPSE_c, dPSE_f, nxc * nyc * nzc);
  compare(dPSW_c, dPSW_f, nxc * nyc * nzc);

  std::cout << "Hello, World!" << std::endl;
}
