#include <buildGd.h>
#include <buildPd.h>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

using DataType = double;

extern "C" void buildg_27_(
    int *nxf, int *nyf, int *nzf, int *nxc, int *nyc, int *nzc, DataType *oPC,
    DataType *oPN, DataType *oPS, DataType *oPE, DataType *oPW, DataType *oPNE,
    DataType *oPNW, DataType *oPSE, DataType *oPSW, DataType *uPC,
    DataType *uPN, DataType *uPS, DataType *uPE, DataType *uPW, DataType *uPNE,
    DataType *uPNW, DataType *uPSE, DataType *uPSW, DataType *dPC,
    DataType *dPN, DataType *dPS, DataType *dPE, DataType *dPW, DataType *dPNE,
    DataType *dPNW, DataType *dPSE, DataType *dPSW, DataType *oC, DataType *oE,
    DataType *oN, DataType *uC, DataType *oNE, DataType *oNW, DataType *uE,
    DataType *uW, DataType *uN, DataType *uS, DataType *uNE, DataType *uNW,
    DataType *uSE, DataType *uSW, DataType *XoC, DataType *XoE, DataType *XoN,
    DataType *XuC, DataType *XoNE, DataType *XoNW, DataType *XuE, DataType *XuW,
    DataType *XuN, DataType *XuS, DataType *XuNE, DataType *XuNW,
    DataType *XuSE, DataType *XuSW);

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
    assert(std::abs(first[i] - second[i]) < 1e-12);
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

  DataType *XoC_c = new DataType[nxc * nyc * nzc]();
  DataType *XoE_c = new DataType[nxc * nyc * nzc]();
  DataType *XoN_c = new DataType[nxc * nyc * nzc]();
  DataType *XuC_c = new DataType[nxc * nyc * nzc]();
  DataType *XoNE_c = new DataType[nxc * nyc * nzc]();
  DataType *XoNW_c = new DataType[nxc * nyc * nzc]();
  DataType *XuE_c = new DataType[nxc * nyc * nzc]();
  DataType *XuW_c = new DataType[nxc * nyc * nzc]();
  DataType *XuN_c = new DataType[nxc * nyc * nzc]();
  DataType *XuS_c = new DataType[nxc * nyc * nzc]();
  DataType *XuNE_c = new DataType[nxc * nyc * nzc]();
  DataType *XuNW_c = new DataType[nxc * nyc * nzc]();
  DataType *XuSE_c = new DataType[nxc * nyc * nzc]();
  DataType *XuSW_c = new DataType[nxc * nyc * nzc]();

  DataType *XoC_f = new DataType[nxc * nyc * nzc]();
  DataType *XoE_f = new DataType[nxc * nyc * nzc]();
  DataType *XoN_f = new DataType[nxc * nyc * nzc]();
  DataType *XuC_f = new DataType[nxc * nyc * nzc]();
  DataType *XoNE_f = new DataType[nxc * nyc * nzc]();
  DataType *XoNW_f = new DataType[nxc * nyc * nzc]();
  DataType *XuE_f = new DataType[nxc * nyc * nzc]();
  DataType *XuW_f = new DataType[nxc * nyc * nzc]();
  DataType *XuN_f = new DataType[nxc * nyc * nzc]();
  DataType *XuS_f = new DataType[nxc * nyc * nzc]();
  DataType *XuNE_f = new DataType[nxc * nyc * nzc]();
  DataType *XuNW_f = new DataType[nxc * nyc * nzc]();
  DataType *XuSE_f = new DataType[nxc * nyc * nzc]();
  DataType *XuSW_f = new DataType[nxc * nyc * nzc]();

  DataType *xf = new DataType[nxf]();
  DataType *yf = new DataType[nyf]();
  DataType *zf = new DataType[nzf]();

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

  initialize_random(oPC_c, oPC_f, nxc * nyc * nzc);
  initialize_random(oPN_c, oPN_f, nxc * nyc * nzc);
  initialize_random(oPS_c, oPS_f, nxc * nyc * nzc);
  initialize_random(oPE_c, oPE_f, nxc * nyc * nzc);
  initialize_random(oPW_c, oPW_f, nxc * nyc * nzc);
  initialize_random(oPNE_c, oPNE_f, nxc * nyc * nzc);
  initialize_random(oPNW_c, oPNW_f, nxc * nyc * nzc);
  initialize_random(oPSE_c, oPSE_f, nxc * nyc * nzc);
  initialize_random(oPSW_c, oPSW_f, nxc * nyc * nzc);
  initialize_random(uPC_c, uPC_f, nxc * nyc * nzc);
  initialize_random(uPN_c, uPN_f, nxc * nyc * nzc);
  initialize_random(uPS_c, uPS_f, nxc * nyc * nzc);
  initialize_random(uPE_c, uPE_f, nxc * nyc * nzc);
  initialize_random(uPW_c, uPW_f, nxc * nyc * nzc);
  initialize_random(uPNE_c, uPNE_f, nxc * nyc * nzc);
  initialize_random(uPNW_c, uPNW_f, nxc * nyc * nzc);
  initialize_random(uPSE_c, uPSE_f, nxc * nyc * nzc);
  initialize_random(uPSW_c, uPSW_f, nxc * nyc * nzc);
  initialize_random(dPC_c, dPC_f, nxc * nyc * nzc);
  initialize_random(dPN_c, dPN_f, nxc * nyc * nzc);
  initialize_random(dPS_c, dPS_f, nxc * nyc * nzc);
  initialize_random(dPE_c, dPE_f, nxc * nyc * nzc);
  initialize_random(dPW_c, dPW_f, nxc * nyc * nzc);
  initialize_random(dPNE_c, dPNE_f, nxc * nyc * nzc);
  initialize_random(dPNW_c, dPNW_f, nxc * nyc * nzc);
  initialize_random(dPSE_c, dPSE_f, nxc * nyc * nzc);
  initialize_random(dPSW_c, dPSW_f, nxc * nyc * nzc);

  buildg_27_(&nxf, &nyf, &nzf, &nxc, &nyc, &nzc, oPC_f, oPN_f, oPS_f, oPE_f,
             oPW_f, oPNE_f, oPNW_f, oPSE_f, oPSW_f, uPC_f, uPN_f, uPS_f, uPE_f,
             uPW_f, uPNE_f, uPNW_f, uPSE_f, uPSW_f, dPC_f, dPN_f, dPS_f, dPE_f,
             dPW_f, dPNE_f, dPNW_f, dPSE_f, dPSW_f, oC, oE, oN, uC, oNE, oNW,
             uE, uW, uN, uS, uNE, uNW, uSE, uSW, XoC_f, XoE_f, XoN_f, XuC_f,
             XoNE_f, XoNW_f, XuE_f, XuW_f, XuN_f, XuS_f, XuNE_f, XuNW_f, XuSE_f,
             XuSW_f);

  VbuildG_27(&nxf, &nyf, &nzf, &nxc, &nyc, &nzc, oPC_c, oPN_c, oPS_c, oPE_c,
             oPW_c, oPNE_c, oPNW_c, oPSE_c, oPSW_c, uPC_c, uPN_c, uPS_c, uPE_c,
             uPW_c, uPNE_c, uPNW_c, uPSE_c, uPSW_c, dPC_c, dPN_c, dPS_c, dPE_c,
             dPW_c, dPNE_c, dPNW_c, dPSE_c, dPSW_c, oC, oE, oN, uC, oNE, oNW,
             uE, uW, uN, uS, uNE, uNW, uSE, uSW, XoC_c, XoE_c, XoN_c, XuC_c,
             XoNE_c, XoNW_c, XuE_c, XuW_c, XuN_c, XuS_c, XuNE_c, XuNW_c, XuSE_c,
             XuSW_c, q);

  q.wait();

  compare(XoC_c, XoC_f, nxc * nyc * nzc);
  compare(XoE_c, XoE_f, nxc * nyc * nzc);
  compare(XoN_c, XoN_f, nxc * nyc * nzc);
  compare(XuC_c, XuC_f, nxc * nyc * nzc);
  compare(XoNE_c, XoNE_f, nxc * nyc * nzc);
  compare(XoNW_c, XoNW_f, nxc * nyc * nzc);
  compare(XuE_c, XuE_f, nxc * nyc * nzc);
  compare(XuW_c, XuW_f, nxc * nyc * nzc);
  compare(XuN_c, XuN_f, nxc * nyc * nzc);
  compare(XuS_c, XuS_f, nxc * nyc * nzc);
  compare(XuNE_c, XuNE_f, nxc * nyc * nzc);
  compare(XuNW_c, XuNW_f, nxc * nyc * nzc);
  compare(XuSE_c, XuSE_f, nxc * nyc * nzc);
  compare(XuSW_c, XuSW_f, nxc * nyc * nzc);

  std::cout << "Hello, World!" << std::endl;
}
