/**
 *  @ingroup PMGC
 *  @author  Tucker Beck [fortran ->c translation], Michael Holst [original]
 *  @brief
 *  @version $Id:
 *
 *  @attention
 *  @verbatim
 *
 * APBS -- Adaptive Poisson-Boltzmann Solver
 *
 * Nathan A. Baker (nathan.baker@pnl.gov)
 * Pacific Northwest National Laboratory
 *
 * Additional contributing authors listed in the code documentation.
 *
 * Copyright (c) 2010-2020 Battelle Memorial Institute.
 * Developed at the Pacific Northwest National Laboratory, operated by Battelle
 * Memorial Institute, Pacific Northwest Division for the U.S. Department
 * Energy. Portions Copyright (c) 2002-2010, Washington University in St. Louis.
 * Portions Copyright (c) 2002-2020, Nathan A. Baker.
 * Portions Copyright (c) 1999-2002, The Regents of the University of
 * California. Portions Copyright (c) 1995, Michael Holst. All rights reserved.
 *
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * -  Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * - Neither the name of Washington University in St. Louis nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @endverbatim
 */

#include "buildGd.h"
#include <sycl/sycl.hpp>

VPUBLIC void VbuildG(int *nxf, int *nyf, int *nzf, int *nxc, int *nyc, int *nzc,
                     int *numdia, DataType *pcFF, DataType *acFF, DataType *ac, sycl::queue& q) {

  MAT2(pcFF, *nxc * *nyc * *nzc, 27);
  MAT2(acFF, *nxf * *nyf * *nzf, 27);
  MAT2(ac, *nxc * *nyc * *nzc, 27);

  // Call the build routine ***
  if (*numdia == 1) {

    VbuildG_1(

        nxf, nyf, nzf, nxc, nyc, nzc,

        RAT2(pcFF, 1, 1), RAT2(pcFF, 1, 2), RAT2(pcFF, 1, 3), RAT2(pcFF, 1, 4),
        RAT2(pcFF, 1, 5), RAT2(pcFF, 1, 6), RAT2(pcFF, 1, 7), RAT2(pcFF, 1, 8),
        RAT2(pcFF, 1, 9), RAT2(pcFF, 1, 10), RAT2(pcFF, 1, 11),
        RAT2(pcFF, 1, 12), RAT2(pcFF, 1, 13), RAT2(pcFF, 1, 14),
        RAT2(pcFF, 1, 15), RAT2(pcFF, 1, 16), RAT2(pcFF, 1, 17),
        RAT2(pcFF, 1, 18), RAT2(pcFF, 1, 19), RAT2(pcFF, 1, 20),
        RAT2(pcFF, 1, 21), RAT2(pcFF, 1, 22), RAT2(pcFF, 1, 23),
        RAT2(pcFF, 1, 24), RAT2(pcFF, 1, 25), RAT2(pcFF, 1, 26),
        RAT2(pcFF, 1, 27),

        RAT2(acFF, 1, 1),

        RAT2(ac, 1, 1), RAT2(ac, 1, 2), RAT2(ac, 1, 3), RAT2(ac, 1, 4),
        RAT2(ac, 1, 5), RAT2(ac, 1, 6), RAT2(ac, 1, 7), RAT2(ac, 1, 8),
        RAT2(ac, 1, 9), RAT2(ac, 1, 10), RAT2(ac, 1, 11), RAT2(ac, 1, 12),
        RAT2(ac, 1, 13), RAT2(ac, 1, 14), q

    );

  } else if (*numdia == 7) {

    VbuildG_7(

        nxf, nyf, nzf, nxc, nyc, nzc,

        RAT2(pcFF, 1, 1), RAT2(pcFF, 1, 2), RAT2(pcFF, 1, 3), RAT2(pcFF, 1, 4),
        RAT2(pcFF, 1, 5), RAT2(pcFF, 1, 6), RAT2(pcFF, 1, 7), RAT2(pcFF, 1, 8),
        RAT2(pcFF, 1, 9), RAT2(pcFF, 1, 10), RAT2(pcFF, 1, 11),
        RAT2(pcFF, 1, 12), RAT2(pcFF, 1, 13), RAT2(pcFF, 1, 14),
        RAT2(pcFF, 1, 15), RAT2(pcFF, 1, 16), RAT2(pcFF, 1, 17),
        RAT2(pcFF, 1, 18), RAT2(pcFF, 1, 19), RAT2(pcFF, 1, 20),
        RAT2(pcFF, 1, 21), RAT2(pcFF, 1, 22), RAT2(pcFF, 1, 23),
        RAT2(pcFF, 1, 24), RAT2(pcFF, 1, 25), RAT2(pcFF, 1, 26),
        RAT2(pcFF, 1, 27),

        RAT2(acFF, 1, 1), RAT2(acFF, 1, 2), RAT2(acFF, 1, 3), RAT2(acFF, 1, 4),

        RAT2(ac, 1, 1), RAT2(ac, 1, 2), RAT2(ac, 1, 3), RAT2(ac, 1, 4),
        RAT2(ac, 1, 5), RAT2(ac, 1, 6), RAT2(ac, 1, 7), RAT2(ac, 1, 8),
        RAT2(ac, 1, 9), RAT2(ac, 1, 10), RAT2(ac, 1, 11), RAT2(ac, 1, 12),
        RAT2(ac, 1, 13), RAT2(ac, 1, 14), q

    );

  } else if (*numdia == 27) {

    VbuildG_27(

        nxf, nyf, nzf, nxc, nyc, nzc,

        RAT2(pcFF, 1, 1), RAT2(pcFF, 1, 2), RAT2(pcFF, 1, 3), RAT2(pcFF, 1, 4),
        RAT2(pcFF, 1, 5), RAT2(pcFF, 1, 6), RAT2(pcFF, 1, 7), RAT2(pcFF, 1, 8),
        RAT2(pcFF, 1, 9), RAT2(pcFF, 1, 10), RAT2(pcFF, 1, 11),
        RAT2(pcFF, 1, 12), RAT2(pcFF, 1, 13), RAT2(pcFF, 1, 14),
        RAT2(pcFF, 1, 15), RAT2(pcFF, 1, 16), RAT2(pcFF, 1, 17),
        RAT2(pcFF, 1, 18), RAT2(pcFF, 1, 19), RAT2(pcFF, 1, 20),
        RAT2(pcFF, 1, 21), RAT2(pcFF, 1, 22), RAT2(pcFF, 1, 23),
        RAT2(pcFF, 1, 24), RAT2(pcFF, 1, 25), RAT2(pcFF, 1, 26),
        RAT2(pcFF, 1, 27),

        RAT2(acFF, 1, 1), RAT2(acFF, 1, 2), RAT2(acFF, 1, 3), RAT2(acFF, 1, 4),
        RAT2(acFF, 1, 5), RAT2(acFF, 1, 6), RAT2(acFF, 1, 7), RAT2(acFF, 1, 8),
        RAT2(acFF, 1, 9), RAT2(acFF, 1, 10), RAT2(acFF, 1, 11),
        RAT2(acFF, 1, 12), RAT2(acFF, 1, 13), RAT2(acFF, 1, 14),

        RAT2(ac, 1, 1), RAT2(ac, 1, 2), RAT2(ac, 1, 3), RAT2(ac, 1, 4),
        RAT2(ac, 1, 5), RAT2(ac, 1, 6), RAT2(ac, 1, 7), RAT2(ac, 1, 8),
        RAT2(ac, 1, 9), RAT2(ac, 1, 10), RAT2(ac, 1, 11), RAT2(ac, 1, 12),
        RAT2(ac, 1, 13), RAT2(ac, 1, 14), q

    );

  } else {
    printf("BUILDG: invalid stencil type given...\n");
  }
}

VPUBLIC void VbuildG_1(int *nxf, int *nyf, int *nzf, int *nx, int *ny, int *nz,
                       DataType *oPC, DataType *oPN, DataType *oPS, DataType *oPE,
                       DataType *oPW, DataType *oPNE, DataType *oPNW, DataType *oPSE,
                       DataType *oPSW, DataType *uPC, DataType *uPN, DataType *uPS,
                       DataType *uPE, DataType *uPW, DataType *uPNE, DataType *uPNW,
                       DataType *uPSE, DataType *uPSW, DataType *dPC, DataType *dPN,
                       DataType *dPS, DataType *dPE, DataType *dPW, DataType *dPNE,
                       DataType *dPNW, DataType *dPSE, DataType *dPSW, DataType *oC,
                       DataType *XoC, DataType *XoE, DataType *XoN, DataType *XuC,
                       DataType *XoNE, DataType *XoNW, DataType *XuE, DataType *XuW,
                       DataType *XuN, DataType *XuS, DataType *XuNE, DataType *XuNW,
                       DataType *XuSE, DataType *XuSW, sycl::queue& q) {

  int i, j, k, ii, jj, kk;
  int im1, ip1, jm1, jp1, km1, kp1;
  int iim1, iip1, jjm1, jjp1, kkm1, kkp1;
  int nxm1, nym1, nzm1;

  DataType TMP1_XOC, TMP2_XOC, TMP3_XOC;

  MAT3(oC, *nxf, *nyf, *nzf);

  MAT3(XoC, *nx, *ny, *nz);
  MAT3(XoE, *nx, *ny, *nz);
  MAT3(XoN, *nx, *ny, *nz);

  MAT3(XuC, *nx, *ny, *nz);

  MAT3(XoNE, *nx, *ny, *nz);
  MAT3(XoNW, *nx, *ny, *nz);

  MAT3(XuE, *nx, *ny, *nz);
  MAT3(XuW, *nx, *ny, *nz);
  MAT3(XuN, *nx, *ny, *nz);
  MAT3(XuS, *nx, *ny, *nz);
  MAT3(XuNE, *nx, *ny, *nz);
  MAT3(XuNW, *nx, *ny, *nz);
  MAT3(XuSE, *nx, *ny, *nz);
  MAT3(XuSW, *nx, *ny, *nz);

  MAT3(oPC, *nx, *ny, *nz);
  MAT3(oPN, *nx, *ny, *nz);
  MAT3(oPS, *nx, *ny, *nz);
  MAT3(oPE, *nx, *ny, *nz);
  MAT3(oPW, *nx, *ny, *nz);
  MAT3(oPNE, *nx, *ny, *nz);
  MAT3(oPNW, *nx, *ny, *nz);
  MAT3(oPSE, *nx, *ny, *nz);
  MAT3(oPSW, *nx, *ny, *nz);

  MAT3(uPC, *nx, *ny, *nz);
  MAT3(uPN, *nx, *ny, *nz);
  MAT3(uPS, *nx, *ny, *nz);
  MAT3(uPE, *nx, *ny, *nz);
  MAT3(uPW, *nx, *ny, *nz);
  MAT3(uPNE, *nx, *ny, *nz);
  MAT3(uPNW, *nx, *ny, *nz);
  MAT3(uPSE, *nx, *ny, *nz);
  MAT3(uPSW, *nx, *ny, *nz);

  MAT3(dPC, *nx, *ny, *nz);
  MAT3(dPN, *nx, *ny, *nz);
  MAT3(dPS, *nx, *ny, *nz);
  MAT3(dPE, *nx, *ny, *nz);
  MAT3(dPW, *nx, *ny, *nz);
  MAT3(dPNE, *nx, *ny, *nz);
  MAT3(dPNW, *nx, *ny, *nz);
  MAT3(dPSE, *nx, *ny, *nz);
  MAT3(dPSW, *nx, *ny, *nz);

  // Define n and determine number of mesh points
  nxm1 = *nx - 1;
  nym1 = *ny - 1;
  nzm1 = *nz - 1;

  // fprintf(data, "%s\n", PRINT_FUNC);

  // Build the operator
      q.parallel_for(sycl::range<3>(*nx - 2, *ny - 2, *nz - 2), [=](sycl::id<3> I) { 
        const int ii = I[0] + 2; // Adjust for 1-based indexing
        const int jj = I[1] + 2; // Adjust for 1-based indexing
        const int kk = I[2] + 2; // Adjust for 1-based indexing

        const int k = 2 * kk - 1;
        const int j = 2 * jj - 1;

        const int i = 2 * ii - 1;

        // Index computations
        const int im1 = i - 1;
        const int ip1 = i + 1;
        const int jm1 = j - 1;
        const int jp1 = j + 1;
        const int km1 = k - 1;
        const int kp1 = k + 1;
        const int iim1 = ii - 1;
        const int iip1 = ii + 1;
        const int jjm1 = jj - 1;
        const int jjp1 = jj + 1;
        const int kkm1 = kk - 1;
        const int kkp1 = kk + 1;

        /* *************************************************************
         * oC
         * *************************************************************/

        // XoC(ii,jj,kk) =
        const auto TMP1_XOC =
            VAT3(uPS, ii, jj, kk) * VAT3(uPS, ii, jj, kk) *
                VAT3(oC, i, jm1, kp1) +
            VAT3(dPSW, ii, jj, kk) * VAT3(dPSW, ii, jj, kk) *
                VAT3(oC, im1, jm1, km1) +
            VAT3(oPSW, ii, jj, kk) * VAT3(oPSW, ii, jj, kk) *
                VAT3(oC, im1, jm1, k) +
            VAT3(uPSW, ii, jj, kk) * VAT3(uPSW, ii, jj, kk) *
                VAT3(oC, im1, jm1, kp1) +
            VAT3(dPW, ii, jj, kk) * VAT3(dPW, ii, jj, kk) *
                VAT3(oC, im1, j, km1) +
            VAT3(oPW, ii, jj, kk) * VAT3(oPW, ii, jj, kk) *
                VAT3(oC, im1, j, k) +
            VAT3(uPNW, ii, jj, kk) * VAT3(uPNW, ii, jj, kk) *
                VAT3(oC, im1, jp1, kp1) +
            VAT3(dPS, ii, jj, kk) * VAT3(dPS, ii, jj, kk) *
                VAT3(oC, i, jm1, km1) +
            VAT3(oPS, ii, jj, kk) * VAT3(oPS, ii, jj, kk) * VAT3(oC, i, jm1, k);

        const auto TMP2_XOC =
            VAT3(dPC, ii, jj, kk) * VAT3(dPC, ii, jj, kk) *
                VAT3(oC, i, j, km1) +
            VAT3(oPC, ii, jj, kk) * VAT3(oPC, ii, jj, kk) * VAT3(oC, i, j, k) +
            VAT3(uPC, ii, jj, kk) * VAT3(uPC, ii, jj, kk) *
                VAT3(oC, i, j, kp1) +
            VAT3(dPN, ii, jj, kk) * VAT3(dPN, ii, jj, kk) *
                VAT3(oC, i, jp1, km1) +
            VAT3(oPN, ii, jj, kk) * VAT3(oPN, ii, jj, kk) *
                VAT3(oC, i, jp1, k) +
            VAT3(uPW, ii, jj, kk) * VAT3(uPW, ii, jj, kk) *
                VAT3(oC, im1, j, kp1) +
            VAT3(dPNW, ii, jj, kk) * VAT3(dPNW, ii, jj, kk) *
                VAT3(oC, im1, jp1, km1) +
            VAT3(oPNW, ii, jj, kk) * VAT3(oPNW, ii, jj, kk) *
                VAT3(oC, im1, jp1, k) +
            VAT3(oPE, ii, jj, kk) * VAT3(oPE, ii, jj, kk) * VAT3(oC, ip1, j, k);

        const auto TMP3_XOC = VAT3(uPE, ii, jj, kk) * VAT3(uPE, ii, jj, kk) *
                       VAT3(oC, ip1, j, kp1) +
                   VAT3(dPNE, ii, jj, kk) * VAT3(dPNE, ii, jj, kk) *
                       VAT3(oC, ip1, jp1, km1) +
                   VAT3(oPNE, ii, jj, kk) * VAT3(oPNE, ii, jj, kk) *
                       VAT3(oC, ip1, jp1, k) +
                   VAT3(uPNE, ii, jj, kk) * VAT3(uPNE, ii, jj, kk) *
                       VAT3(oC, ip1, jp1, kp1) +
                   VAT3(uPN, ii, jj, kk) * VAT3(uPN, ii, jj, kk) *
                       VAT3(oC, i, jp1, kp1) +
                   VAT3(dPSE, ii, jj, kk) * VAT3(dPSE, ii, jj, kk) *
                       VAT3(oC, ip1, jm1, km1) +
                   VAT3(oPSE, ii, jj, kk) * VAT3(oPSE, ii, jj, kk) *
                       VAT3(oC, ip1, jm1, k) +
                   VAT3(uPSE, ii, jj, kk) * VAT3(uPSE, ii, jj, kk) *
                       VAT3(oC, ip1, jm1, kp1) +
                   VAT3(dPE, ii, jj, kk) * VAT3(dPE, ii, jj, kk) *
                       VAT3(oC, ip1, j, km1);

        VAT3(XoC, ii, jj, kk) = TMP1_XOC + TMP2_XOC + TMP3_XOC;

        // fprintf(data, "%19.12E\n", VAT3(XoC, ii, jj, kk));

        /* **************************************************************
         * *** > oE;
         * **************************************************************/

        VAT3(XoE, ii, jj, kk) =
            -VAT3(dPSE, ii, jj, kk) * VAT3(oC, ip1, jm1, km1) *
                VAT3(dPSW, iip1, jj, kk) -
            VAT3(oPSE, ii, jj, kk) * VAT3(oC, ip1, jm1, k) *
                VAT3(oPSW, iip1, jj, kk) -
            VAT3(uPSE, ii, jj, kk) * VAT3(oC, ip1, jm1, kp1) *
                VAT3(uPSW, iip1, jj, kk) -
            VAT3(dPE, ii, jj, kk) * VAT3(oC, ip1, j, km1) *
                VAT3(dPW, iip1, jj, kk) -
            VAT3(oPE, ii, jj, kk) * VAT3(oC, ip1, j, k) *
                VAT3(oPW, iip1, jj, kk) -
            VAT3(uPE, ii, jj, kk) * VAT3(oC, ip1, j, kp1) *
                VAT3(uPW, iip1, jj, kk) -
            VAT3(dPNE, ii, jj, kk) * VAT3(oC, ip1, jp1, km1) *
                VAT3(dPNW, iip1, jj, kk) -
            VAT3(oPNE, ii, jj, kk) * VAT3(oC, ip1, jp1, k) *
                VAT3(oPNW, iip1, jj, kk) -
            VAT3(uPNE, ii, jj, kk) * VAT3(oC, ip1, jp1, kp1) *
                VAT3(uPNW, iip1, jj, kk);

        // fprintf(data, "%19.12E\n", VAT3(XoE, ii, jj, kk));

        /* **************************************************************
         * *** > oN;
         * **************************************************************/

        VAT3(XoN, ii, jj, kk) =
            -VAT3(dPNW, ii, jj, kk) * VAT3(oC, im1, jp1, km1) *
                VAT3(dPSW, ii, jjp1, kk) -
            VAT3(oPNW, ii, jj, kk) * VAT3(oC, im1, jp1, k) *
                VAT3(oPSW, ii, jjp1, kk) -
            VAT3(uPNW, ii, jj, kk) * VAT3(oC, im1, jp1, kp1) *
                VAT3(uPSW, ii, jjp1, kk) -
            VAT3(dPN, ii, jj, kk) * VAT3(oC, i, jp1, km1) *
                VAT3(dPS, ii, jjp1, kk) -
            VAT3(oPN, ii, jj, kk) * VAT3(oC, i, jp1, k) *
                VAT3(oPS, ii, jjp1, kk) -
            VAT3(uPN, ii, jj, kk) * VAT3(oC, i, jp1, kp1) *
                VAT3(uPS, ii, jjp1, kk) -
            VAT3(dPNE, ii, jj, kk) * VAT3(oC, ip1, jp1, km1) *
                VAT3(dPSE, ii, jjp1, kk) -
            VAT3(oPNE, ii, jj, kk) * VAT3(oC, ip1, jp1, k) *
                VAT3(oPSE, ii, jjp1, kk) -
            VAT3(uPNE, ii, jj, kk) * VAT3(oC, ip1, jp1, kp1) *
                VAT3(uPSE, ii, jjp1, kk);

        // fprintf(data, "%19.12E\n", VAT3(XoN, ii, jj, kk));

        /* **************************************************************
         * *** > uC;
         * **************************************************************/

        VAT3(XuC, ii, jj, kk) =
            -VAT3(dPSW, ii, jj, kkp1) * VAT3(oC, im1, jm1, kp1) *
                VAT3(uPSW, ii, jj, kk) -
            VAT3(dPW, ii, jj, kkp1) * VAT3(oC, im1, j, kp1) *
                VAT3(uPW, ii, jj, kk) -
            VAT3(dPNW, ii, jj, kkp1) * VAT3(oC, im1, jp1, kp1) *
                VAT3(uPNW, ii, jj, kk) -
            VAT3(dPS, ii, jj, kkp1) * VAT3(oC, i, jm1, kp1) *
                VAT3(uPS, ii, jj, kk) -
            VAT3(dPC, ii, jj, kkp1) * VAT3(oC, i, j, kp1) *
                VAT3(uPC, ii, jj, kk) -
            VAT3(dPN, ii, jj, kkp1) * VAT3(oC, i, jp1, kp1) *
                VAT3(uPN, ii, jj, kk) -
            VAT3(dPSE, ii, jj, kkp1) * VAT3(oC, ip1, jm1, kp1) *
                VAT3(uPSE, ii, jj, kk) -
            VAT3(dPE, ii, jj, kkp1) * VAT3(oC, ip1, j, kp1) *
                VAT3(uPE, ii, jj, kk) -
            VAT3(dPNE, ii, jj, kkp1) * VAT3(oC, ip1, jp1, kp1) *
                VAT3(uPNE, ii, jj, kk);

        // fprintf(data, "%19.12E\n", VAT3(XuC, ii, jj, kk));

        /* **************************************************************
         * *** > oNE;
         * **************************************************************/

        VAT3(XoNE, ii, jj, kk) =
            -VAT3(dPNE, ii, jj, kk) * VAT3(oC, ip1, jp1, km1) *
                VAT3(dPSW, iip1, jjp1, kk) -
            VAT3(oPNE, ii, jj, kk) * VAT3(oC, ip1, jp1, k) *
                VAT3(oPSW, iip1, jjp1, kk) -
            VAT3(uPNE, ii, jj, kk) * VAT3(oC, ip1, jp1, kp1) *
                VAT3(uPSW, iip1, jjp1, kk);

        // fprintf(data, "%19.12E\n", VAT3(XoNE, ii, jj, kk));

        /* **************************************************************
         * *** > oNW;
         * **************************************************************/

        VAT3(XoNW, ii, jj, kk) =
            -VAT3(dPNW, ii, jj, kk) * VAT3(oC, im1, jp1, km1) *
                VAT3(dPSE, iim1, jjp1, kk) -
            VAT3(oPNW, ii, jj, kk) * VAT3(oC, im1, jp1, k) *
                VAT3(oPSE, iim1, jjp1, kk) -
            VAT3(uPNW, ii, jj, kk) * VAT3(oC, im1, jp1, kp1) *
                VAT3(uPSE, iim1, jjp1, kk);

        // fprintf(data, "%19.12E\n", VAT3(XoNW, ii, jj, kk));

        /* **************************************************************
         * *** > uE;
         * **************************************************************/

        VAT3(XuE, ii, jj, kk) =
            -VAT3(uPSE, ii, jj, kk) * VAT3(oC, ip1, jm1, kp1) *
                VAT3(dPSW, iip1, jj, kkp1) -
            VAT3(uPE, ii, jj, kk) * VAT3(oC, ip1, j, kp1) *
                VAT3(dPW, iip1, jj, kkp1) -
            VAT3(uPNE, ii, jj, kk) * VAT3(oC, ip1, jp1, kp1) *
                VAT3(dPNW, iip1, jj, kkp1);

        // fprintf(data, "%19.12E\n", VAT3(XuE, ii, jj, kk));

        /* **************************************************************
         * *** > uW;
         * **************************************************************/

        VAT3(XuW, ii, jj, kk) =
            -VAT3(uPSW, ii, jj, kk) * VAT3(oC, im1, jm1, kp1) *
                VAT3(dPSE, iim1, jj, kkp1) -
            VAT3(uPW, ii, jj, kk) * VAT3(oC, im1, j, kp1) *
                VAT3(dPE, iim1, jj, kkp1) -
            VAT3(uPNW, ii, jj, kk) * VAT3(oC, im1, jp1, kp1) *
                VAT3(dPNE, iim1, jj, kkp1);

        // fprintf(data, "%19.12E\n", VAT3(XuW, ii, jj, kk));

        /* **************************************************************
         * *** > uN;
         * **************************************************************/

        VAT3(XuN, ii, jj, kk) =
            -VAT3(uPNW, ii, jj, kk) * VAT3(oC, im1, jp1, kp1) *
                VAT3(dPSW, ii, jjp1, kkp1) -
            VAT3(uPN, ii, jj, kk) * VAT3(oC, i, jp1, kp1) *
                VAT3(dPS, ii, jjp1, kkp1) -
            VAT3(uPNE, ii, jj, kk) * VAT3(oC, ip1, jp1, kp1) *
                VAT3(dPSE, ii, jjp1, kkp1);

        // fprintf(data, "%19.12E\n", VAT3(XuN, ii, jj, kk));

        /* **************************************************************
         * *** > uS;
         * **************************************************************/

        VAT3(XuS, ii, jj, kk) =
            -VAT3(uPSW, ii, jj, kk) * VAT3(oC, im1, jm1, kp1) *
                VAT3(dPNW, ii, jjm1, kkp1) -
            VAT3(uPS, ii, jj, kk) * VAT3(oC, i, jm1, kp1) *
                VAT3(dPN, ii, jjm1, kkp1) -
            VAT3(uPSE, ii, jj, kk) * VAT3(oC, ip1, jm1, kp1) *
                VAT3(dPNE, ii, jjm1, kkp1);

        // fprintf(data, "%19.12E\n", VAT3(XuS, ii, jj, kk));

        /* **************************************************************
         * *** > uNE;
         * **************************************************************/

        VAT3(XuNE, ii, jj, kk) = -VAT3(uPNE, ii, jj, kk) *
                                 VAT3(oC, ip1, jp1, kp1) *
                                 VAT3(dPSW, iip1, jjp1, kkp1);

        // fprintf(data, "%19.12E\n", VAT3(XuNE, ii, jj, kk));

        /* **************************************************************
         * *** > uNW;
         * **************************************************************/

        VAT3(XuNW, ii, jj, kk) = -VAT3(uPNW, ii, jj, kk) *
                                 VAT3(oC, im1, jp1, kp1) *
                                 VAT3(dPSE, iim1, jjp1, kkp1);

        // fprintf(data, "%19.12E\n", VAT3(XuNW, ii, jj, kk));

        /* **************************************************************
         * *** > uSE;
         * **************************************************************/

        VAT3(XuSE, ii, jj, kk) = -VAT3(uPSE, ii, jj, kk) *
                                 VAT3(oC, ip1, jm1, kp1) *
                                 VAT3(dPNW, iip1, jjm1, kkp1);

        // fprintf(data, "%19.12E\n", VAT3(XuSE, ii, jj, kk));

        /* **************************************************************
         * *** > uSW;
         * **************************************************************/

        VAT3(XuSW, ii, jj, kk) = -VAT3(uPSW, ii, jj, kk) *
                                 VAT3(oC, im1, jm1, kp1) *
                                 VAT3(dPNE, iim1, jjm1, kkp1);

        // fprintf(data, "%19.12E\n", VAT3(XuSW, ii, jj, kk));
      });
}

VPUBLIC void
VbuildG_7(int *nxf, int *nyf, int *nzf, int *nx, int *ny, int *nz, DataType *oPC,
          DataType *oPN, DataType *oPS, DataType *oPE, DataType *oPW, DataType *oPNE,
          DataType *oPNW, DataType *oPSE, DataType *oPSW, DataType *uPC, DataType *uPN,
          DataType *uPS, DataType *uPE, DataType *uPW, DataType *uPNE, DataType *uPNW,
          DataType *uPSE, DataType *uPSW, DataType *dPC, DataType *dPN, DataType *dPS,
          DataType *dPE, DataType *dPW, DataType *dPNE, DataType *dPNW, DataType *dPSE,
          DataType *dPSW, DataType *oC, DataType *oE, DataType *oN, DataType *uC,
          DataType *XoC, DataType *XoE, DataType *XoN, DataType *XuC, DataType *XoNE,
          DataType *XoNW, DataType *XuE, DataType *XuW, DataType *XuN, DataType *XuS,
          DataType *XuNE, DataType *XuNW, DataType *XuSE, DataType *XuSW, sycl::queue& q) {

  int i, j, k;
  int ii, jj, kk;
  int im1, ip1, im2, ip2;
  int jm1, jp1, jm2, jp2;
  int km1, kp1, km2, kp2;
  int iim1, iip1;
  int jjm1, jjp1;
  int kkm1, kkp1;
  int nxm1, nym1, nzm1;

  DataType TMP1_XOC, TMP2_XOC, TMP3_XOC, TMP4_XOC;
  DataType TMP5_XOC, TMP6_XOC, TMP7_XOC, TMP8_XOC;
  DataType TMP9_XOC;
  DataType TMP1_XOE, TMP2_XOE, TMP3_XOE, TMP4_XOE;
  DataType TMP1_XON, TMP2_XON, TMP3_XON, TMP4_XON;
  DataType TMP1_XUC, TMP2_XUC, TMP3_XUC, TMP4_XUC;

  MAT3(oC, *nxf, *nyf, *nzf);
  MAT3(oE, *nxf, *nyf, *nzf);
  MAT3(oN, *nxf, *nyf, *nzf);

  MAT3(uC, *nxf, *nyf, *nzf);

  MAT3(XoC, *nx, *ny, *nz);
  MAT3(XoE, *nx, *ny, *nz);
  MAT3(XoN, *nx, *ny, *nz);
  MAT3(XoNE, *nx, *ny, *nz);
  MAT3(XoNW, *nx, *ny, *nz);

  MAT3(XuC, *nx, *ny, *nz);
  MAT3(XuE, *nx, *ny, *nz);
  MAT3(XuW, *nx, *ny, *nz);
  MAT3(XuN, *nx, *ny, *nz);
  MAT3(XuS, *nx, *ny, *nz);
  MAT3(XuNE, *nx, *ny, *nz);
  MAT3(XuNW, *nx, *ny, *nz);
  MAT3(XuSE, *nx, *ny, *nz);
  MAT3(XuSW, *nx, *ny, *nz);

  MAT3(oPC, *nx, *ny, *nz);
  MAT3(oPN, *nx, *ny, *nz);
  MAT3(oPS, *nx, *ny, *nz);
  MAT3(oPE, *nx, *ny, *nz);
  MAT3(oPW, *nx, *ny, *nz);
  MAT3(oPNE, *nx, *ny, *nz);
  MAT3(oPNW, *nx, *ny, *nz);
  MAT3(oPSE, *nx, *ny, *nz);
  MAT3(oPSW, *nx, *ny, *nz);

  MAT3(uPC, *nx, *ny, *nz);
  MAT3(uPN, *nx, *ny, *nz);
  MAT3(uPS, *nx, *ny, *nz);
  MAT3(uPE, *nx, *ny, *nz);
  MAT3(uPW, *nx, *ny, *nz);
  MAT3(uPNE, *nx, *ny, *nz);
  MAT3(uPNW, *nx, *ny, *nz);
  MAT3(uPSE, *nx, *ny, *nz);
  MAT3(uPSW, *nx, *ny, *nz);

  MAT3(dPC, *nx, *ny, *nz);
  MAT3(dPN, *nx, *ny, *nz);
  MAT3(dPS, *nx, *ny, *nz);
  MAT3(dPE, *nx, *ny, *nz);
  MAT3(dPW, *nx, *ny, *nz);
  MAT3(dPNE, *nx, *ny, *nz);
  MAT3(dPNW, *nx, *ny, *nz);
  MAT3(dPSE, *nx, *ny, *nz);
  MAT3(dPSW, *nx, *ny, *nz);

  // Define n and determine number of mesh points
  nxm1 = *nx - 1;
  nym1 = *ny - 1;
  nzm1 = *nz - 1;

  // fprintf(data, "%s\n", PRINT_FUNC);

  // Build the operator ***
      q.parallel_for(sycl::range<3>(*nx-2, *ny-2, *nz-2), [=](sycl::id<3> I) {
        const int ii = I[0] + 2; // Adjust for 1-based indexing
        const int jj = I[1] + 2; // Adjust for 1-based indexing
        const int kk = I[2] + 2; // Adjust for 1-based indexing
        const int k = 2 * kk - 1;
        const int j = 2 * jj - 1;       
        const int i = 2 * ii - 1;

        // Index computations
      const int  im1 = i - 1;
      const int  ip1 = i + 1;
      const int  im2 = i - 2;
      const int  ip2 = i + 2;
      const int  jm1 = j - 1;
      const int  jp1 = j + 1;
      const int  jm2 = j - 2;
      const int  jp2 = j + 2;
      const int  km1 = k - 1;
      const int  kp1 = k + 1;
      const int  km2 = k - 2;
      const int  kp2 = k + 2;
      const int  iim1 = ii - 1;
      const int  iip1 = ii + 1;
      const int  jjm1 = jj - 1;
      const int  jjp1 = jj + 1;
      const int  kkm1 = kk - 1;
      const int  kkp1 = kk + 1;

        /* *************************************************************
         * *** > oC;
         * *************************************************************/

        // XoC(ii,jj,kk) =
        const auto TMP1_XOC = VAT3(dPSW, ii, jj, kk) *
                       (VAT3(oC, im1, jm1, km1) * VAT3(dPSW, ii, jj, kk) -
                        VAT3(uC, im1, jm1, km1) * VAT3(oPSW, ii, jj, kk) -
                        VAT3(oN, im1, jm1, km1) * VAT3(dPW, ii, jj, kk) -
                        VAT3(oE, im1, jm1, km1) * VAT3(dPS, ii, jj, kk))

                   + VAT3(oPNE, ii, jj, kk) *
                         (-VAT3(oE, i, jp1, k) * VAT3(oPN, ii, jj, kk) -
                          VAT3(oN, ip1, j, k) * VAT3(oPE, ii, jj, kk) -
                          VAT3(uC, ip1, jp1, km1) * VAT3(dPNE, ii, jj, kk) +
                          VAT3(oC, ip1, jp1, k) * VAT3(oPNE, ii, jj, kk) -
                          VAT3(uC, ip1, jp1, k) * VAT3(uPNE, ii, jj, kk))

                   + VAT3(dPNE, ii, jj, kk) *
                         (-VAT3(oE, i, jp1, km1) * VAT3(dPN, ii, jj, kk) -
                          VAT3(oN, ip1, j, km1) * VAT3(dPE, ii, jj, kk) +
                          VAT3(oC, ip1, jp1, km1) * VAT3(dPNE, ii, jj, kk) -
                          VAT3(uC, ip1, jp1, km1) * VAT3(oPNE, ii, jj, kk));

        const auto TMP2_XOC = VAT3(dPSE, ii, jj, kk) *
                       (-VAT3(oE, i, jm1, km1) * VAT3(dPS, ii, jj, kk) +
                        VAT3(oC, ip1, jm1, km1) * VAT3(dPSE, ii, jj, kk) -
                        VAT3(uC, ip1, jm1, km1) * VAT3(oPSE, ii, jj, kk) -
                        VAT3(oN, ip1, jm1, km1) * VAT3(dPE, ii, jj, kk))

                   + VAT3(uPE, ii, jj, kk) *
                         (-VAT3(oE, i, j, kp1) * VAT3(uPC, ii, jj, kk) -
                          VAT3(oN, ip1, jm1, kp1) * VAT3(uPSE, ii, jj, kk) -
                          VAT3(uC, ip1, j, k) * VAT3(oPE, ii, jj, kk) +
                          VAT3(oC, ip1, j, kp1) * VAT3(uPE, ii, jj, kk) -
                          VAT3(oN, ip1, j, kp1) * VAT3(uPNE, ii, jj, kk))

                   + VAT3(oPE, ii, jj, kk) *
                         (-VAT3(oE, i, j, k) * VAT3(oPC, ii, jj, kk) -
                          VAT3(oN, ip1, jm1, k) * VAT3(oPSE, ii, jj, kk) -
                          VAT3(uC, ip1, j, km1) * VAT3(dPE, ii, jj, kk) +
                          VAT3(oC, ip1, j, k) * VAT3(oPE, ii, jj, kk) -
                          VAT3(uC, ip1, j, k) * VAT3(uPE, ii, jj, kk) -
                          VAT3(oN, ip1, j, k) * VAT3(oPNE, ii, jj, kk));

        const auto TMP3_XOC = +VAT3(dPE, ii, jj, kk) *
                       (-VAT3(oE, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                        VAT3(oN, ip1, jm1, km1) * VAT3(dPSE, ii, jj, kk) +
                        VAT3(oC, ip1, j, km1) * VAT3(dPE, ii, jj, kk) -
                        VAT3(uC, ip1, j, km1) * VAT3(oPE, ii, jj, kk) -
                        VAT3(oN, ip1, j, km1) * VAT3(dPNE, ii, jj, kk))

                   + VAT3(uPSE, ii, jj, kk) *
                         (-VAT3(oE, i, jm1, kp1) * VAT3(uPS, ii, jj, kk) -
                          VAT3(uC, ip1, jm1, k) * VAT3(oPSE, ii, jj, kk) +
                          VAT3(oC, ip1, jm1, kp1) * VAT3(uPSE, ii, jj, kk) -
                          VAT3(oN, ip1, jm1, kp1) * VAT3(uPE, ii, jj, kk))

                   + VAT3(uPNE, ii, jj, kk) *
                         (-VAT3(oE, i, jp1, kp1) * VAT3(uPN, ii, jj, kk) -
                          VAT3(oN, ip1, j, kp1) * VAT3(uPE, ii, jj, kk) -
                          VAT3(uC, ip1, jp1, k) * VAT3(oPNE, ii, jj, kk) +
                          VAT3(oC, ip1, jp1, kp1) * VAT3(uPNE, ii, jj, kk));

        const auto TMP4_XOC = +VAT3(oPS, ii, jj, kk) *
                       (-VAT3(oE, im1, jm1, k) * VAT3(oPSW, ii, jj, kk) -
                        VAT3(uC, i, jm1, km1) * VAT3(dPS, ii, jj, kk) +
                        VAT3(oC, i, jm1, k) * VAT3(oPS, ii, jj, kk) -
                        VAT3(uC, i, jm1, k) * VAT3(uPS, ii, jj, kk) -
                        VAT3(oN, i, jm1, k) * VAT3(oPC, ii, jj, kk) -
                        VAT3(oE, i, jm1, k) * VAT3(oPSE, ii, jj, kk))

                   + VAT3(dPS, ii, jj, kk) *
                         (-VAT3(oE, im1, jm1, km1) * VAT3(dPSW, ii, jj, kk) +
                          VAT3(oC, i, jm1, km1) * VAT3(dPS, ii, jj, kk) -
                          VAT3(uC, i, jm1, km1) * VAT3(oPS, ii, jj, kk) -
                          VAT3(oN, i, jm1, km1) * VAT3(dPC, ii, jj, kk) -
                          VAT3(oE, i, jm1, km1) * VAT3(dPSE, ii, jj, kk))

                   + VAT3(oPSE, ii, jj, kk) *
                         (-VAT3(oE, i, jm1, k) * VAT3(oPS, ii, jj, kk) -
                          VAT3(uC, ip1, jm1, km1) * VAT3(dPSE, ii, jj, kk) +
                          VAT3(oC, ip1, jm1, k) * VAT3(oPSE, ii, jj, kk) -
                          VAT3(uC, ip1, jm1, k) * VAT3(uPSE, ii, jj, kk) -
                          VAT3(oN, ip1, jm1, k) * VAT3(oPE, ii, jj, kk));

        const auto TMP5_XOC = +VAT3(dPN, ii, jj, kk) *
                       (-VAT3(oE, im1, jp1, km1) * VAT3(dPNW, ii, jj, kk) -
                        VAT3(oN, i, j, km1) * VAT3(dPC, ii, jj, kk) +
                        VAT3(oC, i, jp1, km1) * VAT3(dPN, ii, jj, kk) -
                        VAT3(uC, i, jp1, km1) * VAT3(oPN, ii, jj, kk) -
                        VAT3(oE, i, jp1, km1) * VAT3(dPNE, ii, jj, kk))

                   + VAT3(uPC, ii, jj, kk) *
                         (-VAT3(oE, im1, j, kp1) * VAT3(uPW, ii, jj, kk) -
                          VAT3(oN, i, jm1, kp1) * VAT3(uPS, ii, jj, kk) -
                          VAT3(uC, i, j, k) * VAT3(oPC, ii, jj, kk) +
                          VAT3(oC, i, j, kp1) * VAT3(uPC, ii, jj, kk) -
                          VAT3(oN, i, j, kp1) * VAT3(uPN, ii, jj, kk) -
                          VAT3(oE, i, j, kp1) * VAT3(uPE, ii, jj, kk))

                   + VAT3(oPC, ii, jj, kk) *
                         (-VAT3(oE, im1, j, k) * VAT3(oPW, ii, jj, kk) -
                          VAT3(oN, i, jm1, k) * VAT3(oPS, ii, jj, kk) -
                          VAT3(uC, i, j, km1) * VAT3(dPC, ii, jj, kk) +
                          VAT3(oC, i, j, k) * VAT3(oPC, ii, jj, kk) -
                          VAT3(uC, i, j, k) * VAT3(uPC, ii, jj, kk) -
                          VAT3(oN, i, j, k) * VAT3(oPN, ii, jj, kk) -
                          VAT3(oE, i, j, k) * VAT3(oPE, ii, jj, kk));

        const auto TMP6_XOC = +VAT3(dPC, ii, jj, kk) *
                       (-VAT3(oE, im1, j, km1) * VAT3(dPW, ii, jj, kk) -
                        VAT3(oN, i, jm1, km1) * VAT3(dPS, ii, jj, kk) +
                        VAT3(oC, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                        VAT3(uC, i, j, km1) * VAT3(oPC, ii, jj, kk) -
                        VAT3(oN, i, j, km1) * VAT3(dPN, ii, jj, kk) -
                        VAT3(oE, i, j, km1) * VAT3(dPE, ii, jj, kk))

                   + VAT3(uPS, ii, jj, kk) *
                         (-VAT3(oE, im1, jm1, kp1) * VAT3(uPSW, ii, jj, kk) -
                          VAT3(uC, i, jm1, k) * VAT3(oPS, ii, jj, kk) +
                          VAT3(oC, i, jm1, kp1) * VAT3(uPS, ii, jj, kk) -
                          VAT3(oN, i, jm1, kp1) * VAT3(uPC, ii, jj, kk) -
                          VAT3(oE, i, jm1, kp1) * VAT3(uPSE, ii, jj, kk))

                   + VAT3(uPNW, ii, jj, kk) *
                         (-VAT3(oN, im1, j, kp1) * VAT3(uPW, ii, jj, kk) -
                          VAT3(uC, im1, jp1, k) * VAT3(oPNW, ii, jj, kk) +
                          VAT3(oC, im1, jp1, kp1) * VAT3(uPNW, ii, jj, kk) -
                          VAT3(oE, im1, jp1, kp1) * VAT3(uPN, ii, jj, kk));

        const auto TMP7_XOC = +VAT3(dPW, ii, jj, kk) *
                       (-VAT3(oN, im1, jm1, km1) * VAT3(dPSW, ii, jj, kk) +
                        VAT3(oC, im1, j, km1) * VAT3(dPW, ii, jj, kk) -
                        VAT3(uC, im1, j, km1) * VAT3(oPW, ii, jj, kk) -
                        VAT3(oN, im1, j, km1) * VAT3(dPNW, ii, jj, kk) -
                        VAT3(oE, im1, j, km1) * VAT3(dPC, ii, jj, kk))

                   + VAT3(uPSW, ii, jj, kk) *
                         (-VAT3(uC, im1, jm1, k) * VAT3(oPSW, ii, jj, kk) +
                          VAT3(oC, im1, jm1, kp1) * VAT3(uPSW, ii, jj, kk) -
                          VAT3(oN, im1, jm1, kp1) * VAT3(uPW, ii, jj, kk) -
                          VAT3(oE, im1, jm1, kp1) * VAT3(uPS, ii, jj, kk))

                   + VAT3(oPSW, ii, jj, kk) *
                         (-VAT3(uC, im1, jm1, km1) * VAT3(dPSW, ii, jj, kk) +
                          VAT3(oC, im1, jm1, k) * VAT3(oPSW, ii, jj, kk) -
                          VAT3(uC, im1, jm1, k) * VAT3(uPSW, ii, jj, kk) -
                          VAT3(oN, im1, jm1, k) * VAT3(oPW, ii, jj, kk) -
                          VAT3(oE, im1, jm1, k) * VAT3(oPS, ii, jj, kk));

        const auto TMP8_XOC = +VAT3(oPNW, ii, jj, kk) *
                       (-VAT3(oN, im1, j, k) * VAT3(oPW, ii, jj, kk) -
                        VAT3(uC, im1, jp1, km1) * VAT3(dPNW, ii, jj, kk) +
                        VAT3(oC, im1, jp1, k) * VAT3(oPNW, ii, jj, kk) -
                        VAT3(uC, im1, jp1, k) * VAT3(uPNW, ii, jj, kk) -
                        VAT3(oE, im1, jp1, k) * VAT3(oPN, ii, jj, kk))

                   + VAT3(dPNW, ii, jj, kk) *
                         (-VAT3(oN, im1, j, km1) * VAT3(dPW, ii, jj, kk) +
                          VAT3(oC, im1, jp1, km1) * VAT3(dPNW, ii, jj, kk) -
                          VAT3(uC, im1, jp1, km1) * VAT3(oPNW, ii, jj, kk) -
                          VAT3(oE, im1, jp1, km1) * VAT3(dPN, ii, jj, kk))

                   + VAT3(oPW, ii, jj, kk) *
                         (-VAT3(oN, im1, jm1, k) * VAT3(oPSW, ii, jj, kk) -
                          VAT3(uC, im1, j, km1) * VAT3(dPW, ii, jj, kk) +
                          VAT3(oC, im1, j, k) * VAT3(oPW, ii, jj, kk) -
                          VAT3(uC, im1, j, k) * VAT3(uPW, ii, jj, kk) -
                          VAT3(oN, im1, j, k) * VAT3(oPNW, ii, jj, kk) -
                          VAT3(oE, im1, j, k) * VAT3(oPC, ii, jj, kk));

        const auto TMP9_XOC = +VAT3(uPW, ii, jj, kk) *
                       (-VAT3(oN, im1, jm1, kp1) * VAT3(uPSW, ii, jj, kk) -
                        VAT3(uC, im1, j, k) * VAT3(oPW, ii, jj, kk) +
                        VAT3(oC, im1, j, kp1) * VAT3(uPW, ii, jj, kk) -
                        VAT3(oN, im1, j, kp1) * VAT3(uPNW, ii, jj, kk) -
                        VAT3(oE, im1, j, kp1) * VAT3(uPC, ii, jj, kk))

                   + VAT3(uPN, ii, jj, kk) *
                         (-VAT3(oE, im1, jp1, kp1) * VAT3(uPNW, ii, jj, kk) -
                          VAT3(oN, i, j, kp1) * VAT3(uPC, ii, jj, kk) -
                          VAT3(uC, i, jp1, k) * VAT3(oPN, ii, jj, kk) +
                          VAT3(oC, i, jp1, kp1) * VAT3(uPN, ii, jj, kk) -
                          VAT3(oE, i, jp1, kp1) * VAT3(uPNE, ii, jj, kk))

                   + VAT3(oPN, ii, jj, kk) *
                         (-VAT3(oE, im1, jp1, k) * VAT3(oPNW, ii, jj, kk) -
                          VAT3(oN, i, j, k) * VAT3(oPC, ii, jj, kk) -
                          VAT3(uC, i, jp1, km1) * VAT3(dPN, ii, jj, kk) +
                          VAT3(oC, i, jp1, k) * VAT3(oPN, ii, jj, kk) -
                          VAT3(uC, i, jp1, k) * VAT3(uPN, ii, jj, kk) -
                          VAT3(oE, i, jp1, k) * VAT3(oPNE, ii, jj, kk));

        VAT3(XoC, ii, jj, kk) = TMP1_XOC + TMP2_XOC + TMP3_XOC + TMP4_XOC +
                                TMP5_XOC + TMP6_XOC + TMP7_XOC + TMP8_XOC +
                                TMP9_XOC;

        // fprintf(data, "%19.12E\n", VAT3(XoC, ii, jj, kk));

        /* *************************************************************
         * *** > oE;
         * *************************************************************/

        // VAT3( XoE, ii,jj,kk) =
        const auto TMP1_XOE = VAT3(dPS, ii, jj, kk) * VAT3(oE, i, jm1, km1) *
                       VAT3(dPSW, iip1, jj, kk) +
                   VAT3(oPS, ii, jj, kk) * VAT3(oE, i, jm1, k) *
                       VAT3(oPSW, iip1, jj, kk) +
                   VAT3(uPS, ii, jj, kk) * VAT3(oE, i, jm1, kp1) *
                       VAT3(uPSW, iip1, jj, kk) +
                   VAT3(dPC, ii, jj, kk) * VAT3(oE, i, j, km1) *
                       VAT3(dPW, iip1, jj, kk) +
                   VAT3(oPC, ii, jj, kk) * VAT3(oE, i, j, k) *
                       VAT3(oPW, iip1, jj, kk) +
                   VAT3(uPC, ii, jj, kk) * VAT3(oE, i, j, kp1) *
                       VAT3(uPW, iip1, jj, kk) +
                   VAT3(dPN, ii, jj, kk) * VAT3(oE, i, jp1, km1) *
                       VAT3(dPNW, iip1, jj, kk) +
                   VAT3(oPN, ii, jj, kk) * VAT3(oE, i, jp1, k) *
                       VAT3(oPNW, iip1, jj, kk) +
                   VAT3(uPN, ii, jj, kk) * VAT3(oE, i, jp1, kp1) *
                       VAT3(uPNW, iip1, jj, kk)

                   - VAT3(dPSE, ii, jj, kk) *
                         (VAT3(oC, ip1, jm1, km1) * VAT3(dPSW, iip1, jj, kk) -
                          VAT3(uC, ip1, jm1, km1) * VAT3(oPSW, iip1, jj, kk) -
                          VAT3(oN, ip1, jm1, km1) * VAT3(dPW, iip1, jj, kk) -
                          VAT3(oE, ip1, jm1, km1) * VAT3(dPS, iip1, jj, kk));

        const auto TMP2_XOE = -VAT3(oPSE, ii, jj, kk) *
                       (-VAT3(uC, ip1, jm1, km1) * VAT3(dPSW, iip1, jj, kk) +
                        VAT3(oC, ip1, jm1, k) * VAT3(oPSW, iip1, jj, kk) -
                        VAT3(uC, ip1, jm1, k) * VAT3(uPSW, iip1, jj, kk) -
                        VAT3(oN, ip1, jm1, k) * VAT3(oPW, iip1, jj, kk) -
                        VAT3(oE, ip1, jm1, k) * VAT3(oPS, iip1, jj, kk))

                   - VAT3(uPSE, ii, jj, kk) *
                         (-VAT3(uC, ip1, jm1, k) * VAT3(oPSW, iip1, jj, kk) +
                          VAT3(oC, ip1, jm1, kp1) * VAT3(uPSW, iip1, jj, kk) -
                          VAT3(oN, ip1, jm1, kp1) * VAT3(uPW, iip1, jj, kk) -
                          VAT3(oE, ip1, jm1, kp1) * VAT3(uPS, iip1, jj, kk))

                   - VAT3(dPE, ii, jj, kk) *
                         (-VAT3(oN, ip1, jm1, km1) * VAT3(dPSW, iip1, jj, kk) +
                          VAT3(oC, ip1, j, km1) * VAT3(dPW, iip1, jj, kk) -
                          VAT3(uC, ip1, j, km1) * VAT3(oPW, iip1, jj, kk) -
                          VAT3(oN, ip1, j, km1) * VAT3(dPNW, iip1, jj, kk) -
                          VAT3(oE, ip1, j, km1) * VAT3(dPC, iip1, jj, kk));

        const auto TMP3_XOE = -VAT3(oPE, ii, jj, kk) *
                       (-VAT3(oN, ip1, jm1, k) * VAT3(oPSW, iip1, jj, kk) -
                        VAT3(uC, ip1, j, km1) * VAT3(dPW, iip1, jj, kk) +
                        VAT3(oC, ip1, j, k) * VAT3(oPW, iip1, jj, kk) -
                        VAT3(uC, ip1, j, k) * VAT3(uPW, iip1, jj, kk) -
                        VAT3(oN, ip1, j, k) * VAT3(oPNW, iip1, jj, kk) -
                        VAT3(oE, ip1, j, k) * VAT3(oPC, iip1, jj, kk))

                   - VAT3(uPE, ii, jj, kk) *
                         (-VAT3(oN, ip1, jm1, kp1) * VAT3(uPSW, iip1, jj, kk) -
                          VAT3(uC, ip1, j, k) * VAT3(oPW, iip1, jj, kk) +
                          VAT3(oC, ip1, j, kp1) * VAT3(uPW, iip1, jj, kk) -
                          VAT3(oN, ip1, j, kp1) * VAT3(uPNW, iip1, jj, kk) -
                          VAT3(oE, ip1, j, kp1) * VAT3(uPC, iip1, jj, kk))

                   - VAT3(dPNE, ii, jj, kk) *
                         (-VAT3(oN, ip1, j, km1) * VAT3(dPW, iip1, jj, kk) +
                          VAT3(oC, ip1, jp1, km1) * VAT3(dPNW, iip1, jj, kk) -
                          VAT3(uC, ip1, jp1, km1) * VAT3(oPNW, iip1, jj, kk) -
                          VAT3(oE, ip1, jp1, km1) * VAT3(dPN, iip1, jj, kk));

        const auto TMP4_XOE = -VAT3(oPNE, ii, jj, kk) *
                       (-VAT3(oN, ip1, j, k) * VAT3(oPW, iip1, jj, kk) -
                        VAT3(uC, ip1, jp1, km1) * VAT3(dPNW, iip1, jj, kk) +
                        VAT3(oC, ip1, jp1, k) * VAT3(oPNW, iip1, jj, kk) -
                        VAT3(uC, ip1, jp1, k) * VAT3(uPNW, iip1, jj, kk) -
                        VAT3(oE, ip1, jp1, k) * VAT3(oPN, iip1, jj, kk))

                   - VAT3(uPNE, ii, jj, kk) *
                         (-VAT3(oN, ip1, j, kp1) * VAT3(uPW, iip1, jj, kk) -
                          VAT3(uC, ip1, jp1, k) * VAT3(oPNW, iip1, jj, kk) +
                          VAT3(oC, ip1, jp1, kp1) * VAT3(uPNW, iip1, jj, kk) -
                          VAT3(oE, ip1, jp1, kp1) * VAT3(uPN, iip1, jj, kk));

        VAT3(XoE, ii, jj, kk) = TMP1_XOE + TMP2_XOE + TMP3_XOE + TMP4_XOE;

        // fprintf(data, "%19.12E\n", VAT3(XoE, ii, jj, kk));

        /// @todo:  Finish re-alignment

        /* *************************************************************
         * *** > oN;
         * *************************************************************/

        // VAT3( XoN, ii,jj,kk) =
        const auto TMP1_XON = VAT3(dPW, ii, jj, kk) * VAT3(oN, im1, j, km1) *
                       VAT3(dPSW, ii, jjp1, kk) +
                   VAT3(oPW, ii, jj, kk) * VAT3(oN, im1, j, k) *
                       VAT3(oPSW, ii, jjp1, kk) +
                   VAT3(uPW, ii, jj, kk) * VAT3(oN, im1, j, kp1) *
                       VAT3(uPSW, ii, jjp1, kk)

                   - VAT3(dPNW, ii, jj, kk) *
                         (VAT3(oC, im1, jp1, km1) * VAT3(dPSW, ii, jjp1, kk) -
                          VAT3(uC, im1, jp1, km1) * VAT3(oPSW, ii, jjp1, kk) -
                          VAT3(oN, im1, jp1, km1) * VAT3(dPW, ii, jjp1, kk) -
                          VAT3(oE, im1, jp1, km1) * VAT3(dPS, ii, jjp1, kk))

                   - VAT3(oPNW, ii, jj, kk) *
                         (-VAT3(uC, im1, jp1, km1) * VAT3(dPSW, ii, jjp1, kk) +
                          VAT3(oC, im1, jp1, k) * VAT3(oPSW, ii, jjp1, kk) -
                          VAT3(uC, im1, jp1, k) * VAT3(uPSW, ii, jjp1, kk) -
                          VAT3(oN, im1, jp1, k) * VAT3(oPW, ii, jjp1, kk) -
                          VAT3(oE, im1, jp1, k) * VAT3(oPS, ii, jjp1, kk));

        const auto TMP2_XON = -VAT3(uPNW, ii, jj, kk) *
                       (-VAT3(uC, im1, jp1, k) * VAT3(oPSW, ii, jjp1, kk) +
                        VAT3(oC, im1, jp1, kp1) * VAT3(uPSW, ii, jjp1, kk) -
                        VAT3(oN, im1, jp1, kp1) * VAT3(uPW, ii, jjp1, kk) -
                        VAT3(oE, im1, jp1, kp1) * VAT3(uPS, ii, jjp1, kk))

                   + VAT3(dPC, ii, jj, kk) * VAT3(oN, i, j, km1) *
                         VAT3(dPS, ii, jjp1, kk) +
                   VAT3(oPC, ii, jj, kk) * VAT3(oN, i, j, k) *
                       VAT3(oPS, ii, jjp1, kk) +
                   VAT3(uPC, ii, jj, kk) * VAT3(oN, i, j, kp1) *
                       VAT3(uPS, ii, jjp1, kk)

                   - VAT3(dPN, ii, jj, kk) *
                         (-VAT3(oE, im1, jp1, km1) * VAT3(dPSW, ii, jjp1, kk) +
                          VAT3(oC, i, jp1, km1) * VAT3(dPS, ii, jjp1, kk) -
                          VAT3(uC, i, jp1, km1) * VAT3(oPS, ii, jjp1, kk) -
                          VAT3(oN, i, jp1, km1) * VAT3(dPC, ii, jjp1, kk) -
                          VAT3(oE, i, jp1, km1) * VAT3(dPSE, ii, jjp1, kk));

        const auto TMP3_XON = -VAT3(oPN, ii, jj, kk) *
                       (-VAT3(oE, im1, jp1, k) * VAT3(oPSW, ii, jjp1, kk) -
                        VAT3(uC, i, jp1, km1) * VAT3(dPS, ii, jjp1, kk) +
                        VAT3(oC, i, jp1, k) * VAT3(oPS, ii, jjp1, kk) -
                        VAT3(uC, i, jp1, k) * VAT3(uPS, ii, jjp1, kk) -
                        VAT3(oN, i, jp1, k) * VAT3(oPC, ii, jjp1, kk) -
                        VAT3(oE, i, jp1, k) * VAT3(oPSE, ii, jjp1, kk))

                   - VAT3(uPN, ii, jj, kk) *
                         (-VAT3(oE, im1, jp1, kp1) * VAT3(uPSW, ii, jjp1, kk) -
                          VAT3(uC, i, jp1, k) * VAT3(oPS, ii, jjp1, kk) +
                          VAT3(oC, i, jp1, kp1) * VAT3(uPS, ii, jjp1, kk) -
                          VAT3(oN, i, jp1, kp1) * VAT3(uPC, ii, jjp1, kk) -
                          VAT3(oE, i, jp1, kp1) * VAT3(uPSE, ii, jjp1, kk))

                   + VAT3(dPE, ii, jj, kk) * VAT3(oN, ip1, j, km1) *
                         VAT3(dPSE, ii, jjp1, kk) +
                   VAT3(oPE, ii, jj, kk) * VAT3(oN, ip1, j, k) *
                       VAT3(oPSE, ii, jjp1, kk) +
                   VAT3(uPE, ii, jj, kk) * VAT3(oN, ip1, j, kp1) *
                       VAT3(uPSE, ii, jjp1, kk);

        const auto TMP4_XON = -VAT3(dPNE, ii, jj, kk) *
                       (-VAT3(oE, i, jp1, km1) * VAT3(dPS, ii, jjp1, kk) +
                        VAT3(oC, ip1, jp1, km1) * VAT3(dPSE, ii, jjp1, kk) -
                        VAT3(uC, ip1, jp1, km1) * VAT3(oPSE, ii, jjp1, kk) -
                        VAT3(oN, ip1, jp1, km1) * VAT3(dPE, ii, jjp1, kk))

                   - VAT3(oPNE, ii, jj, kk) *
                         (-VAT3(oE, i, jp1, k) * VAT3(oPS, ii, jjp1, kk) -
                          VAT3(uC, ip1, jp1, km1) * VAT3(dPSE, ii, jjp1, kk) +
                          VAT3(oC, ip1, jp1, k) * VAT3(oPSE, ii, jjp1, kk) -
                          VAT3(uC, ip1, jp1, k) * VAT3(uPSE, ii, jjp1, kk) -
                          VAT3(oN, ip1, jp1, k) * VAT3(oPE, ii, jjp1, kk))

                   - VAT3(uPNE, ii, jj, kk) *
                         (-VAT3(oE, i, jp1, kp1) * VAT3(uPS, ii, jjp1, kk) -
                          VAT3(uC, ip1, jp1, k) * VAT3(oPSE, ii, jjp1, kk) +
                          VAT3(oC, ip1, jp1, kp1) * VAT3(uPSE, ii, jjp1, kk) -
                          VAT3(oN, ip1, jp1, kp1) * VAT3(uPE, ii, jjp1, kk));

        VAT3(XoN, ii, jj, kk) = TMP1_XON + TMP2_XON + TMP3_XON + TMP4_XON;

        // fprintf(data, "%19.12E\n", VAT3(XoN, ii, jj, kk));

        /* *************************************************************
         * *** > uC;
         * *************************************************************/

        // VAT3( XuC, ii,jj,kk) =
        const auto TMP1_XUC = VAT3(oPSW, ii, jj, kk) * VAT3(uC, im1, jm1, k) *
                       VAT3(dPSW, ii, jj, kkp1)

                   - VAT3(uPSW, ii, jj, kk) *
                         (VAT3(oC, im1, jm1, kp1) * VAT3(dPSW, ii, jj, kkp1) -
                          VAT3(uC, im1, jm1, kp1) * VAT3(oPSW, ii, jj, kkp1) -
                          VAT3(oN, im1, jm1, kp1) * VAT3(dPW, ii, jj, kkp1) -
                          VAT3(oE, im1, jm1, kp1) * VAT3(dPS, ii, jj, kkp1))

                   + VAT3(oPW, ii, jj, kk) * VAT3(uC, im1, j, k) *
                         VAT3(dPW, ii, jj, kkp1)

                   - VAT3(uPW, ii, jj, kk) *
                         (-VAT3(oN, im1, jm1, kp1) * VAT3(dPSW, ii, jj, kkp1) +
                          VAT3(oC, im1, j, kp1) * VAT3(dPW, ii, jj, kkp1) -
                          VAT3(uC, im1, j, kp1) * VAT3(oPW, ii, jj, kkp1) -
                          VAT3(oN, im1, j, kp1) * VAT3(dPNW, ii, jj, kkp1) -
                          VAT3(oE, im1, j, kp1) * VAT3(dPC, ii, jj, kkp1))

                   + VAT3(oPNW, ii, jj, kk) * VAT3(uC, im1, jp1, k) *
                         VAT3(dPNW, ii, jj, kkp1);

        const auto TMP2_XUC = -VAT3(uPNW, ii, jj, kk) *
                       (-VAT3(oN, im1, j, kp1) * VAT3(dPW, ii, jj, kkp1) +
                        VAT3(oC, im1, jp1, kp1) * VAT3(dPNW, ii, jj, kkp1) -
                        VAT3(uC, im1, jp1, kp1) * VAT3(oPNW, ii, jj, kkp1) -
                        VAT3(oE, im1, jp1, kp1) * VAT3(dPN, ii, jj, kkp1))

                   + VAT3(oPS, ii, jj, kk) * VAT3(uC, i, jm1, k) *
                         VAT3(dPS, ii, jj, kkp1)

                   - VAT3(uPS, ii, jj, kk) *
                         (-VAT3(oE, im1, jm1, kp1) * VAT3(dPSW, ii, jj, kkp1) +
                          VAT3(oC, i, jm1, kp1) * VAT3(dPS, ii, jj, kkp1) -
                          VAT3(uC, i, jm1, kp1) * VAT3(oPS, ii, jj, kkp1) -
                          VAT3(oN, i, jm1, kp1) * VAT3(dPC, ii, jj, kkp1) -
                          VAT3(oE, i, jm1, kp1) * VAT3(dPSE, ii, jj, kkp1))

                   + VAT3(oPC, ii, jj, kk) * VAT3(uC, i, j, k) *
                         VAT3(dPC, ii, jj, kkp1)

                   - VAT3(uPC, ii, jj, kk) *
                         (-VAT3(oE, im1, j, kp1) * VAT3(dPW, ii, jj, kkp1) -
                          VAT3(oN, i, jm1, kp1) * VAT3(dPS, ii, jj, kkp1) +
                          VAT3(oC, i, j, kp1) * VAT3(dPC, ii, jj, kkp1) -
                          VAT3(uC, i, j, kp1) * VAT3(oPC, ii, jj, kkp1) -
                          VAT3(oN, i, j, kp1) * VAT3(dPN, ii, jj, kkp1) -
                          VAT3(oE, i, j, kp1) * VAT3(dPE, ii, jj, kkp1));

        const auto TMP3_XUC = +VAT3(oPN, ii, jj, kk) * VAT3(uC, i, jp1, k) *
                       VAT3(dPN, ii, jj, kkp1)

                   - VAT3(uPN, ii, jj, kk) *
                         (-VAT3(oE, im1, jp1, kp1) * VAT3(dPNW, ii, jj, kkp1) -
                          VAT3(oN, i, j, kp1) * VAT3(dPC, ii, jj, kkp1) +
                          VAT3(oC, i, jp1, kp1) * VAT3(dPN, ii, jj, kkp1) -
                          VAT3(uC, i, jp1, kp1) * VAT3(oPN, ii, jj, kkp1) -
                          VAT3(oE, i, jp1, kp1) * VAT3(dPNE, ii, jj, kkp1))

                   + VAT3(oPSE, ii, jj, kk) * VAT3(uC, ip1, jm1, k) *
                         VAT3(dPSE, ii, jj, kkp1)

                   - VAT3(uPSE, ii, jj, kk) *
                         (-VAT3(oE, i, jm1, kp1) * VAT3(dPS, ii, jj, kkp1) +
                          VAT3(oC, ip1, jm1, kp1) * VAT3(dPSE, ii, jj, kkp1) -
                          VAT3(uC, ip1, jm1, kp1) * VAT3(oPSE, ii, jj, kkp1) -
                          VAT3(oN, ip1, jm1, kp1) * VAT3(dPE, ii, jj, kkp1));

        const auto TMP4_XUC = +VAT3(oPE, ii, jj, kk) * VAT3(uC, ip1, j, k) *
                       VAT3(dPE, ii, jj, kkp1)

                   - VAT3(uPE, ii, jj, kk) *
                         (-VAT3(oE, i, j, kp1) * VAT3(dPC, ii, jj, kkp1) -
                          VAT3(oN, ip1, jm1, kp1) * VAT3(dPSE, ii, jj, kkp1) +
                          VAT3(oC, ip1, j, kp1) * VAT3(dPE, ii, jj, kkp1) -
                          VAT3(uC, ip1, j, kp1) * VAT3(oPE, ii, jj, kkp1) -
                          VAT3(oN, ip1, j, kp1) * VAT3(dPNE, ii, jj, kkp1))

                   + VAT3(oPNE, ii, jj, kk) * VAT3(uC, ip1, jp1, k) *
                         VAT3(dPNE, ii, jj, kkp1)

                   - VAT3(uPNE, ii, jj, kk) *
                         (-VAT3(oE, i, jp1, kp1) * VAT3(dPN, ii, jj, kkp1) -
                          VAT3(oN, ip1, j, kp1) * VAT3(dPE, ii, jj, kkp1) +
                          VAT3(oC, ip1, jp1, kp1) * VAT3(dPNE, ii, jj, kkp1) -
                          VAT3(uC, ip1, jp1, kp1) * VAT3(oPNE, ii, jj, kkp1));

        VAT3(XuC, ii, jj, kk) = TMP1_XUC + TMP2_XUC + TMP3_XUC + TMP4_XUC;

        // fprintf(data, "%19.12E\n", VAT3(XuC, ii, jj, kk));

        /* *************************************************************
         * *** > oNE;
         * *************************************************************/

        VAT3(XoNE, ii, jj, kk) =
            VAT3(dPN, ii, jj, kk) * VAT3(oE, i, jp1, km1) *
                VAT3(dPSW, iip1, jjp1, kk) +
            VAT3(oPN, ii, jj, kk) * VAT3(oE, i, jp1, k) *
                VAT3(oPSW, iip1, jjp1, kk) +
            VAT3(uPN, ii, jj, kk) * VAT3(oE, i, jp1, kp1) *
                VAT3(uPSW, iip1, jjp1, kk) +
            VAT3(dPE, ii, jj, kk) * VAT3(oN, ip1, j, km1) *
                VAT3(dPSW, iip1, jjp1, kk) +
            VAT3(oPE, ii, jj, kk) * VAT3(oN, ip1, j, k) *
                VAT3(oPSW, iip1, jjp1, kk) +
            VAT3(uPE, ii, jj, kk) * VAT3(oN, ip1, j, kp1) *
                VAT3(uPSW, iip1, jjp1, kk)

            - VAT3(dPNE, ii, jj, kk) *
                  (VAT3(oC, ip1, jp1, km1) * VAT3(dPSW, iip1, jjp1, kk) -
                   VAT3(uC, ip1, jp1, km1) * VAT3(oPSW, iip1, jjp1, kk) -
                   VAT3(oN, ip1, jp1, km1) * VAT3(dPW, iip1, jjp1, kk) -
                   VAT3(oE, ip1, jp1, km1) * VAT3(dPS, iip1, jjp1, kk))

            - VAT3(oPNE, ii, jj, kk) *
                  (-VAT3(uC, ip1, jp1, km1) * VAT3(dPSW, iip1, jjp1, kk) +
                   VAT3(oC, ip1, jp1, k) * VAT3(oPSW, iip1, jjp1, kk) -
                   VAT3(uC, ip1, jp1, k) * VAT3(uPSW, iip1, jjp1, kk) -
                   VAT3(oN, ip1, jp1, k) * VAT3(oPW, iip1, jjp1, kk) -
                   VAT3(oE, ip1, jp1, k) * VAT3(oPS, iip1, jjp1, kk))

            - VAT3(uPNE, ii, jj, kk) *
                  (-VAT3(uC, ip1, jp1, k) * VAT3(oPSW, iip1, jjp1, kk) +
                   VAT3(oC, ip1, jp1, kp1) * VAT3(uPSW, iip1, jjp1, kk) -
                   VAT3(oN, ip1, jp1, kp1) * VAT3(uPW, iip1, jjp1, kk) -
                   VAT3(oE, ip1, jp1, kp1) * VAT3(uPS, iip1, jjp1, kk));

        // fprintf(data, "%19.12E\n", VAT3(XoNE, ii, jj, kk));

        /* *************************************************************
         * *** > oNW;
         * *************************************************************/

        VAT3(XoNW, ii, jj, kk) =
            VAT3(dPW, ii, jj, kk) * VAT3(oN, im1, j, km1) *
                VAT3(dPSE, iim1, jjp1, kk) +
            VAT3(oPW, ii, jj, kk) * VAT3(oN, im1, j, k) *
                VAT3(oPSE, iim1, jjp1, kk) +
            VAT3(uPW, ii, jj, kk) * VAT3(oN, im1, j, kp1) *
                VAT3(uPSE, iim1, jjp1, kk)

            - VAT3(dPNW, ii, jj, kk) *
                  (-VAT3(oE, im2, jp1, km1) * VAT3(dPS, iim1, jjp1, kk) +
                   VAT3(oC, im1, jp1, km1) * VAT3(dPSE, iim1, jjp1, kk) -
                   VAT3(uC, im1, jp1, km1) * VAT3(oPSE, iim1, jjp1, kk) -
                   VAT3(oN, im1, jp1, km1) * VAT3(dPE, iim1, jjp1, kk))

            - VAT3(oPNW, ii, jj, kk) *
                  (-VAT3(oE, im2, jp1, k) * VAT3(oPS, iim1, jjp1, kk) -
                   VAT3(uC, im1, jp1, km1) * VAT3(dPSE, iim1, jjp1, kk) +
                   VAT3(oC, im1, jp1, k) * VAT3(oPSE, iim1, jjp1, kk) -
                   VAT3(uC, im1, jp1, k) * VAT3(uPSE, iim1, jjp1, kk) -
                   VAT3(oN, im1, jp1, k) * VAT3(oPE, iim1, jjp1, kk))

            - VAT3(uPNW, ii, jj, kk) *
                  (-VAT3(oE, im2, jp1, kp1) * VAT3(uPS, iim1, jjp1, kk) -
                   VAT3(uC, im1, jp1, k) * VAT3(oPSE, iim1, jjp1, kk) +
                   VAT3(oC, im1, jp1, kp1) * VAT3(uPSE, iim1, jjp1, kk) -
                   VAT3(oN, im1, jp1, kp1) * VAT3(uPE, iim1, jjp1, kk))

            + VAT3(dPN, ii, jj, kk) * VAT3(oE, im1, jp1, km1) *
                  VAT3(dPSE, iim1, jjp1, kk) +
            VAT3(oPN, ii, jj, kk) * VAT3(oE, im1, jp1, k) *
                VAT3(oPSE, iim1, jjp1, kk) +
            VAT3(uPN, ii, jj, kk) * VAT3(oE, im1, jp1, kp1) *
                VAT3(uPSE, iim1, jjp1, kk);

        // fprintf(data, "%19.12E\n", VAT3(XoNW, ii, jj, kk));

        /* *************************************************************
         * *** > uE;
         * *************************************************************/

        VAT3(XuE, ii, jj, kk) =
            VAT3(uPS, ii, jj, kk) * VAT3(oE, i, jm1, kp1) *
                VAT3(dPSW, iip1, jj, kkp1) +
            VAT3(uPC, ii, jj, kk) * VAT3(oE, i, j, kp1) *
                VAT3(dPW, iip1, jj, kkp1) +
            VAT3(uPN, ii, jj, kk) * VAT3(oE, i, jp1, kp1) *
                VAT3(dPNW, iip1, jj, kkp1) +
            VAT3(oPSE, ii, jj, kk) * VAT3(uC, ip1, jm1, k) *
                VAT3(dPSW, iip1, jj, kkp1)

            - VAT3(uPSE, ii, jj, kk) *
                  (VAT3(oC, ip1, jm1, kp1) * VAT3(dPSW, iip1, jj, kkp1) -
                   VAT3(uC, ip1, jm1, kp1) * VAT3(oPSW, iip1, jj, kkp1) -
                   VAT3(oN, ip1, jm1, kp1) * VAT3(dPW, iip1, jj, kkp1) -
                   VAT3(oE, ip1, jm1, kp1) * VAT3(dPS, iip1, jj, kkp1))

            + VAT3(oPE, ii, jj, kk) * VAT3(uC, ip1, j, k) *
                  VAT3(dPW, iip1, jj, kkp1)

            - VAT3(uPE, ii, jj, kk) *
                  (-VAT3(oN, ip1, jm1, kp1) * VAT3(dPSW, iip1, jj, kkp1) +
                   VAT3(oC, ip1, j, kp1) * VAT3(dPW, iip1, jj, kkp1) -
                   VAT3(uC, ip1, j, kp1) * VAT3(oPW, iip1, jj, kkp1) -
                   VAT3(oN, ip1, j, kp1) * VAT3(dPNW, iip1, jj, kkp1) -
                   VAT3(oE, ip1, j, kp1) * VAT3(dPC, iip1, jj, kkp1))

            + VAT3(oPNE, ii, jj, kk) * VAT3(uC, ip1, jp1, k) *
                  VAT3(dPNW, iip1, jj, kkp1)

            - VAT3(uPNE, ii, jj, kk) *
                  (-VAT3(oN, ip1, j, kp1) * VAT3(dPW, iip1, jj, kkp1) +
                   VAT3(oC, ip1, jp1, kp1) * VAT3(dPNW, iip1, jj, kkp1) -
                   VAT3(uC, ip1, jp1, kp1) * VAT3(oPNW, iip1, jj, kkp1) -
                   VAT3(oE, ip1, jp1, kp1) * VAT3(dPN, iip1, jj, kkp1));

        // fprintf(data, "%19.12E\n", VAT3(XuE, ii, jj, kk));

        /* *************************************************************
         * *** > uW;
         * *************************************************************/

        VAT3(XuW, ii, jj, kk) =
            VAT3(oPSW, ii, jj, kk) * VAT3(uC, im1, jm1, k) *
                VAT3(dPSE, iim1, jj, kkp1)

            - VAT3(uPSW, ii, jj, kk) *
                  (-VAT3(oE, im2, jm1, kp1) * VAT3(dPS, iim1, jj, kkp1) +
                   VAT3(oC, im1, jm1, kp1) * VAT3(dPSE, iim1, jj, kkp1) -
                   VAT3(uC, im1, jm1, kp1) * VAT3(oPSE, iim1, jj, kkp1) -
                   VAT3(oN, im1, jm1, kp1) * VAT3(dPE, iim1, jj, kkp1))

            + VAT3(oPW, ii, jj, kk) * VAT3(uC, im1, j, k) *
                  VAT3(dPE, iim1, jj, kkp1)

            - VAT3(uPW, ii, jj, kk) *
                  (-VAT3(oE, im2, j, kp1) * VAT3(dPC, iim1, jj, kkp1) -
                   VAT3(oN, im1, jm1, kp1) * VAT3(dPSE, iim1, jj, kkp1) +
                   VAT3(oC, im1, j, kp1) * VAT3(dPE, iim1, jj, kkp1) -
                   VAT3(uC, im1, j, kp1) * VAT3(oPE, iim1, jj, kkp1) -
                   VAT3(oN, im1, j, kp1) * VAT3(dPNE, iim1, jj, kkp1))

            + VAT3(oPNW, ii, jj, kk) * VAT3(uC, im1, jp1, k) *
                  VAT3(dPNE, iim1, jj, kkp1)

            - VAT3(uPNW, ii, jj, kk) *
                  (-VAT3(oE, im2, jp1, kp1) * VAT3(dPN, iim1, jj, kkp1) -
                   VAT3(oN, im1, j, kp1) * VAT3(dPE, iim1, jj, kkp1) +
                   VAT3(oC, im1, jp1, kp1) * VAT3(dPNE, iim1, jj, kkp1) -
                   VAT3(uC, im1, jp1, kp1) * VAT3(oPNE, iim1, jj, kkp1))

            + VAT3(uPS, ii, jj, kk) * VAT3(oE, im1, jm1, kp1) *
                  VAT3(dPSE, iim1, jj, kkp1) +
            VAT3(uPC, ii, jj, kk) * VAT3(oE, im1, j, kp1) *
                VAT3(dPE, iim1, jj, kkp1) +
            VAT3(uPN, ii, jj, kk) * VAT3(oE, im1, jp1, kp1) *
                VAT3(dPNE, iim1, jj, kkp1);

        // fprintf(data, "%19.12E\n", VAT3(XuW, ii, jj, kk));

        /* *************************************************************
         * *** > uN;
         * *************************************************************/

        VAT3(XuN, ii, jj, kk) =
            VAT3(uPW, ii, jj, kk) * VAT3(oN, im1, j, kp1) *
                VAT3(dPSW, ii, jjp1, kkp1) +
            VAT3(oPNW, ii, jj, kk) * VAT3(uC, im1, jp1, k) *
                VAT3(dPSW, ii, jjp1, kkp1)

            - VAT3(uPNW, ii, jj, kk) *
                  (VAT3(oC, im1, jp1, kp1) * VAT3(dPSW, ii, jjp1, kkp1) -
                   VAT3(uC, im1, jp1, kp1) * VAT3(oPSW, ii, jjp1, kkp1) -
                   VAT3(oN, im1, jp1, kp1) * VAT3(dPW, ii, jjp1, kkp1) -
                   VAT3(oE, im1, jp1, kp1) * VAT3(dPS, ii, jjp1, kkp1))

            + VAT3(uPC, ii, jj, kk) * VAT3(oN, i, j, kp1) *
                  VAT3(dPS, ii, jjp1, kkp1) +
            VAT3(oPN, ii, jj, kk) * VAT3(uC, i, jp1, k) *
                VAT3(dPS, ii, jjp1, kkp1)

            - VAT3(uPN, ii, jj, kk) *
                  (-VAT3(oE, im1, jp1, kp1) * VAT3(dPSW, ii, jjp1, kkp1) +
                   VAT3(oC, i, jp1, kp1) * VAT3(dPS, ii, jjp1, kkp1) -
                   VAT3(uC, i, jp1, kp1) * VAT3(oPS, ii, jjp1, kkp1) -
                   VAT3(oN, i, jp1, kp1) * VAT3(dPC, ii, jjp1, kkp1) -
                   VAT3(oE, i, jp1, kp1) * VAT3(dPSE, ii, jjp1, kkp1))

            + VAT3(uPE, ii, jj, kk) * VAT3(oN, ip1, j, kp1) *
                  VAT3(dPSE, ii, jjp1, kkp1) +
            VAT3(oPNE, ii, jj, kk) * VAT3(uC, ip1, jp1, k) *
                VAT3(dPSE, ii, jjp1, kkp1)

            - VAT3(uPNE, ii, jj, kk) *
                  (-VAT3(oE, i, jp1, kp1) * VAT3(dPS, ii, jjp1, kkp1) +
                   VAT3(oC, ip1, jp1, kp1) * VAT3(dPSE, ii, jjp1, kkp1) -
                   VAT3(uC, ip1, jp1, kp1) * VAT3(oPSE, ii, jjp1, kkp1) -
                   VAT3(oN, ip1, jp1, kp1) * VAT3(dPE, ii, jjp1, kkp1));

        // fprintf(data, "%19.12E\n", VAT3(XuN, ii, jj, kk));

        /* *************************************************************
         * *** > uS;
         * *************************************************************/

        VAT3(XuS, ii, jj, kk) =
            VAT3(oPSW, ii, jj, kk) * VAT3(uC, im1, jm1, k) *
                VAT3(dPNW, ii, jjm1, kkp1)

            - VAT3(uPSW, ii, jj, kk) *
                  (-VAT3(oN, im1, jm2, kp1) * VAT3(dPW, ii, jjm1, kkp1) +
                   VAT3(oC, im1, jm1, kp1) * VAT3(dPNW, ii, jjm1, kkp1) -
                   VAT3(uC, im1, jm1, kp1) * VAT3(oPNW, ii, jjm1, kkp1) -
                   VAT3(oE, im1, jm1, kp1) * VAT3(dPN, ii, jjm1, kkp1))

            + VAT3(uPW, ii, jj, kk) * VAT3(oN, im1, jm1, kp1) *
                  VAT3(dPNW, ii, jjm1, kkp1) +
            VAT3(oPS, ii, jj, kk) * VAT3(uC, i, jm1, k) *
                VAT3(dPN, ii, jjm1, kkp1)

            - VAT3(uPS, ii, jj, kk) *
                  (-VAT3(oE, im1, jm1, kp1) * VAT3(dPNW, ii, jjm1, kkp1) -
                   VAT3(oN, i, jm2, kp1) * VAT3(dPC, ii, jjm1, kkp1) +
                   VAT3(oC, i, jm1, kp1) * VAT3(dPN, ii, jjm1, kkp1) -
                   VAT3(uC, i, jm1, kp1) * VAT3(oPN, ii, jjm1, kkp1) -
                   VAT3(oE, i, jm1, kp1) * VAT3(dPNE, ii, jjm1, kkp1))

            + VAT3(uPC, ii, jj, kk) * VAT3(oN, i, jm1, kp1) *
                  VAT3(dPN, ii, jjm1, kkp1) +
            VAT3(oPSE, ii, jj, kk) * VAT3(uC, ip1, jm1, k) *
                VAT3(dPNE, ii, jjm1, kkp1)

            - VAT3(uPSE, ii, jj, kk) *
                  (-VAT3(oE, i, jm1, kp1) * VAT3(dPN, ii, jjm1, kkp1) -
                   VAT3(oN, ip1, jm2, kp1) * VAT3(dPE, ii, jjm1, kkp1) +
                   VAT3(oC, ip1, jm1, kp1) * VAT3(dPNE, ii, jjm1, kkp1) -
                   VAT3(uC, ip1, jm1, kp1) * VAT3(oPNE, ii, jjm1, kkp1))

            + VAT3(uPE, ii, jj, kk) * VAT3(oN, ip1, jm1, kp1) *
                  VAT3(dPNE, ii, jjm1, kkp1);

        // fprintf(data, "%19.12E\n", VAT3(XuS, ii, jj, kk));

        /* *************************************************************
         * *** > uNE;
         * *************************************************************/

        VAT3(XuNE, ii, jj, kk) =
            VAT3(uPN, ii, jj, kk) * VAT3(oE, i, jp1, kp1) *
                VAT3(dPSW, iip1, jjp1, kkp1) +
            VAT3(uPE, ii, jj, kk) * VAT3(oN, ip1, j, kp1) *
                VAT3(dPSW, iip1, jjp1, kkp1) +
            VAT3(oPNE, ii, jj, kk) * VAT3(uC, ip1, jp1, k) *
                VAT3(dPSW, iip1, jjp1, kkp1)

            - VAT3(uPNE, ii, jj, kk) *
                  (VAT3(oC, ip1, jp1, kp1) * VAT3(dPSW, iip1, jjp1, kkp1) -
                   VAT3(uC, ip1, jp1, kp1) * VAT3(oPSW, iip1, jjp1, kkp1) -
                   VAT3(oN, ip1, jp1, kp1) * VAT3(dPW, iip1, jjp1, kkp1) -
                   VAT3(oE, ip1, jp1, kp1) * VAT3(dPS, iip1, jjp1, kkp1));

        // fprintf(data, "%19.12E\n", VAT3(XuNE, ii, jj, kk));

        /* *************************************************************
         * *** > uNW;
         * *************************************************************/

        VAT3(XuNW, ii, jj, kk) =
            VAT3(uPW, ii, jj, kk) * VAT3(oN, im1, j, kp1) *
                VAT3(dPSE, iim1, jjp1, kkp1) +
            VAT3(oPNW, ii, jj, kk) * VAT3(uC, im1, jp1, k) *
                VAT3(dPSE, iim1, jjp1, kkp1)

            - VAT3(uPNW, ii, jj, kk) *
                  (-VAT3(oE, im2, jp1, kp1) * VAT3(dPS, iim1, jjp1, kkp1) +
                   VAT3(oC, im1, jp1, kp1) * VAT3(dPSE, iim1, jjp1, kkp1) -
                   VAT3(uC, im1, jp1, kp1) * VAT3(oPSE, iim1, jjp1, kkp1) -
                   VAT3(oN, im1, jp1, kp1) * VAT3(dPE, iim1, jjp1, kkp1))

            + VAT3(uPN, ii, jj, kk) * VAT3(oE, im1, jp1, kp1) *
                  VAT3(dPSE, iim1, jjp1, kkp1);

        // fprintf(data, "%19.12E\n", VAT3(XuNW, ii, jj, kk));

        /* *************************************************************
         * *** > uSE;
         * *************************************************************/

        VAT3(XuSE, ii, jj, kk) =
            VAT3(uPS, ii, jj, kk) * VAT3(oE, i, jm1, kp1) *
                VAT3(dPNW, iip1, jjm1, kkp1) +
            VAT3(oPSE, ii, jj, kk) * VAT3(uC, ip1, jm1, k) *
                VAT3(dPNW, iip1, jjm1, kkp1)

            - VAT3(uPSE, ii, jj, kk) *
                  (-VAT3(oN, ip1, jm2, kp1) * VAT3(dPW, iip1, jjm1, kkp1) +
                   VAT3(oC, ip1, jm1, kp1) * VAT3(dPNW, iip1, jjm1, kkp1) -
                   VAT3(uC, ip1, jm1, kp1) * VAT3(oPNW, iip1, jjm1, kkp1) -
                   VAT3(oE, ip1, jm1, kp1) * VAT3(dPN, iip1, jjm1, kkp1))

            + VAT3(uPE, ii, jj, kk) * VAT3(oN, ip1, jm1, kp1) *
                  VAT3(dPNW, iip1, jjm1, kkp1);

        // fprintf(data, "%19.12E\n", VAT3(XuSE, ii, jj, kk));

        /* *************************************************************
         * *** > uSW;
         * *************************************************************/

        VAT3(XuSW, ii, jj, kk) =
            VAT3(oPSW, ii, jj, kk) * VAT3(uC, im1, jm1, k) *
                VAT3(dPNE, iim1, jjm1, kkp1)

            - VAT3(uPSW, ii, jj, kk) *
                  (-VAT3(oE, im2, jm1, kp1) * VAT3(dPN, iim1, jjm1, kkp1) -
                   VAT3(oN, im1, jm2, kp1) * VAT3(dPE, iim1, jjm1, kkp1) +
                   VAT3(oC, im1, jm1, kp1) * VAT3(dPNE, iim1, jjm1, kkp1) -
                   VAT3(uC, im1, jm1, kp1) * VAT3(oPNE, iim1, jjm1, kkp1))

            + VAT3(uPW, ii, jj, kk) * VAT3(oN, im1, jm1, kp1) *
                  VAT3(dPNE, iim1, jjm1, kkp1) +
            VAT3(uPS, ii, jj, kk) * VAT3(oE, im1, jm1, kp1) *
                VAT3(dPNE, iim1, jjm1, kkp1);

        // fprintf(data, "%19.12E\n", VAT3(XuSW, ii, jj, kk));
      });
    
  
}

VPUBLIC void
VbuildG_27(int *nxf, int *nyf, int *nzf, int *nx, int *ny, int *nz, DataType *oPC,
           DataType *oPN, DataType *oPS, DataType *oPE, DataType *oPW, DataType *oPNE,
           DataType *oPNW, DataType *oPSE, DataType *oPSW, DataType *uPC, DataType *uPN,
           DataType *uPS, DataType *uPE, DataType *uPW, DataType *uPNE, DataType *uPNW,
           DataType *uPSE, DataType *uPSW, DataType *dPC, DataType *dPN, DataType *dPS,
           DataType *dPE, DataType *dPW, DataType *dPNE, DataType *dPNW, DataType *dPSE,
           DataType *dPSW, DataType *oC, DataType *oE, DataType *oN, DataType *uC,
           DataType *oNE, DataType *oNW, DataType *uE, DataType *uW, DataType *uN,
           DataType *uS, DataType *uNE, DataType *uNW, DataType *uSE, DataType *uSW,
           DataType *XoC, DataType *XoE, DataType *XoN, DataType *XuC, DataType *XoNE,
           DataType *XoNW, DataType *XuE, DataType *XuW, DataType *XuN, DataType *XuS,
           DataType *XuNE, DataType *XuNW, DataType *XuSE, DataType *XuSW, sycl::queue &q) {


  MAT3(oC, *nxf, *nyf, *nzf);
  MAT3(oE, *nxf, *nyf, *nzf);
  MAT3(oN, *nxf, *nyf, *nzf);
  MAT3(oNE, *nxf, *nyf, *nzf);
  MAT3(oNW, *nxf, *nyf, *nzf);

  MAT3(uC, *nxf, *nyf, *nzf);
  MAT3(uE, *nxf, *nyf, *nzf);
  MAT3(uW, *nxf, *nyf, *nzf);
  MAT3(uN, *nxf, *nyf, *nzf);
  MAT3(uS, *nxf, *nyf, *nzf);
  MAT3(uNE, *nxf, *nyf, *nzf);
  MAT3(uNW, *nxf, *nyf, *nzf);
  MAT3(uSE, *nxf, *nyf, *nzf);
  MAT3(uSW, *nxf, *nyf, *nzf);

  MAT3(XoC, *nx, *ny, *nz);
  MAT3(XoE, *nx, *ny, *nz);
  MAT3(XoN, *nx, *ny, *nz);
  MAT3(XoNE, *nx, *ny, *nz);
  MAT3(XoNW, *nx, *ny, *nz);

  MAT3(XuC, *nx, *ny, *nz);
  MAT3(XuE, *nx, *ny, *nz);
  MAT3(XuW, *nx, *ny, *nz);
  MAT3(XuN, *nx, *ny, *nz);
  MAT3(XuS, *nx, *ny, *nz);
  MAT3(XuNE, *nx, *ny, *nz);
  MAT3(XuNW, *nx, *ny, *nz);
  MAT3(XuSE, *nx, *ny, *nz);
  MAT3(XuSW, *nx, *ny, *nz);

  MAT3(oPC, *nx, *ny, *nz);
  MAT3(oPN, *nx, *ny, *nz);
  MAT3(oPS, *nx, *ny, *nz);
  MAT3(oPE, *nx, *ny, *nz);
  MAT3(oPW, *nx, *ny, *nz);
  MAT3(oPNE, *nx, *ny, *nz);
  MAT3(oPNW, *nx, *ny, *nz);
  MAT3(oPSE, *nx, *ny, *nz);
  MAT3(oPSW, *nx, *ny, *nz);

  MAT3(uPC, *nx, *ny, *nz);
  MAT3(uPN, *nx, *ny, *nz);
  MAT3(uPS, *nx, *ny, *nz);
  MAT3(uPE, *nx, *ny, *nz);
  MAT3(uPW, *nx, *ny, *nz);
  MAT3(uPNE, *nx, *ny, *nz);
  MAT3(uPNW, *nx, *ny, *nz);
  MAT3(uPSE, *nx, *ny, *nz);
  MAT3(uPSW, *nx, *ny, *nz);

  MAT3(dPC, *nx, *ny, *nz);
  MAT3(dPN, *nx, *ny, *nz);
  MAT3(dPS, *nx, *ny, *nz);
  MAT3(dPE, *nx, *ny, *nz);
  MAT3(dPW, *nx, *ny, *nz);
  MAT3(dPNE, *nx, *ny, *nz);
  MAT3(dPNW, *nx, *ny, *nz);
  MAT3(dPSE, *nx, *ny, *nz);
  MAT3(dPSW, *nx, *ny, *nz);

  // define n and determine number of mesh points ***
  const auto nxm1 = *nx - 1;
  const auto nym1 = *ny - 1;
  const auto nzm1 = *nz - 1;

  // fprintf(data, "%s\n", PRINT_FUNC);

  // Build the operator ***
      q.parallel_for(sycl::range<3>(*nx-2, *ny-2, *nz-2), [=](sycl::id<3> idx) {
        const auto ii = idx[0] + 2; // offset by 2 to account for boundary conditions
        const auto jj = idx[1] + 2;
        const auto kk = idx[2] + 2;

        // Define indices


         
      const auto k = 2 * kk - 1;
      const auto j = 2 * jj - 1;
      const auto i = 2 * ii - 1;

        // Index computations
      const auto im1 = i - 1;
      const auto ip1 = i + 1;
      const auto im2 = i - 2;
      const auto ip2 = i + 2;
      const auto jm1 = j - 1;
      const auto jp1 = j + 1;
      const auto jm2 = j - 2;
      const auto jp2 = j + 2;
      const auto km1 = k - 1;
      const auto kp1 = k + 1;
      const auto km2 = k - 2;
      const auto kp2 = k + 2;
      const auto iim1 = ii - 1;
      const auto iip1 = ii + 1;
      const auto jjm1 = jj - 1;
      const auto jjp1 = jj + 1;
      const auto kkm1 = kk - 1;
      const auto kkp1 = kk + 1;

        /* *************************************************************
         * *** > oC;
         * *************************************************************/

        // VAT3( XoC, ii,jj,kk) =
        const auto TMP1_XOC = VAT3(oPN, ii, jj, kk) *
                   (-VAT3(uNE, im1, j, km1) * VAT3(dPW, ii, jj, kk) -
                    VAT3(oNE, im1, j, k) * VAT3(oPW, ii, jj, kk) -
                    VAT3(uSW, i, jp1, k) * VAT3(uPW, ii, jj, kk) -
                    VAT3(uE, im1, jp1, km1) * VAT3(dPNW, ii, jj, kk) -
                    VAT3(oE, im1, jp1, k) * VAT3(oPNW, ii, jj, kk) -
                    VAT3(uW, i, jp1, k) * VAT3(uPNW, ii, jj, kk) -
                    VAT3(uN, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                    VAT3(oN, i, j, k) * VAT3(oPC, ii, jj, kk) -
                    VAT3(uS, i, jp1, k) * VAT3(uPC, ii, jj, kk) -
                    VAT3(uC, i, jp1, km1) * VAT3(dPN, ii, jj, kk) +
                    VAT3(oC, i, jp1, k) * VAT3(oPN, ii, jj, kk) -
                    VAT3(uC, i, jp1, k) * VAT3(uPN, ii, jj, kk) -
                    VAT3(uNW, ip1, j, km1) * VAT3(dPE, ii, jj, kk) -
                    VAT3(oNW, ip1, j, k) * VAT3(oPE, ii, jj, kk) -
                    VAT3(uSE, i, jp1, k) * VAT3(uPE, ii, jj, kk) -
                    VAT3(uW, ip1, jp1, km1) * VAT3(dPNE, ii, jj, kk) -
                    VAT3(oE, i, jp1, k) * VAT3(oPNE, ii, jj, kk) -
                    VAT3(uE, i, jp1, k) * VAT3(uPNE, ii, jj, kk));

        const auto TMP2_XOC = +VAT3(dPN, ii, jj, kk) *
                   (-VAT3(oNE, im1, j, km1) * VAT3(dPW, ii, jj, kk) -
                    VAT3(uSW, i, jp1, km1) * VAT3(oPW, ii, jj, kk) -
                    VAT3(oE, im1, jp1, km1) * VAT3(dPNW, ii, jj, kk) -
                    VAT3(uW, i, jp1, km1) * VAT3(oPNW, ii, jj, kk) -
                    VAT3(oN, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                    VAT3(uS, i, jp1, km1) * VAT3(oPC, ii, jj, kk) +
                    VAT3(oC, i, jp1, km1) * VAT3(dPN, ii, jj, kk) -
                    VAT3(uC, i, jp1, km1) * VAT3(oPN, ii, jj, kk) -
                    VAT3(oNW, ip1, j, km1) * VAT3(dPE, ii, jj, kk) -
                    VAT3(uSE, i, jp1, km1) * VAT3(oPE, ii, jj, kk) -
                    VAT3(oE, i, jp1, km1) * VAT3(dPNE, ii, jj, kk) -
                    VAT3(uE, i, jp1, km1) * VAT3(oPNE, ii, jj, kk));

        const auto TMP3_XOC = +VAT3(dPC, ii, jj, kk) *
                   (-VAT3(oNE, im1, jm1, km1) * VAT3(dPSW, ii, jj, kk) -
                    VAT3(uSW, i, j, km1) * VAT3(oPSW, ii, jj, kk) -
                    VAT3(oE, im1, j, km1) * VAT3(dPW, ii, jj, kk) -
                    VAT3(uW, i, j, km1) * VAT3(oPW, ii, jj, kk) -
                    VAT3(oNW, i, j, km1) * VAT3(dPNW, ii, jj, kk) -
                    VAT3(uNW, i, j, km1) * VAT3(oPNW, ii, jj, kk) -
                    VAT3(oN, i, jm1, km1) * VAT3(dPS, ii, jj, kk) -
                    VAT3(uS, i, j, km1) * VAT3(oPS, ii, jj, kk) +
                    VAT3(oC, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                    VAT3(uC, i, j, km1) * VAT3(oPC, ii, jj, kk) -
                    VAT3(oN, i, j, km1) * VAT3(dPN, ii, jj, kk) -
                    VAT3(uN, i, j, km1) * VAT3(oPN, ii, jj, kk) -
                    VAT3(oNW, ip1, jm1, km1) * VAT3(dPSE, ii, jj, kk) -
                    VAT3(uSE, i, j, km1) * VAT3(oPSE, ii, jj, kk) -
                    VAT3(oE, i, j, km1) * VAT3(dPE, ii, jj, kk) -
                    VAT3(uE, i, j, km1) * VAT3(oPE, ii, jj, kk) -
                    VAT3(oNE, i, j, km1) * VAT3(dPNE, ii, jj, kk) -
                    VAT3(uNE, i, j, km1) * VAT3(oPNE, ii, jj, kk));

        const auto TMP4_XOC = +VAT3(uPC, ii, jj, kk) *
                   (-VAT3(uNE, im1, jm1, k) * VAT3(oPSW, ii, jj, kk) -
                    VAT3(oNE, im1, jm1, kp1) * VAT3(uPSW, ii, jj, kk) -
                    VAT3(uE, im1, j, k) * VAT3(oPW, ii, jj, kk) -
                    VAT3(oE, im1, j, kp1) * VAT3(uPW, ii, jj, kk) -
                    VAT3(uSE, im1, jp1, k) * VAT3(oPNW, ii, jj, kk) -
                    VAT3(oNW, i, j, kp1) * VAT3(uPNW, ii, jj, kk) -
                    VAT3(uN, i, jm1, k) * VAT3(oPS, ii, jj, kk) -
                    VAT3(oN, i, jm1, kp1) * VAT3(uPS, ii, jj, kk) -
                    VAT3(uC, i, j, k) * VAT3(oPC, ii, jj, kk) +
                    VAT3(oC, i, j, kp1) * VAT3(uPC, ii, jj, kk) -
                    VAT3(uS, i, jp1, k) * VAT3(oPN, ii, jj, kk) -
                    VAT3(oN, i, j, kp1) * VAT3(uPN, ii, jj, kk) -
                    VAT3(uNW, ip1, jm1, k) * VAT3(oPSE, ii, jj, kk) -
                    VAT3(oNW, ip1, jm1, kp1) * VAT3(uPSE, ii, jj, kk) -
                    VAT3(uW, ip1, j, k) * VAT3(oPE, ii, jj, kk) -
                    VAT3(oE, i, j, kp1) * VAT3(uPE, ii, jj, kk) -
                    VAT3(uSW, ip1, jp1, k) * VAT3(oPNE, ii, jj, kk) -
                    VAT3(oNE, i, j, kp1) * VAT3(uPNE, ii, jj, kk));

        const auto TMP5_XOC = +VAT3(oPC, ii, jj, kk) *
                   (-VAT3(uW, ip1, j, km1) * VAT3(dPE, ii, jj, kk) -
                    VAT3(oE, im1, j, k) * VAT3(oPW, ii, jj, kk) -
                    VAT3(uSE, im1, jp1, km1) * VAT3(dPNW, ii, jj, kk) -
                    VAT3(uNE, im1, jm1, km1) * VAT3(dPSW, ii, jj, kk) -
                    VAT3(uN, i, jm1, km1) * VAT3(dPS, ii, jj, kk) -
                    VAT3(oNE, im1, jm1, k) * VAT3(oPSW, ii, jj, kk) -
                    VAT3(uE, im1, j, km1) * VAT3(dPW, ii, jj, kk) -
                    VAT3(oNW, ip1, jm1, k) * VAT3(oPSE, ii, jj, kk) -
                    VAT3(uC, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                    VAT3(uNW, ip1, jm1, km1) * VAT3(dPSE, ii, jj, kk) -
                    VAT3(uSW, ip1, jp1, km1) * VAT3(dPNE, ii, jj, kk) -
                    VAT3(uS, i, jp1, km1) * VAT3(dPN, ii, jj, kk) -
                    VAT3(oN, i, jm1, k) * VAT3(oPS, ii, jj, kk) -
                    VAT3(uNE, i, j, k) * VAT3(uPNE, ii, jj, kk) -
                    VAT3(oNE, i, j, k) * VAT3(oPNE, ii, jj, kk) -
                    VAT3(uE, i, j, k) * VAT3(uPE, ii, jj, kk) -
                    VAT3(uSE, i, j, k) * VAT3(uPSE, ii, jj, kk) -
                    VAT3(oN, i, j, k) * VAT3(oPN, ii, jj, kk) -
                    VAT3(oE, i, j, k) * VAT3(oPE, ii, jj, kk) -
                    VAT3(uS, i, j, k) * VAT3(uPS, ii, jj, kk) +
                    VAT3(oC, i, j, k) * VAT3(oPC, ii, jj, kk) -
                    VAT3(uSW, i, j, k) * VAT3(uPSW, ii, jj, kk) -
                    VAT3(uN, i, j, k) * VAT3(uPN, ii, jj, kk) -
                    VAT3(uC, i, j, k) * VAT3(uPC, ii, jj, kk) -
                    VAT3(uW, i, j, k) * VAT3(uPW, ii, jj, kk) -
                    VAT3(oNW, i, j, k) * VAT3(oPNW, ii, jj, kk) -
                    VAT3(uNW, i, j, k) * VAT3(uPNW, ii, jj, kk));

        const auto TMP6_XOC = +VAT3(uPS, ii, jj, kk) *
                   (-VAT3(uE, im1, jm1, k) * VAT3(oPSW, ii, jj, kk) -
                    VAT3(oE, im1, jm1, kp1) * VAT3(uPSW, ii, jj, kk) -
                    VAT3(uSE, im1, j, k) * VAT3(oPW, ii, jj, kk) -
                    VAT3(oNW, i, jm1, kp1) * VAT3(uPW, ii, jj, kk) -
                    VAT3(uC, i, jm1, k) * VAT3(oPS, ii, jj, kk) +
                    VAT3(oC, i, jm1, kp1) * VAT3(uPS, ii, jj, kk) -
                    VAT3(uS, i, j, k) * VAT3(oPC, ii, jj, kk) -
                    VAT3(oN, i, jm1, kp1) * VAT3(uPC, ii, jj, kk) -
                    VAT3(uW, ip1, jm1, k) * VAT3(oPSE, ii, jj, kk) -
                    VAT3(oE, i, jm1, kp1) * VAT3(uPSE, ii, jj, kk) -
                    VAT3(uSW, ip1, j, k) * VAT3(oPE, ii, jj, kk) -
                    VAT3(oNE, i, jm1, kp1) * VAT3(uPE, ii, jj, kk));

        const auto TMP7_XOC = +VAT3(oPS, ii, jj, kk) *
                   (-VAT3(uE, im1, jm1, km1) * VAT3(dPSW, ii, jj, kk) -
                    VAT3(oE, im1, jm1, k) * VAT3(oPSW, ii, jj, kk) -
                    VAT3(uW, i, jm1, k) * VAT3(uPSW, ii, jj, kk) -
                    VAT3(uSE, im1, j, km1) * VAT3(dPW, ii, jj, kk) -
                    VAT3(oNW, i, jm1, k) * VAT3(oPW, ii, jj, kk) -
                    VAT3(uNW, i, jm1, k) * VAT3(uPW, ii, jj, kk) -
                    VAT3(uC, i, jm1, km1) * VAT3(dPS, ii, jj, kk) +
                    VAT3(oC, i, jm1, k) * VAT3(oPS, ii, jj, kk) -
                    VAT3(uC, i, jm1, k) * VAT3(uPS, ii, jj, kk) -
                    VAT3(uS, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                    VAT3(oN, i, jm1, k) * VAT3(oPC, ii, jj, kk) -
                    VAT3(uN, i, jm1, k) * VAT3(uPC, ii, jj, kk) -
                    VAT3(uW, ip1, jm1, km1) * VAT3(dPSE, ii, jj, kk) -
                    VAT3(oE, i, jm1, k) * VAT3(oPSE, ii, jj, kk) -
                    VAT3(uE, i, jm1, k) * VAT3(uPSE, ii, jj, kk) -
                    VAT3(uSW, ip1, j, km1) * VAT3(dPE, ii, jj, kk) -
                    VAT3(oNE, i, jm1, k) * VAT3(oPE, ii, jj, kk) -
                    VAT3(uNE, i, jm1, k) * VAT3(uPE, ii, jj, kk));

        const auto TMP8_XOC = +VAT3(dPS, ii, jj, kk) *
                   (-VAT3(oE, im1, jm1, km1) * VAT3(dPSW, ii, jj, kk) -
                    VAT3(uW, i, jm1, km1) * VAT3(oPSW, ii, jj, kk) -
                    VAT3(oNW, i, jm1, km1) * VAT3(dPW, ii, jj, kk) -
                    VAT3(uNW, i, jm1, km1) * VAT3(oPW, ii, jj, kk) +
                    VAT3(oC, i, jm1, km1) * VAT3(dPS, ii, jj, kk) -
                    VAT3(uC, i, jm1, km1) * VAT3(oPS, ii, jj, kk) -
                    VAT3(oN, i, jm1, km1) * VAT3(dPC, ii, jj, kk) -
                    VAT3(uN, i, jm1, km1) * VAT3(oPC, ii, jj, kk) -
                    VAT3(oE, i, jm1, km1) * VAT3(dPSE, ii, jj, kk) -
                    VAT3(uE, i, jm1, km1) * VAT3(oPSE, ii, jj, kk) -
                    VAT3(oNE, i, jm1, km1) * VAT3(dPE, ii, jj, kk) -
                    VAT3(uNE, i, jm1, km1) * VAT3(oPE, ii, jj, kk));

        const auto TMP9_XOC = +VAT3(uPNW, ii, jj, kk) *
                   (-VAT3(uN, im1, j, k) * VAT3(oPW, ii, jj, kk) -
                    VAT3(oN, im1, j, kp1) * VAT3(uPW, ii, jj, kk) -
                    VAT3(uC, im1, jp1, k) * VAT3(oPNW, ii, jj, kk) +
                    VAT3(oC, im1, jp1, kp1) * VAT3(uPNW, ii, jj, kk) -
                    VAT3(uNW, i, j, k) * VAT3(oPC, ii, jj, kk) -
                    VAT3(oNW, i, j, kp1) * VAT3(uPC, ii, jj, kk) -
                    VAT3(uW, i, jp1, k) * VAT3(oPN, ii, jj, kk) -
                    VAT3(oE, im1, jp1, kp1) * VAT3(uPN, ii, jj, kk));

        const auto TMP10_XOC = +VAT3(oPNW, ii, jj, kk) *
                    (-VAT3(uN, im1, j, km1) * VAT3(dPW, ii, jj, kk) -
                     VAT3(oN, im1, j, k) * VAT3(oPW, ii, jj, kk) -
                     VAT3(uS, im1, jp1, k) * VAT3(uPW, ii, jj, kk) -
                     VAT3(uC, im1, jp1, km1) * VAT3(dPNW, ii, jj, kk) +
                     VAT3(oC, im1, jp1, k) * VAT3(oPNW, ii, jj, kk) -
                     VAT3(uC, im1, jp1, k) * VAT3(uPNW, ii, jj, kk) -
                     VAT3(uNW, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                     VAT3(oNW, i, j, k) * VAT3(oPC, ii, jj, kk) -
                     VAT3(uSE, im1, jp1, k) * VAT3(uPC, ii, jj, kk) -
                     VAT3(uW, i, jp1, km1) * VAT3(dPN, ii, jj, kk) -
                     VAT3(oE, im1, jp1, k) * VAT3(oPN, ii, jj, kk) -
                     VAT3(uE, im1, jp1, k) * VAT3(uPN, ii, jj, kk));

        const auto TMP11_XOC = +VAT3(uPW, ii, jj, kk) *
                    (-VAT3(uN, im1, jm1, k) * VAT3(oPSW, ii, jj, kk) -
                     VAT3(oN, im1, jm1, kp1) * VAT3(uPSW, ii, jj, kk) -
                     VAT3(uC, im1, j, k) * VAT3(oPW, ii, jj, kk) +
                     VAT3(oC, im1, j, kp1) * VAT3(uPW, ii, jj, kk) -
                     VAT3(uS, im1, jp1, k) * VAT3(oPNW, ii, jj, kk) -
                     VAT3(oN, im1, j, kp1) * VAT3(uPNW, ii, jj, kk) -
                     VAT3(uNW, i, jm1, k) * VAT3(oPS, ii, jj, kk) -
                     VAT3(oNW, i, jm1, kp1) * VAT3(uPS, ii, jj, kk) -
                     VAT3(uW, i, j, k) * VAT3(oPC, ii, jj, kk) -
                     VAT3(oE, im1, j, kp1) * VAT3(uPC, ii, jj, kk) -
                     VAT3(uSW, i, jp1, k) * VAT3(oPN, ii, jj, kk) -
                     VAT3(oNE, im1, j, kp1) * VAT3(uPN, ii, jj, kk));

        const auto TMP12_XOC = +VAT3(dPNW, ii, jj, kk) *
                    (-VAT3(oN, im1, j, km1) * VAT3(dPW, ii, jj, kk) -
                     VAT3(uS, im1, jp1, km1) * VAT3(oPW, ii, jj, kk) +
                     VAT3(oC, im1, jp1, km1) * VAT3(dPNW, ii, jj, kk) -
                     VAT3(uC, im1, jp1, km1) * VAT3(oPNW, ii, jj, kk) -
                     VAT3(oNW, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                     VAT3(uSE, im1, jp1, km1) * VAT3(oPC, ii, jj, kk) -
                     VAT3(oE, im1, jp1, km1) * VAT3(dPN, ii, jj, kk) -
                     VAT3(uE, im1, jp1, km1) * VAT3(oPN, ii, jj, kk));

        const auto TMP13_XOC = +VAT3(oPW, ii, jj, kk) *
                    (-VAT3(uN, im1, jm1, km1) * VAT3(dPSW, ii, jj, kk) -
                     VAT3(oN, im1, jm1, k) * VAT3(oPSW, ii, jj, kk) -
                     VAT3(uS, im1, j, k) * VAT3(uPSW, ii, jj, kk) -
                     VAT3(uC, im1, j, km1) * VAT3(dPW, ii, jj, kk) +
                     VAT3(oC, im1, j, k) * VAT3(oPW, ii, jj, kk) -
                     VAT3(uC, im1, j, k) * VAT3(uPW, ii, jj, kk) -
                     VAT3(uS, im1, jp1, km1) * VAT3(dPNW, ii, jj, kk) -
                     VAT3(oN, im1, j, k) * VAT3(oPNW, ii, jj, kk) -
                     VAT3(uN, im1, j, k) * VAT3(uPNW, ii, jj, kk) -
                     VAT3(uNW, i, jm1, km1) * VAT3(dPS, ii, jj, kk) -
                     VAT3(oNW, i, jm1, k) * VAT3(oPS, ii, jj, kk) -
                     VAT3(uSE, im1, j, k) * VAT3(uPS, ii, jj, kk) -
                     VAT3(uW, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                     VAT3(oE, im1, j, k) * VAT3(oPC, ii, jj, kk) -
                     VAT3(uE, im1, j, k) * VAT3(uPC, ii, jj, kk) -
                     VAT3(uSW, i, jp1, km1) * VAT3(dPN, ii, jj, kk) -
                     VAT3(oNE, im1, j, k) * VAT3(oPN, ii, jj, kk) -
                     VAT3(uNE, im1, j, k) * VAT3(uPN, ii, jj, kk));

        const auto TMP14_XOC = +VAT3(uPSW, ii, jj, kk) *
                    (-VAT3(uC, im1, jm1, k) * VAT3(oPSW, ii, jj, kk) +
                     VAT3(oC, im1, jm1, kp1) * VAT3(uPSW, ii, jj, kk) -
                     VAT3(uS, im1, j, k) * VAT3(oPW, ii, jj, kk) -
                     VAT3(oN, im1, jm1, kp1) * VAT3(uPW, ii, jj, kk) -
                     VAT3(uW, i, jm1, k) * VAT3(oPS, ii, jj, kk) -
                     VAT3(oE, im1, jm1, kp1) * VAT3(uPS, ii, jj, kk) -
                     VAT3(uSW, i, j, k) * VAT3(oPC, ii, jj, kk) -
                     VAT3(oNE, im1, jm1, kp1) * VAT3(uPC, ii, jj, kk));

       const auto  TMP15_XOC = +VAT3(oPSW, ii, jj, kk) *
                    (-VAT3(uC, im1, jm1, km1) * VAT3(dPSW, ii, jj, kk) +
                     VAT3(oC, im1, jm1, k) * VAT3(oPSW, ii, jj, kk) -
                     VAT3(uC, im1, jm1, k) * VAT3(uPSW, ii, jj, kk) -
                     VAT3(uS, im1, j, km1) * VAT3(dPW, ii, jj, kk) -
                     VAT3(oN, im1, jm1, k) * VAT3(oPW, ii, jj, kk) -
                     VAT3(uN, im1, jm1, k) * VAT3(uPW, ii, jj, kk) -
                     VAT3(uW, i, jm1, km1) * VAT3(dPS, ii, jj, kk) -
                     VAT3(oE, im1, jm1, k) * VAT3(oPS, ii, jj, kk) -
                     VAT3(uE, im1, jm1, k) * VAT3(uPS, ii, jj, kk) -
                     VAT3(uSW, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                     VAT3(oNE, im1, jm1, k) * VAT3(oPC, ii, jj, kk) -
                     VAT3(uNE, im1, jm1, k) * VAT3(uPC, ii, jj, kk));

        const auto TMP16_XOC = +VAT3(dPW, ii, jj, kk) *
                    (-VAT3(oN, im1, jm1, km1) * VAT3(dPSW, ii, jj, kk) -
                     VAT3(uS, im1, j, km1) * VAT3(oPSW, ii, jj, kk) +
                     VAT3(oC, im1, j, km1) * VAT3(dPW, ii, jj, kk) -
                     VAT3(uC, im1, j, km1) * VAT3(oPW, ii, jj, kk) -
                     VAT3(oN, im1, j, km1) * VAT3(dPNW, ii, jj, kk) -
                     VAT3(uN, im1, j, km1) * VAT3(oPNW, ii, jj, kk) -
                     VAT3(oNW, i, jm1, km1) * VAT3(dPS, ii, jj, kk) -
                     VAT3(uSE, im1, j, km1) * VAT3(oPS, ii, jj, kk) -
                     VAT3(oE, im1, j, km1) * VAT3(dPC, ii, jj, kk) -
                     VAT3(uE, im1, j, km1) * VAT3(oPC, ii, jj, kk) -
                     VAT3(oNE, im1, j, km1) * VAT3(dPN, ii, jj, kk) -
                     VAT3(uNE, im1, j, km1) * VAT3(oPN, ii, jj, kk));

        const auto TMP17_XOC = +VAT3(uPNE, ii, jj, kk) *
                    (-VAT3(uNE, i, j, k) * VAT3(oPC, ii, jj, kk) -
                     VAT3(oNE, i, j, kp1) * VAT3(uPC, ii, jj, kk) -
                     VAT3(uE, i, jp1, k) * VAT3(oPN, ii, jj, kk) -
                     VAT3(oE, i, jp1, kp1) * VAT3(uPN, ii, jj, kk) -
                     VAT3(uN, ip1, j, k) * VAT3(oPE, ii, jj, kk) -
                     VAT3(oN, ip1, j, kp1) * VAT3(uPE, ii, jj, kk) -
                     VAT3(uC, ip1, jp1, k) * VAT3(oPNE, ii, jj, kk) +
                     VAT3(oC, ip1, jp1, kp1) * VAT3(uPNE, ii, jj, kk));

        const auto TMP18_XOC = +VAT3(uPE, ii, jj, kk) *
                    (-VAT3(uNE, i, jm1, k) * VAT3(oPS, ii, jj, kk) -
                     VAT3(oNE, i, jm1, kp1) * VAT3(uPS, ii, jj, kk) -
                     VAT3(uE, i, j, k) * VAT3(oPC, ii, jj, kk) -
                     VAT3(oE, i, j, kp1) * VAT3(uPC, ii, jj, kk) -
                     VAT3(uSE, i, jp1, k) * VAT3(oPN, ii, jj, kk) -
                     VAT3(oNW, ip1, j, kp1) * VAT3(uPN, ii, jj, kk) -
                     VAT3(uN, ip1, jm1, k) * VAT3(oPSE, ii, jj, kk) -
                     VAT3(oN, ip1, jm1, kp1) * VAT3(uPSE, ii, jj, kk) -
                     VAT3(uC, ip1, j, k) * VAT3(oPE, ii, jj, kk) +
                     VAT3(oC, ip1, j, kp1) * VAT3(uPE, ii, jj, kk) -
                     VAT3(uS, ip1, jp1, k) * VAT3(oPNE, ii, jj, kk) -
                     VAT3(oN, ip1, j, kp1) * VAT3(uPNE, ii, jj, kk));

        const auto TMP19_XOC = +VAT3(dPNE, ii, jj, kk) *
                    (-VAT3(oNE, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                     VAT3(uSW, ip1, jp1, km1) * VAT3(oPC, ii, jj, kk) -
                     VAT3(oE, i, jp1, km1) * VAT3(dPN, ii, jj, kk) -
                     VAT3(uW, ip1, jp1, km1) * VAT3(oPN, ii, jj, kk) -
                     VAT3(oN, ip1, j, km1) * VAT3(dPE, ii, jj, kk) -
                     VAT3(uS, ip1, jp1, km1) * VAT3(oPE, ii, jj, kk) +
                     VAT3(oC, ip1, jp1, km1) * VAT3(dPNE, ii, jj, kk) -
                     VAT3(uC, ip1, jp1, km1) * VAT3(oPNE, ii, jj, kk));

        const auto TMP20_XOC = +VAT3(oPNE, ii, jj, kk) *
                    (-VAT3(uNE, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                     VAT3(oNE, i, j, k) * VAT3(oPC, ii, jj, kk) -
                     VAT3(uSW, ip1, jp1, k) * VAT3(uPC, ii, jj, kk) -
                     VAT3(uE, i, jp1, km1) * VAT3(dPN, ii, jj, kk) -
                     VAT3(oE, i, jp1, k) * VAT3(oPN, ii, jj, kk) -
                     VAT3(uW, ip1, jp1, k) * VAT3(uPN, ii, jj, kk) -
                     VAT3(uN, ip1, j, km1) * VAT3(dPE, ii, jj, kk) -
                     VAT3(oN, ip1, j, k) * VAT3(oPE, ii, jj, kk) -
                     VAT3(uS, ip1, jp1, k) * VAT3(uPE, ii, jj, kk) -
                     VAT3(uC, ip1, jp1, km1) * VAT3(dPNE, ii, jj, kk) +
                     VAT3(oC, ip1, jp1, k) * VAT3(oPNE, ii, jj, kk) -
                     VAT3(uC, ip1, jp1, k) * VAT3(uPNE, ii, jj, kk));

        const auto TMP21_XOC = +VAT3(oPSE, ii, jj, kk) *
                    (-VAT3(uE, i, jm1, km1) * VAT3(dPS, ii, jj, kk) -
                     VAT3(oE, i, jm1, k) * VAT3(oPS, ii, jj, kk) -
                     VAT3(uW, ip1, jm1, k) * VAT3(uPS, ii, jj, kk) -
                     VAT3(uSE, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                     VAT3(oNW, ip1, jm1, k) * VAT3(oPC, ii, jj, kk) -
                     VAT3(uNW, ip1, jm1, k) * VAT3(uPC, ii, jj, kk) -
                     VAT3(uC, ip1, jm1, km1) * VAT3(dPSE, ii, jj, kk) +
                     VAT3(oC, ip1, jm1, k) * VAT3(oPSE, ii, jj, kk) -
                     VAT3(uC, ip1, jm1, k) * VAT3(uPSE, ii, jj, kk) -
                     VAT3(uS, ip1, j, km1) * VAT3(dPE, ii, jj, kk) -
                     VAT3(oN, ip1, jm1, k) * VAT3(oPE, ii, jj, kk) -
                     VAT3(uN, ip1, jm1, k) * VAT3(uPE, ii, jj, kk));

        const auto TMP22_XOC = +VAT3(dPSE, ii, jj, kk) *
                    (-VAT3(oE, i, jm1, km1) * VAT3(dPS, ii, jj, kk) -
                     VAT3(uW, ip1, jm1, km1) * VAT3(oPS, ii, jj, kk) -
                     VAT3(oNW, ip1, jm1, km1) * VAT3(dPC, ii, jj, kk) -
                     VAT3(uNW, ip1, jm1, km1) * VAT3(oPC, ii, jj, kk) +
                     VAT3(oC, ip1, jm1, km1) * VAT3(dPSE, ii, jj, kk) -
                     VAT3(uC, ip1, jm1, km1) * VAT3(oPSE, ii, jj, kk) -
                     VAT3(oN, ip1, jm1, km1) * VAT3(dPE, ii, jj, kk) -
                     VAT3(uN, ip1, jm1, km1) * VAT3(oPE, ii, jj, kk));

        const auto TMP23_XOC = +VAT3(uPSE, ii, jj, kk) *
                    (-VAT3(uE, i, jm1, k) * VAT3(oPS, ii, jj, kk) -
                     VAT3(oE, i, jm1, kp1) * VAT3(uPS, ii, jj, kk) -
                     VAT3(uSE, i, j, k) * VAT3(oPC, ii, jj, kk) -
                     VAT3(oNW, ip1, jm1, kp1) * VAT3(uPC, ii, jj, kk) -
                     VAT3(uC, ip1, jm1, k) * VAT3(oPSE, ii, jj, kk) +
                     VAT3(oC, ip1, jm1, kp1) * VAT3(uPSE, ii, jj, kk) -
                     VAT3(uS, ip1, j, k) * VAT3(oPE, ii, jj, kk) -
                     VAT3(oN, ip1, jm1, kp1) * VAT3(uPE, ii, jj, kk));

        const auto TMP24_XOC = +VAT3(oPE, ii, jj, kk) *
                    (-VAT3(uNE, i, jm1, km1) * VAT3(dPS, ii, jj, kk) -
                     VAT3(oNE, i, jm1, k) * VAT3(oPS, ii, jj, kk) -
                     VAT3(uSW, ip1, j, k) * VAT3(uPS, ii, jj, kk) -
                     VAT3(uE, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                     VAT3(oE, i, j, k) * VAT3(oPC, ii, jj, kk) -
                     VAT3(uW, ip1, j, k) * VAT3(uPC, ii, jj, kk) -
                     VAT3(uSE, i, jp1, km1) * VAT3(dPN, ii, jj, kk) -
                     VAT3(oNW, ip1, j, k) * VAT3(oPN, ii, jj, kk) -
                     VAT3(uNW, ip1, j, k) * VAT3(uPN, ii, jj, kk) -
                     VAT3(uN, ip1, jm1, km1) * VAT3(dPSE, ii, jj, kk) -
                     VAT3(oN, ip1, jm1, k) * VAT3(oPSE, ii, jj, kk) -
                     VAT3(uS, ip1, j, k) * VAT3(uPSE, ii, jj, kk) -
                     VAT3(uC, ip1, j, km1) * VAT3(dPE, ii, jj, kk) +
                     VAT3(oC, ip1, j, k) * VAT3(oPE, ii, jj, kk) -
                     VAT3(uC, ip1, j, k) * VAT3(uPE, ii, jj, kk) -
                     VAT3(uS, ip1, jp1, km1) * VAT3(dPNE, ii, jj, kk) -
                     VAT3(oN, ip1, j, k) * VAT3(oPNE, ii, jj, kk) -
                     VAT3(uN, ip1, j, k) * VAT3(uPNE, ii, jj, kk));

        const auto TMP25_XOC = +VAT3(dPE, ii, jj, kk) *
                    (-VAT3(oNE, i, jm1, km1) * VAT3(dPS, ii, jj, kk) -
                     VAT3(uSW, ip1, j, km1) * VAT3(oPS, ii, jj, kk) -
                     VAT3(oE, i, j, km1) * VAT3(dPC, ii, jj, kk) -
                     VAT3(uW, ip1, j, km1) * VAT3(oPC, ii, jj, kk) -
                     VAT3(oNW, ip1, j, km1) * VAT3(dPN, ii, jj, kk) -
                     VAT3(uNW, ip1, j, km1) * VAT3(oPN, ii, jj, kk) -
                     VAT3(oN, ip1, jm1, km1) * VAT3(dPSE, ii, jj, kk) -
                     VAT3(uS, ip1, j, km1) * VAT3(oPSE, ii, jj, kk) +
                     VAT3(oC, ip1, j, km1) * VAT3(dPE, ii, jj, kk) -
                     VAT3(uC, ip1, j, km1) * VAT3(oPE, ii, jj, kk) -
                     VAT3(oN, ip1, j, km1) * VAT3(dPNE, ii, jj, kk) -
                     VAT3(uN, ip1, j, km1) * VAT3(oPNE, ii, jj, kk));

        const auto TMP26_XOC = +VAT3(uPN, ii, jj, kk) *
                    (-VAT3(uNE, im1, j, k) * VAT3(oPW, ii, jj, kk) -
                     VAT3(oNE, im1, j, kp1) * VAT3(uPW, ii, jj, kk) -
                     VAT3(uE, im1, jp1, k) * VAT3(oPNW, ii, jj, kk) -
                     VAT3(oE, im1, jp1, kp1) * VAT3(uPNW, ii, jj, kk) -
                     VAT3(uN, i, j, k) * VAT3(oPC, ii, jj, kk) -
                     VAT3(oN, i, j, kp1) * VAT3(uPC, ii, jj, kk) -
                     VAT3(uC, i, jp1, k) * VAT3(oPN, ii, jj, kk) +
                     VAT3(oC, i, jp1, kp1) * VAT3(uPN, ii, jj, kk) -
                     VAT3(uNW, ip1, j, k) * VAT3(oPE, ii, jj, kk) -
                     VAT3(oNW, ip1, j, kp1) * VAT3(uPE, ii, jj, kk) -
                     VAT3(uW, ip1, jp1, k) * VAT3(oPNE, ii, jj, kk) -
                     VAT3(oE, i, jp1, kp1) * VAT3(uPNE, ii, jj, kk));

        const auto TMP27_XOC = +VAT3(dPSW, ii, jj, kk) *
                    (VAT3(oC, im1, jm1, km1) * VAT3(dPSW, ii, jj, kk) -
                     VAT3(uC, im1, jm1, km1) * VAT3(oPSW, ii, jj, kk) -
                     VAT3(oN, im1, jm1, km1) * VAT3(dPW, ii, jj, kk) -
                     VAT3(uN, im1, jm1, km1) * VAT3(oPW, ii, jj, kk) -
                     VAT3(oE, im1, jm1, km1) * VAT3(dPS, ii, jj, kk) -
                     VAT3(uE, im1, jm1, km1) * VAT3(oPS, ii, jj, kk) -
                     VAT3(oNE, im1, jm1, km1) * VAT3(dPC, ii, jj, kk) -
                     VAT3(uNE, im1, jm1, km1) * VAT3(oPC, ii, jj, kk));

        VAT3(XoC, ii, jj, kk) =
            TMP1_XOC + TMP2_XOC + TMP3_XOC + TMP4_XOC + TMP5_XOC + TMP6_XOC +
            TMP7_XOC + TMP8_XOC + TMP9_XOC + TMP10_XOC + TMP11_XOC + TMP12_XOC +
            TMP13_XOC + TMP14_XOC + TMP15_XOC + TMP16_XOC + TMP17_XOC +
            TMP18_XOC + TMP19_XOC + TMP20_XOC + TMP21_XOC + TMP22_XOC +
            TMP23_XOC + TMP24_XOC + TMP25_XOC + TMP26_XOC + TMP27_XOC;

        // fprintf(data, "%19.12E\n", VAT3(XoC, ii, jj, kk));

        /* *************************************************************
         * *** > OE;
         * *************************************************************/

        // VAT3( XoE, ii,jj,kk) =
        const auto TMP1_XOE = -VAT3(dPS, ii, jj, kk) *
                       (-VAT3(oE, i, jm1, km1) * VAT3(dPSW, iip1, jj, kk) -
                        VAT3(uE, i, jm1, km1) * VAT3(oPSW, iip1, jj, kk) -
                        VAT3(oNE, i, jm1, km1) * VAT3(dPW, iip1, jj, kk) -
                        VAT3(uNE, i, jm1, km1) * VAT3(oPW, iip1, jj, kk))

                   - VAT3(oPS, ii, jj, kk) *
                         (-VAT3(uW, ip1, jm1, km1) * VAT3(dPSW, iip1, jj, kk) -
                          VAT3(oE, i, jm1, k) * VAT3(oPSW, iip1, jj, kk) -
                          VAT3(uE, i, jm1, k) * VAT3(uPSW, iip1, jj, kk) -
                          VAT3(uSW, ip1, j, km1) * VAT3(dPW, iip1, jj, kk) -
                          VAT3(oNE, i, jm1, k) * VAT3(oPW, iip1, jj, kk) -
                          VAT3(uNE, i, jm1, k) * VAT3(uPW, iip1, jj, kk))

                   - VAT3(uPS, ii, jj, kk) *
                         (-VAT3(uW, ip1, jm1, k) * VAT3(oPSW, iip1, jj, kk) -
                          VAT3(oE, i, jm1, kp1) * VAT3(uPSW, iip1, jj, kk) -
                          VAT3(uSW, ip1, j, k) * VAT3(oPW, iip1, jj, kk) -
                          VAT3(oNE, i, jm1, kp1) * VAT3(uPW, iip1, jj, kk));

        const auto TMP2_XOE = -VAT3(dPC, ii, jj, kk) *
                       (-VAT3(oNW, ip1, jm1, km1) * VAT3(dPSW, iip1, jj, kk) -
                        VAT3(uSE, i, j, km1) * VAT3(oPSW, iip1, jj, kk) -
                        VAT3(oE, i, j, km1) * VAT3(dPW, iip1, jj, kk) -
                        VAT3(uE, i, j, km1) * VAT3(oPW, iip1, jj, kk) -
                        VAT3(oNE, i, j, km1) * VAT3(dPNW, iip1, jj, kk) -
                        VAT3(uNE, i, j, km1) * VAT3(oPNW, iip1, jj, kk))

                   - VAT3(oPC, ii, jj, kk) *
                         (-VAT3(uNW, ip1, jm1, km1) * VAT3(dPSW, iip1, jj, kk) -
                          VAT3(oNW, ip1, jm1, k) * VAT3(oPSW, iip1, jj, kk) -
                          VAT3(uSE, i, j, k) * VAT3(uPSW, iip1, jj, kk) -
                          VAT3(uW, ip1, j, km1) * VAT3(dPW, iip1, jj, kk) -
                          VAT3(oE, i, j, k) * VAT3(oPW, iip1, jj, kk) -
                          VAT3(uE, i, j, k) * VAT3(uPW, iip1, jj, kk) -
                          VAT3(uSW, ip1, jp1, km1) * VAT3(dPNW, iip1, jj, kk) -
                          VAT3(oNE, i, j, k) * VAT3(oPNW, iip1, jj, kk) -
                          VAT3(uNE, i, j, k) * VAT3(uPNW, iip1, jj, kk));

        const auto TMP3_XOE = -VAT3(uPC, ii, jj, kk) *
                       (-VAT3(uNW, ip1, jm1, k) * VAT3(oPSW, iip1, jj, kk) -
                        VAT3(oNW, ip1, jm1, kp1) * VAT3(uPSW, iip1, jj, kk) -
                        VAT3(uW, ip1, j, k) * VAT3(oPW, iip1, jj, kk) -
                        VAT3(oE, i, j, kp1) * VAT3(uPW, iip1, jj, kk) -
                        VAT3(uSW, ip1, jp1, k) * VAT3(oPNW, iip1, jj, kk) -
                        VAT3(oNE, i, j, kp1) * VAT3(uPNW, iip1, jj, kk))

                   - VAT3(dPN, ii, jj, kk) *
                         (-VAT3(oNW, ip1, j, km1) * VAT3(dPW, iip1, jj, kk) -
                          VAT3(uSE, i, jp1, km1) * VAT3(oPW, iip1, jj, kk) -
                          VAT3(oE, i, jp1, km1) * VAT3(dPNW, iip1, jj, kk) -
                          VAT3(uE, i, jp1, km1) * VAT3(oPNW, iip1, jj, kk))

                   - VAT3(oPN, ii, jj, kk) *
                         (-VAT3(uNW, ip1, j, km1) * VAT3(dPW, iip1, jj, kk) -
                          VAT3(oNW, ip1, j, k) * VAT3(oPW, iip1, jj, kk) -
                          VAT3(uSE, i, jp1, k) * VAT3(uPW, iip1, jj, kk) -
                          VAT3(uW, ip1, jp1, km1) * VAT3(dPNW, iip1, jj, kk) -
                          VAT3(oE, i, jp1, k) * VAT3(oPNW, iip1, jj, kk) -
                          VAT3(uE, i, jp1, k) * VAT3(uPNW, iip1, jj, kk));

        const auto TMP4_XOE = -VAT3(uPN, ii, jj, kk) *
                       (-VAT3(uNW, ip1, j, k) * VAT3(oPW, iip1, jj, kk) -
                        VAT3(oNW, ip1, j, kp1) * VAT3(uPW, iip1, jj, kk) -
                        VAT3(uW, ip1, jp1, k) * VAT3(oPNW, iip1, jj, kk) -
                        VAT3(oE, i, jp1, kp1) * VAT3(uPNW, iip1, jj, kk))

                   - VAT3(dPSE, ii, jj, kk) *
                         (VAT3(oC, ip1, jm1, km1) * VAT3(dPSW, iip1, jj, kk) -
                          VAT3(uC, ip1, jm1, km1) * VAT3(oPSW, iip1, jj, kk) -
                          VAT3(oN, ip1, jm1, km1) * VAT3(dPW, iip1, jj, kk) -
                          VAT3(uN, ip1, jm1, km1) * VAT3(oPW, iip1, jj, kk) -
                          VAT3(oE, ip1, jm1, km1) * VAT3(dPS, iip1, jj, kk) -
                          VAT3(uE, ip1, jm1, km1) * VAT3(oPS, iip1, jj, kk) -
                          VAT3(oNE, ip1, jm1, km1) * VAT3(dPC, iip1, jj, kk) -
                          VAT3(uNE, ip1, jm1, km1) * VAT3(oPC, iip1, jj, kk));

        const auto TMP5_XOE = -VAT3(oPSE, ii, jj, kk) *
                   (-VAT3(uC, ip1, jm1, km1) * VAT3(dPSW, iip1, jj, kk) +
                    VAT3(oC, ip1, jm1, k) * VAT3(oPSW, iip1, jj, kk) -
                    VAT3(uC, ip1, jm1, k) * VAT3(uPSW, iip1, jj, kk) -
                    VAT3(uS, ip1, j, km1) * VAT3(dPW, iip1, jj, kk) -
                    VAT3(oN, ip1, jm1, k) * VAT3(oPW, iip1, jj, kk) -
                    VAT3(uN, ip1, jm1, k) * VAT3(uPW, iip1, jj, kk) -
                    VAT3(uW, ip2, jm1, km1) * VAT3(dPS, iip1, jj, kk) -
                    VAT3(oE, ip1, jm1, k) * VAT3(oPS, iip1, jj, kk) -
                    VAT3(uE, ip1, jm1, k) * VAT3(uPS, iip1, jj, kk) -
                    VAT3(uSW, ip2, j, km1) * VAT3(dPC, iip1, jj, kk) -
                    VAT3(oNE, ip1, jm1, k) * VAT3(oPC, iip1, jj, kk) -
                    VAT3(uNE, ip1, jm1, k) * VAT3(uPC, iip1, jj, kk));

        const auto TMP6_XOE = -VAT3(uPSE, ii, jj, kk) *
                   (-VAT3(uC, ip1, jm1, k) * VAT3(oPSW, iip1, jj, kk) +
                    VAT3(oC, ip1, jm1, kp1) * VAT3(uPSW, iip1, jj, kk) -
                    VAT3(uS, ip1, j, k) * VAT3(oPW, iip1, jj, kk) -
                    VAT3(oN, ip1, jm1, kp1) * VAT3(uPW, iip1, jj, kk) -
                    VAT3(uW, ip2, jm1, k) * VAT3(oPS, iip1, jj, kk) -
                    VAT3(oE, ip1, jm1, kp1) * VAT3(uPS, iip1, jj, kk) -
                    VAT3(uSW, ip2, j, k) * VAT3(oPC, iip1, jj, kk) -
                    VAT3(oNE, ip1, jm1, kp1) * VAT3(uPC, iip1, jj, kk));

        const auto TMP7_XOE = -VAT3(dPE, ii, jj, kk) *
                   (-VAT3(oN, ip1, jm1, km1) * VAT3(dPSW, iip1, jj, kk) -
                    VAT3(uS, ip1, j, km1) * VAT3(oPSW, iip1, jj, kk) +
                    VAT3(oC, ip1, j, km1) * VAT3(dPW, iip1, jj, kk) -
                    VAT3(uC, ip1, j, km1) * VAT3(oPW, iip1, jj, kk) -
                    VAT3(oN, ip1, j, km1) * VAT3(dPNW, iip1, jj, kk) -
                    VAT3(uN, ip1, j, km1) * VAT3(oPNW, iip1, jj, kk) -
                    VAT3(oNW, ip2, jm1, km1) * VAT3(dPS, iip1, jj, kk) -
                    VAT3(uSE, ip1, j, km1) * VAT3(oPS, iip1, jj, kk) -
                    VAT3(oE, ip1, j, km1) * VAT3(dPC, iip1, jj, kk) -
                    VAT3(uE, ip1, j, km1) * VAT3(oPC, iip1, jj, kk) -
                    VAT3(oNE, ip1, j, km1) * VAT3(dPN, iip1, jj, kk) -
                    VAT3(uNE, ip1, j, km1) * VAT3(oPN, iip1, jj, kk));

        const auto TMP8_XOE = -VAT3(oPE, ii, jj, kk) *
                   (-VAT3(uN, ip1, jm1, km1) * VAT3(dPSW, iip1, jj, kk) -
                    VAT3(oN, ip1, jm1, k) * VAT3(oPSW, iip1, jj, kk) -
                    VAT3(uS, ip1, j, k) * VAT3(uPSW, iip1, jj, kk) -
                    VAT3(uC, ip1, j, km1) * VAT3(dPW, iip1, jj, kk) +
                    VAT3(oC, ip1, j, k) * VAT3(oPW, iip1, jj, kk) -
                    VAT3(uC, ip1, j, k) * VAT3(uPW, iip1, jj, kk) -
                    VAT3(uS, ip1, jp1, km1) * VAT3(dPNW, iip1, jj, kk) -
                    VAT3(oN, ip1, j, k) * VAT3(oPNW, iip1, jj, kk) -
                    VAT3(uN, ip1, j, k) * VAT3(uPNW, iip1, jj, kk) -
                    VAT3(uNW, ip2, jm1, km1) * VAT3(dPS, iip1, jj, kk) -
                    VAT3(oNW, ip2, jm1, k) * VAT3(oPS, iip1, jj, kk) -
                    VAT3(uSE, ip1, j, k) * VAT3(uPS, iip1, jj, kk) -
                    VAT3(uW, ip2, j, km1) * VAT3(dPC, iip1, jj, kk) -
                    VAT3(oE, ip1, j, k) * VAT3(oPC, iip1, jj, kk) -
                    VAT3(uE, ip1, j, k) * VAT3(uPC, iip1, jj, kk) -
                    VAT3(uSW, ip2, jp1, km1) * VAT3(dPN, iip1, jj, kk) -
                    VAT3(oNE, ip1, j, k) * VAT3(oPN, iip1, jj, kk) -
                    VAT3(uNE, ip1, j, k) * VAT3(uPN, iip1, jj, kk));

        const auto TMP9_XOE = -VAT3(uPE, ii, jj, kk) *
                   (-VAT3(uN, ip1, jm1, k) * VAT3(oPSW, iip1, jj, kk) -
                    VAT3(oN, ip1, jm1, kp1) * VAT3(uPSW, iip1, jj, kk) -
                    VAT3(uC, ip1, j, k) * VAT3(oPW, iip1, jj, kk) +
                    VAT3(oC, ip1, j, kp1) * VAT3(uPW, iip1, jj, kk) -
                    VAT3(uS, ip1, jp1, k) * VAT3(oPNW, iip1, jj, kk) -
                    VAT3(oN, ip1, j, kp1) * VAT3(uPNW, iip1, jj, kk) -
                    VAT3(uNW, ip2, jm1, k) * VAT3(oPS, iip1, jj, kk) -
                    VAT3(oNW, ip2, jm1, kp1) * VAT3(uPS, iip1, jj, kk) -
                    VAT3(uW, ip2, j, k) * VAT3(oPC, iip1, jj, kk) -
                    VAT3(oE, ip1, j, kp1) * VAT3(uPC, iip1, jj, kk) -
                    VAT3(uSW, ip2, jp1, k) * VAT3(oPN, iip1, jj, kk) -
                    VAT3(oNE, ip1, j, kp1) * VAT3(uPN, iip1, jj, kk));

        const auto TMP10_XOE = -VAT3(dPNE, ii, jj, kk) *
                    (-VAT3(oN, ip1, j, km1) * VAT3(dPW, iip1, jj, kk) -
                     VAT3(uS, ip1, jp1, km1) * VAT3(oPW, iip1, jj, kk) +
                     VAT3(oC, ip1, jp1, km1) * VAT3(dPNW, iip1, jj, kk) -
                     VAT3(uC, ip1, jp1, km1) * VAT3(oPNW, iip1, jj, kk) -
                     VAT3(oNW, ip2, j, km1) * VAT3(dPC, iip1, jj, kk) -
                     VAT3(uSE, ip1, jp1, km1) * VAT3(oPC, iip1, jj, kk) -
                     VAT3(oE, ip1, jp1, km1) * VAT3(dPN, iip1, jj, kk) -
                     VAT3(uE, ip1, jp1, km1) * VAT3(oPN, iip1, jj, kk));

        const auto TMP11_XOE = -VAT3(oPNE, ii, jj, kk) *
                    (-VAT3(uN, ip1, j, km1) * VAT3(dPW, iip1, jj, kk) -
                     VAT3(oN, ip1, j, k) * VAT3(oPW, iip1, jj, kk) -
                     VAT3(uS, ip1, jp1, k) * VAT3(uPW, iip1, jj, kk) -
                     VAT3(uC, ip1, jp1, km1) * VAT3(dPNW, iip1, jj, kk) +
                     VAT3(oC, ip1, jp1, k) * VAT3(oPNW, iip1, jj, kk) -
                     VAT3(uC, ip1, jp1, k) * VAT3(uPNW, iip1, jj, kk) -
                     VAT3(uNW, ip2, j, km1) * VAT3(dPC, iip1, jj, kk) -
                     VAT3(oNW, ip2, j, k) * VAT3(oPC, iip1, jj, kk) -
                     VAT3(uSE, ip1, jp1, k) * VAT3(uPC, iip1, jj, kk) -
                     VAT3(uW, ip2, jp1, km1) * VAT3(dPN, iip1, jj, kk) -
                     VAT3(oE, ip1, jp1, k) * VAT3(oPN, iip1, jj, kk) -
                     VAT3(uE, ip1, jp1, k) * VAT3(uPN, iip1, jj, kk));

        const auto TMP12_XOE = -VAT3(uPNE, ii, jj, kk) *
                    (-VAT3(uN, ip1, j, k) * VAT3(oPW, iip1, jj, kk) -
                     VAT3(oN, ip1, j, kp1) * VAT3(uPW, iip1, jj, kk) -
                     VAT3(uC, ip1, jp1, k) * VAT3(oPNW, iip1, jj, kk) +
                     VAT3(oC, ip1, jp1, kp1) * VAT3(uPNW, iip1, jj, kk) -
                     VAT3(uNW, ip2, j, k) * VAT3(oPC, iip1, jj, kk) -
                     VAT3(oNW, ip2, j, kp1) * VAT3(uPC, iip1, jj, kk) -
                     VAT3(uW, ip2, jp1, k) * VAT3(oPN, iip1, jj, kk) -
                     VAT3(oE, ip1, jp1, kp1) * VAT3(uPN, iip1, jj, kk));

        VAT3(XoE, ii, jj, kk) = TMP1_XOE + TMP2_XOE + TMP3_XOE + TMP4_XOE +
                                TMP5_XOE + TMP6_XOE + TMP7_XOE + TMP8_XOE +
                                TMP9_XOE + TMP10_XOE + TMP11_XOE + TMP12_XOE;

        // fprintf(data, "%19.12E\n", VAT3(XoE, ii, jj, kk));

        /* *************************************************************
         * *** > ON;
         * *************************************************************/

        // VAT3( XoN, ii,jj,kk) =
        const auto TMP1_XON = -VAT3(dPW, ii, jj, kk) *
                       (-VAT3(oN, im1, j, km1) * VAT3(dPSW, ii, jjp1, kk) -
                        VAT3(uN, im1, j, km1) * VAT3(oPSW, ii, jjp1, kk) -
                        VAT3(oNE, im1, j, km1) * VAT3(dPS, ii, jjp1, kk) -
                        VAT3(uNE, im1, j, km1) * VAT3(oPS, ii, jjp1, kk))

                   - VAT3(oPW, ii, jj, kk) *
                         (-VAT3(uS, im1, jp1, km1) * VAT3(dPSW, ii, jjp1, kk) -
                          VAT3(oN, im1, j, k) * VAT3(oPSW, ii, jjp1, kk) -
                          VAT3(uN, im1, j, k) * VAT3(uPSW, ii, jjp1, kk) -
                          VAT3(uSW, i, jp1, km1) * VAT3(dPS, ii, jjp1, kk) -
                          VAT3(oNE, im1, j, k) * VAT3(oPS, ii, jjp1, kk) -
                          VAT3(uNE, im1, j, k) * VAT3(uPS, ii, jjp1, kk))

                   - VAT3(uPW, ii, jj, kk) *
                         (-VAT3(uS, im1, jp1, k) * VAT3(oPSW, ii, jjp1, kk) -
                          VAT3(oN, im1, j, kp1) * VAT3(uPSW, ii, jjp1, kk) -
                          VAT3(uSW, i, jp1, k) * VAT3(oPS, ii, jjp1, kk) -
                          VAT3(oNE, im1, j, kp1) * VAT3(uPS, ii, jjp1, kk));

        const auto TMP2_XON = -VAT3(dPNW, ii, jj, kk) *
                   (VAT3(oC, im1, jp1, km1) * VAT3(dPSW, ii, jjp1, kk) -
                    VAT3(uC, im1, jp1, km1) * VAT3(oPSW, ii, jjp1, kk) -
                    VAT3(oN, im1, jp1, km1) * VAT3(dPW, ii, jjp1, kk) -
                    VAT3(uN, im1, jp1, km1) * VAT3(oPW, ii, jjp1, kk) -
                    VAT3(oE, im1, jp1, km1) * VAT3(dPS, ii, jjp1, kk) -
                    VAT3(uE, im1, jp1, km1) * VAT3(oPS, ii, jjp1, kk) -
                    VAT3(oNE, im1, jp1, km1) * VAT3(dPC, ii, jjp1, kk) -
                    VAT3(uNE, im1, jp1, km1) * VAT3(oPC, ii, jjp1, kk));

        const auto TMP3_XON = -VAT3(oPNW, ii, jj, kk) *
                   (-VAT3(uC, im1, jp1, km1) * VAT3(dPSW, ii, jjp1, kk) +
                    VAT3(oC, im1, jp1, k) * VAT3(oPSW, ii, jjp1, kk) -
                    VAT3(uC, im1, jp1, k) * VAT3(uPSW, ii, jjp1, kk) -
                    VAT3(uS, im1, jp2, km1) * VAT3(dPW, ii, jjp1, kk) -
                    VAT3(oN, im1, jp1, k) * VAT3(oPW, ii, jjp1, kk) -
                    VAT3(uN, im1, jp1, k) * VAT3(uPW, ii, jjp1, kk) -
                    VAT3(uW, i, jp1, km1) * VAT3(dPS, ii, jjp1, kk) -
                    VAT3(oE, im1, jp1, k) * VAT3(oPS, ii, jjp1, kk) -
                    VAT3(uE, im1, jp1, k) * VAT3(uPS, ii, jjp1, kk) -
                    VAT3(uSW, i, jp2, km1) * VAT3(dPC, ii, jjp1, kk) -
                    VAT3(oNE, im1, jp1, k) * VAT3(oPC, ii, jjp1, kk) -
                    VAT3(uNE, im1, jp1, k) * VAT3(uPC, ii, jjp1, kk));

        const auto TMP4_XON = -VAT3(uPNW, ii, jj, kk) *
                       (-VAT3(uC, im1, jp1, k) * VAT3(oPSW, ii, jjp1, kk) +
                        VAT3(oC, im1, jp1, kp1) * VAT3(uPSW, ii, jjp1, kk) -
                        VAT3(uS, im1, jp2, k) * VAT3(oPW, ii, jjp1, kk) -
                        VAT3(oN, im1, jp1, kp1) * VAT3(uPW, ii, jjp1, kk) -
                        VAT3(uW, i, jp1, k) * VAT3(oPS, ii, jjp1, kk) -
                        VAT3(oE, im1, jp1, kp1) * VAT3(uPS, ii, jjp1, kk) -
                        VAT3(uSW, i, jp2, k) * VAT3(oPC, ii, jjp1, kk) -
                        VAT3(oNE, im1, jp1, kp1) * VAT3(uPC, ii, jjp1, kk))

                   - VAT3(dPC, ii, jj, kk) *
                         (-VAT3(oNW, i, j, km1) * VAT3(dPSW, ii, jjp1, kk) -
                          VAT3(uNW, i, j, km1) * VAT3(oPSW, ii, jjp1, kk) -
                          VAT3(oN, i, j, km1) * VAT3(dPS, ii, jjp1, kk) -
                          VAT3(uN, i, j, km1) * VAT3(oPS, ii, jjp1, kk) -
                          VAT3(oNE, i, j, km1) * VAT3(dPSE, ii, jjp1, kk) -
                          VAT3(uNE, i, j, km1) * VAT3(oPSE, ii, jjp1, kk));

        const auto TMP5_XON = -VAT3(oPC, ii, jj, kk) *
                       (-VAT3(uSE, im1, jp1, km1) * VAT3(dPSW, ii, jjp1, kk) -
                        VAT3(oNW, i, j, k) * VAT3(oPSW, ii, jjp1, kk) -
                        VAT3(uNW, i, j, k) * VAT3(uPSW, ii, jjp1, kk) -
                        VAT3(uS, i, jp1, km1) * VAT3(dPS, ii, jjp1, kk) -
                        VAT3(oN, i, j, k) * VAT3(oPS, ii, jjp1, kk) -
                        VAT3(uN, i, j, k) * VAT3(uPS, ii, jjp1, kk) -
                        VAT3(uSW, ip1, jp1, km1) * VAT3(dPSE, ii, jjp1, kk) -
                        VAT3(oNE, i, j, k) * VAT3(oPSE, ii, jjp1, kk) -
                        VAT3(uNE, i, j, k) * VAT3(uPSE, ii, jjp1, kk))

                   - VAT3(uPC, ii, jj, kk) *
                         (-VAT3(uSE, im1, jp1, k) * VAT3(oPSW, ii, jjp1, kk) -
                          VAT3(oNW, i, j, kp1) * VAT3(uPSW, ii, jjp1, kk) -
                          VAT3(uS, i, jp1, k) * VAT3(oPS, ii, jjp1, kk) -
                          VAT3(oN, i, j, kp1) * VAT3(uPS, ii, jjp1, kk) -
                          VAT3(uSW, ip1, jp1, k) * VAT3(oPSE, ii, jjp1, kk) -
                          VAT3(oNE, i, j, kp1) * VAT3(uPSE, ii, jjp1, kk));

        const auto TMP6_XON = -VAT3(dPN, ii, jj, kk) *
                   (-VAT3(oE, im1, jp1, km1) * VAT3(dPSW, ii, jjp1, kk) -
                    VAT3(uW, i, jp1, km1) * VAT3(oPSW, ii, jjp1, kk) -
                    VAT3(oNW, i, jp1, km1) * VAT3(dPW, ii, jjp1, kk) -
                    VAT3(uNW, i, jp1, km1) * VAT3(oPW, ii, jjp1, kk) +
                    VAT3(oC, i, jp1, km1) * VAT3(dPS, ii, jjp1, kk) -
                    VAT3(uC, i, jp1, km1) * VAT3(oPS, ii, jjp1, kk) -
                    VAT3(oN, i, jp1, km1) * VAT3(dPC, ii, jjp1, kk) -
                    VAT3(uN, i, jp1, km1) * VAT3(oPC, ii, jjp1, kk) -
                    VAT3(oE, i, jp1, km1) * VAT3(dPSE, ii, jjp1, kk) -
                    VAT3(uE, i, jp1, km1) * VAT3(oPSE, ii, jjp1, kk) -
                    VAT3(oNE, i, jp1, km1) * VAT3(dPE, ii, jjp1, kk) -
                    VAT3(uNE, i, jp1, km1) * VAT3(oPE, ii, jjp1, kk));

        const auto TMP7_XON = -VAT3(oPN, ii, jj, kk) *
                   (-VAT3(uE, im1, jp1, km1) * VAT3(dPSW, ii, jjp1, kk) -
                    VAT3(oE, im1, jp1, k) * VAT3(oPSW, ii, jjp1, kk) -
                    VAT3(uW, i, jp1, k) * VAT3(uPSW, ii, jjp1, kk) -
                    VAT3(uSE, im1, jp2, km1) * VAT3(dPW, ii, jjp1, kk) -
                    VAT3(oNW, i, jp1, k) * VAT3(oPW, ii, jjp1, kk) -
                    VAT3(uNW, i, jp1, k) * VAT3(uPW, ii, jjp1, kk) -
                    VAT3(uC, i, jp1, km1) * VAT3(dPS, ii, jjp1, kk) +
                    VAT3(oC, i, jp1, k) * VAT3(oPS, ii, jjp1, kk) -
                    VAT3(uC, i, jp1, k) * VAT3(uPS, ii, jjp1, kk) -
                    VAT3(uS, i, jp2, km1) * VAT3(dPC, ii, jjp1, kk) -
                    VAT3(oN, i, jp1, k) * VAT3(oPC, ii, jjp1, kk) -
                    VAT3(uN, i, jp1, k) * VAT3(uPC, ii, jjp1, kk) -
                    VAT3(uW, ip1, jp1, km1) * VAT3(dPSE, ii, jjp1, kk) -
                    VAT3(oE, i, jp1, k) * VAT3(oPSE, ii, jjp1, kk) -
                    VAT3(uE, i, jp1, k) * VAT3(uPSE, ii, jjp1, kk) -
                    VAT3(uSW, ip1, jp2, km1) * VAT3(dPE, ii, jjp1, kk) -
                    VAT3(oNE, i, jp1, k) * VAT3(oPE, ii, jjp1, kk) -
                    VAT3(uNE, i, jp1, k) * VAT3(uPE, ii, jjp1, kk));

        const auto TMP8_XON = -VAT3(uPN, ii, jj, kk) *
                       (-VAT3(uE, im1, jp1, k) * VAT3(oPSW, ii, jjp1, kk) -
                        VAT3(oE, im1, jp1, kp1) * VAT3(uPSW, ii, jjp1, kk) -
                        VAT3(uSE, im1, jp2, k) * VAT3(oPW, ii, jjp1, kk) -
                        VAT3(oNW, i, jp1, kp1) * VAT3(uPW, ii, jjp1, kk) -
                        VAT3(uC, i, jp1, k) * VAT3(oPS, ii, jjp1, kk) +
                        VAT3(oC, i, jp1, kp1) * VAT3(uPS, ii, jjp1, kk) -
                        VAT3(uS, i, jp2, k) * VAT3(oPC, ii, jjp1, kk) -
                        VAT3(oN, i, jp1, kp1) * VAT3(uPC, ii, jjp1, kk) -
                        VAT3(uW, ip1, jp1, k) * VAT3(oPSE, ii, jjp1, kk) -
                        VAT3(oE, i, jp1, kp1) * VAT3(uPSE, ii, jjp1, kk) -
                        VAT3(uSW, ip1, jp2, k) * VAT3(oPE, ii, jjp1, kk) -
                        VAT3(oNE, i, jp1, kp1) * VAT3(uPE, ii, jjp1, kk))

                   - VAT3(dPE, ii, jj, kk) *
                         (-VAT3(oNW, ip1, j, km1) * VAT3(dPS, ii, jjp1, kk) -
                          VAT3(uNW, ip1, j, km1) * VAT3(oPS, ii, jjp1, kk) -
                          VAT3(oN, ip1, j, km1) * VAT3(dPSE, ii, jjp1, kk) -
                          VAT3(uN, ip1, j, km1) * VAT3(oPSE, ii, jjp1, kk));

        const auto TMP9_XON = -VAT3(oPE, ii, jj, kk) *
                       (-VAT3(uSE, i, jp1, km1) * VAT3(dPS, ii, jjp1, kk) -
                        VAT3(oNW, ip1, j, k) * VAT3(oPS, ii, jjp1, kk) -
                        VAT3(uNW, ip1, j, k) * VAT3(uPS, ii, jjp1, kk) -
                        VAT3(uS, ip1, jp1, km1) * VAT3(dPSE, ii, jjp1, kk) -
                        VAT3(oN, ip1, j, k) * VAT3(oPSE, ii, jjp1, kk) -
                        VAT3(uN, ip1, j, k) * VAT3(uPSE, ii, jjp1, kk))

                   - VAT3(uPE, ii, jj, kk) *
                         (-VAT3(uSE, i, jp1, k) * VAT3(oPS, ii, jjp1, kk) -
                          VAT3(oNW, ip1, j, kp1) * VAT3(uPS, ii, jjp1, kk) -
                          VAT3(uS, ip1, jp1, k) * VAT3(oPSE, ii, jjp1, kk) -
                          VAT3(oN, ip1, j, kp1) * VAT3(uPSE, ii, jjp1, kk));

        const auto TMP10_XON = -VAT3(dPNE, ii, jj, kk) *
                    (-VAT3(oE, i, jp1, km1) * VAT3(dPS, ii, jjp1, kk) -
                     VAT3(uW, ip1, jp1, km1) * VAT3(oPS, ii, jjp1, kk) -
                     VAT3(oNW, ip1, jp1, km1) * VAT3(dPC, ii, jjp1, kk) -
                     VAT3(uNW, ip1, jp1, km1) * VAT3(oPC, ii, jjp1, kk) +
                     VAT3(oC, ip1, jp1, km1) * VAT3(dPSE, ii, jjp1, kk) -
                     VAT3(uC, ip1, jp1, km1) * VAT3(oPSE, ii, jjp1, kk) -
                     VAT3(oN, ip1, jp1, km1) * VAT3(dPE, ii, jjp1, kk) -
                     VAT3(uN, ip1, jp1, km1) * VAT3(oPE, ii, jjp1, kk));

        const auto TMP11_XON = -VAT3(oPNE, ii, jj, kk) *
                    (-VAT3(uE, i, jp1, km1) * VAT3(dPS, ii, jjp1, kk) -
                     VAT3(oE, i, jp1, k) * VAT3(oPS, ii, jjp1, kk) -
                     VAT3(uW, ip1, jp1, k) * VAT3(uPS, ii, jjp1, kk) -
                     VAT3(uSE, i, jp2, km1) * VAT3(dPC, ii, jjp1, kk) -
                     VAT3(oNW, ip1, jp1, k) * VAT3(oPC, ii, jjp1, kk) -
                     VAT3(uNW, ip1, jp1, k) * VAT3(uPC, ii, jjp1, kk) -
                     VAT3(uC, ip1, jp1, km1) * VAT3(dPSE, ii, jjp1, kk) +
                     VAT3(oC, ip1, jp1, k) * VAT3(oPSE, ii, jjp1, kk) -
                     VAT3(uC, ip1, jp1, k) * VAT3(uPSE, ii, jjp1, kk) -
                     VAT3(uS, ip1, jp2, km1) * VAT3(dPE, ii, jjp1, kk) -
                     VAT3(oN, ip1, jp1, k) * VAT3(oPE, ii, jjp1, kk) -
                     VAT3(uN, ip1, jp1, k) * VAT3(uPE, ii, jjp1, kk));

        const auto TMP12_XON = -VAT3(uPNE, ii, jj, kk) *
                    (-VAT3(uE, i, jp1, k) * VAT3(oPS, ii, jjp1, kk) -
                     VAT3(oE, i, jp1, kp1) * VAT3(uPS, ii, jjp1, kk) -
                     VAT3(uSE, i, jp2, k) * VAT3(oPC, ii, jjp1, kk) -
                     VAT3(oNW, ip1, jp1, kp1) * VAT3(uPC, ii, jjp1, kk) -
                     VAT3(uC, ip1, jp1, k) * VAT3(oPSE, ii, jjp1, kk) +
                     VAT3(oC, ip1, jp1, kp1) * VAT3(uPSE, ii, jjp1, kk) -
                     VAT3(uS, ip1, jp2, k) * VAT3(oPE, ii, jjp1, kk) -
                     VAT3(oN, ip1, jp1, kp1) * VAT3(uPE, ii, jjp1, kk));

        VAT3(XoN, ii, jj, kk) = TMP1_XON + TMP2_XON + TMP3_XON + TMP4_XON +
                                TMP5_XON + TMP6_XON + TMP7_XON + TMP8_XON +
                                TMP9_XON + TMP10_XON + TMP11_XON + TMP12_XON;

        // fprintf(data, "%19.12E\n", VAT3(XoN, ii, jj, kk));

        /* *************************************************************
         * *** > UC;
         * *************************************************************/

        // VAT3( XuC, ii,jj,kk) =
        const auto TMP1_XUC = -VAT3(oPSW, ii, jj, kk) *
                       (-VAT3(uC, im1, jm1, k) * VAT3(dPSW, ii, jj, kkp1) -
                        VAT3(uN, im1, jm1, k) * VAT3(dPW, ii, jj, kkp1) -
                        VAT3(uE, im1, jm1, k) * VAT3(dPS, ii, jj, kkp1) -
                        VAT3(uNE, im1, jm1, k) * VAT3(dPC, ii, jj, kkp1))

                   - VAT3(uPSW, ii, jj, kk) *
                         (VAT3(oC, im1, jm1, kp1) * VAT3(dPSW, ii, jj, kkp1) -
                          VAT3(uC, im1, jm1, kp1) * VAT3(oPSW, ii, jj, kkp1) -
                          VAT3(oN, im1, jm1, kp1) * VAT3(dPW, ii, jj, kkp1) -
                          VAT3(uN, im1, jm1, kp1) * VAT3(oPW, ii, jj, kkp1) -
                          VAT3(oE, im1, jm1, kp1) * VAT3(dPS, ii, jj, kkp1) -
                          VAT3(uE, im1, jm1, kp1) * VAT3(oPS, ii, jj, kkp1) -
                          VAT3(oNE, im1, jm1, kp1) * VAT3(dPC, ii, jj, kkp1) -
                          VAT3(uNE, im1, jm1, kp1) * VAT3(oPC, ii, jj, kkp1));

        const auto TMP2_XUC = -VAT3(oPW, ii, jj, kk) *
                   (-VAT3(uS, im1, j, k) * VAT3(dPSW, ii, jj, kkp1) -
                    VAT3(uC, im1, j, k) * VAT3(dPW, ii, jj, kkp1) -
                    VAT3(uN, im1, j, k) * VAT3(dPNW, ii, jj, kkp1) -
                    VAT3(uSE, im1, j, k) * VAT3(dPS, ii, jj, kkp1) -
                    VAT3(uE, im1, j, k) * VAT3(dPC, ii, jj, kkp1) -
                    VAT3(uNE, im1, j, k) * VAT3(dPN, ii, jj, kkp1));

        const auto TMP3_XUC = -VAT3(uPW, ii, jj, kk) *
                       (-VAT3(oN, im1, jm1, kp1) * VAT3(dPSW, ii, jj, kkp1) -
                        VAT3(uS, im1, j, kp1) * VAT3(oPSW, ii, jj, kkp1) +
                        VAT3(oC, im1, j, kp1) * VAT3(dPW, ii, jj, kkp1) -
                        VAT3(uC, im1, j, kp1) * VAT3(oPW, ii, jj, kkp1) -
                        VAT3(oN, im1, j, kp1) * VAT3(dPNW, ii, jj, kkp1) -
                        VAT3(uN, im1, j, kp1) * VAT3(oPNW, ii, jj, kkp1) -
                        VAT3(oNW, i, jm1, kp1) * VAT3(dPS, ii, jj, kkp1) -
                        VAT3(uSE, im1, j, kp1) * VAT3(oPS, ii, jj, kkp1) -
                        VAT3(oE, im1, j, kp1) * VAT3(dPC, ii, jj, kkp1) -
                        VAT3(uE, im1, j, kp1) * VAT3(oPC, ii, jj, kkp1) -
                        VAT3(oNE, im1, j, kp1) * VAT3(dPN, ii, jj, kkp1) -
                        VAT3(uNE, im1, j, kp1) * VAT3(oPN, ii, jj, kkp1))

                   - VAT3(oPNW, ii, jj, kk) *
                         (-VAT3(uS, im1, jp1, k) * VAT3(dPW, ii, jj, kkp1) -
                          VAT3(uC, im1, jp1, k) * VAT3(dPNW, ii, jj, kkp1) -
                          VAT3(uSE, im1, jp1, k) * VAT3(dPC, ii, jj, kkp1) -
                          VAT3(uE, im1, jp1, k) * VAT3(dPN, ii, jj, kkp1));

        const auto TMP4_XUC = -VAT3(uPNW, ii, jj, kk) *
                       (-VAT3(oN, im1, j, kp1) * VAT3(dPW, ii, jj, kkp1) -
                        VAT3(uS, im1, jp1, kp1) * VAT3(oPW, ii, jj, kkp1) +
                        VAT3(oC, im1, jp1, kp1) * VAT3(dPNW, ii, jj, kkp1) -
                        VAT3(uC, im1, jp1, kp1) * VAT3(oPNW, ii, jj, kkp1) -
                        VAT3(oNW, i, j, kp1) * VAT3(dPC, ii, jj, kkp1) -
                        VAT3(uSE, im1, jp1, kp1) * VAT3(oPC, ii, jj, kkp1) -
                        VAT3(oE, im1, jp1, kp1) * VAT3(dPN, ii, jj, kkp1) -
                        VAT3(uE, im1, jp1, kp1) * VAT3(oPN, ii, jj, kkp1))

                   - VAT3(oPS, ii, jj, kk) *
                         (-VAT3(uW, i, jm1, k) * VAT3(dPSW, ii, jj, kkp1) -
                          VAT3(uNW, i, jm1, k) * VAT3(dPW, ii, jj, kkp1) -
                          VAT3(uC, i, jm1, k) * VAT3(dPS, ii, jj, kkp1) -
                          VAT3(uN, i, jm1, k) * VAT3(dPC, ii, jj, kkp1) -
                          VAT3(uE, i, jm1, k) * VAT3(dPSE, ii, jj, kkp1) -
                          VAT3(uNE, i, jm1, k) * VAT3(dPE, ii, jj, kkp1));

        const auto TMP5_XUC = -VAT3(uPS, ii, jj, kk) *
                   (-VAT3(oE, im1, jm1, kp1) * VAT3(dPSW, ii, jj, kkp1) -
                    VAT3(uW, i, jm1, kp1) * VAT3(oPSW, ii, jj, kkp1) -
                    VAT3(oNW, i, jm1, kp1) * VAT3(dPW, ii, jj, kkp1) -
                    VAT3(uNW, i, jm1, kp1) * VAT3(oPW, ii, jj, kkp1) +
                    VAT3(oC, i, jm1, kp1) * VAT3(dPS, ii, jj, kkp1) -
                    VAT3(uC, i, jm1, kp1) * VAT3(oPS, ii, jj, kkp1) -
                    VAT3(oN, i, jm1, kp1) * VAT3(dPC, ii, jj, kkp1) -
                    VAT3(uN, i, jm1, kp1) * VAT3(oPC, ii, jj, kkp1) -
                    VAT3(oE, i, jm1, kp1) * VAT3(dPSE, ii, jj, kkp1) -
                    VAT3(uE, i, jm1, kp1) * VAT3(oPSE, ii, jj, kkp1) -
                    VAT3(oNE, i, jm1, kp1) * VAT3(dPE, ii, jj, kkp1) -
                    VAT3(uNE, i, jm1, kp1) * VAT3(oPE, ii, jj, kkp1));

        const auto TMP6_XUC = -VAT3(oPC, ii, jj, kk) *
                   (-VAT3(uSW, i, j, k) * VAT3(dPSW, ii, jj, kkp1) -
                    VAT3(uW, i, j, k) * VAT3(dPW, ii, jj, kkp1) -
                    VAT3(uNW, i, j, k) * VAT3(dPNW, ii, jj, kkp1) -
                    VAT3(uS, i, j, k) * VAT3(dPS, ii, jj, kkp1) -
                    VAT3(uC, i, j, k) * VAT3(dPC, ii, jj, kkp1) -
                    VAT3(uN, i, j, k) * VAT3(dPN, ii, jj, kkp1) -
                    VAT3(uSE, i, j, k) * VAT3(dPSE, ii, jj, kkp1) -
                    VAT3(uE, i, j, k) * VAT3(dPE, ii, jj, kkp1) -
                    VAT3(uNE, i, j, k) * VAT3(dPNE, ii, jj, kkp1));

        const auto TMP7_XUC = -VAT3(uPC, ii, jj, kk) *
                   (-VAT3(oNE, im1, jm1, kp1) * VAT3(dPSW, ii, jj, kkp1) -
                    VAT3(uSW, i, j, kp1) * VAT3(oPSW, ii, jj, kkp1) -
                    VAT3(oE, im1, j, kp1) * VAT3(dPW, ii, jj, kkp1) -
                    VAT3(uW, i, j, kp1) * VAT3(oPW, ii, jj, kkp1) -
                    VAT3(oNW, i, j, kp1) * VAT3(dPNW, ii, jj, kkp1) -
                    VAT3(uNW, i, j, kp1) * VAT3(oPNW, ii, jj, kkp1) -
                    VAT3(oN, i, jm1, kp1) * VAT3(dPS, ii, jj, kkp1) -
                    VAT3(uS, i, j, kp1) * VAT3(oPS, ii, jj, kkp1) +
                    VAT3(oC, i, j, kp1) * VAT3(dPC, ii, jj, kkp1) -
                    VAT3(uC, i, j, kp1) * VAT3(oPC, ii, jj, kkp1) -
                    VAT3(oN, i, j, kp1) * VAT3(dPN, ii, jj, kkp1) -
                    VAT3(uN, i, j, kp1) * VAT3(oPN, ii, jj, kkp1) -
                    VAT3(oNW, ip1, jm1, kp1) * VAT3(dPSE, ii, jj, kkp1) -
                    VAT3(uSE, i, j, kp1) * VAT3(oPSE, ii, jj, kkp1) -
                    VAT3(oE, i, j, kp1) * VAT3(dPE, ii, jj, kkp1) -
                    VAT3(uE, i, j, kp1) * VAT3(oPE, ii, jj, kkp1) -
                    VAT3(oNE, i, j, kp1) * VAT3(dPNE, ii, jj, kkp1) -
                    VAT3(uNE, i, j, kp1) * VAT3(oPNE, ii, jj, kkp1));

        const auto TMP8_XUC = -VAT3(oPN, ii, jj, kk) *
                   (-VAT3(uSW, i, jp1, k) * VAT3(dPW, ii, jj, kkp1) -
                    VAT3(uW, i, jp1, k) * VAT3(dPNW, ii, jj, kkp1) -
                    VAT3(uS, i, jp1, k) * VAT3(dPC, ii, jj, kkp1) -
                    VAT3(uC, i, jp1, k) * VAT3(dPN, ii, jj, kkp1) -
                    VAT3(uSE, i, jp1, k) * VAT3(dPE, ii, jj, kkp1) -
                    VAT3(uE, i, jp1, k) * VAT3(dPNE, ii, jj, kkp1));

        const auto TMP9_XUC = -VAT3(uPN, ii, jj, kk) *
                       (-VAT3(oNE, im1, j, kp1) * VAT3(dPW, ii, jj, kkp1) -
                        VAT3(uSW, i, jp1, kp1) * VAT3(oPW, ii, jj, kkp1) -
                        VAT3(oE, im1, jp1, kp1) * VAT3(dPNW, ii, jj, kkp1) -
                        VAT3(uW, i, jp1, kp1) * VAT3(oPNW, ii, jj, kkp1) -
                        VAT3(oN, i, j, kp1) * VAT3(dPC, ii, jj, kkp1) -
                        VAT3(uS, i, jp1, kp1) * VAT3(oPC, ii, jj, kkp1) +
                        VAT3(oC, i, jp1, kp1) * VAT3(dPN, ii, jj, kkp1) -
                        VAT3(uC, i, jp1, kp1) * VAT3(oPN, ii, jj, kkp1) -
                        VAT3(oNW, ip1, j, kp1) * VAT3(dPE, ii, jj, kkp1) -
                        VAT3(uSE, i, jp1, kp1) * VAT3(oPE, ii, jj, kkp1) -
                        VAT3(oE, i, jp1, kp1) * VAT3(dPNE, ii, jj, kkp1) -
                        VAT3(uE, i, jp1, kp1) * VAT3(oPNE, ii, jj, kkp1))

                   - VAT3(oPSE, ii, jj, kk) *
                         (-VAT3(uW, ip1, jm1, k) * VAT3(dPS, ii, jj, kkp1) -
                          VAT3(uNW, ip1, jm1, k) * VAT3(dPC, ii, jj, kkp1) -
                          VAT3(uC, ip1, jm1, k) * VAT3(dPSE, ii, jj, kkp1) -
                          VAT3(uN, ip1, jm1, k) * VAT3(dPE, ii, jj, kkp1));

        const auto TMP10_XUC = -VAT3(uPSE, ii, jj, kk) *
                        (-VAT3(oE, i, jm1, kp1) * VAT3(dPS, ii, jj, kkp1) -
                         VAT3(uW, ip1, jm1, kp1) * VAT3(oPS, ii, jj, kkp1) -
                         VAT3(oNW, ip1, jm1, kp1) * VAT3(dPC, ii, jj, kkp1) -
                         VAT3(uNW, ip1, jm1, kp1) * VAT3(oPC, ii, jj, kkp1) +
                         VAT3(oC, ip1, jm1, kp1) * VAT3(dPSE, ii, jj, kkp1) -
                         VAT3(uC, ip1, jm1, kp1) * VAT3(oPSE, ii, jj, kkp1) -
                         VAT3(oN, ip1, jm1, kp1) * VAT3(dPE, ii, jj, kkp1) -
                         VAT3(uN, ip1, jm1, kp1) * VAT3(oPE, ii, jj, kkp1))

                    - VAT3(oPE, ii, jj, kk) *
                          (-VAT3(uSW, ip1, j, k) * VAT3(dPS, ii, jj, kkp1) -
                           VAT3(uW, ip1, j, k) * VAT3(dPC, ii, jj, kkp1) -
                           VAT3(uNW, ip1, j, k) * VAT3(dPN, ii, jj, kkp1) -
                           VAT3(uS, ip1, j, k) * VAT3(dPSE, ii, jj, kkp1) -
                           VAT3(uC, ip1, j, k) * VAT3(dPE, ii, jj, kkp1) -
                           VAT3(uN, ip1, j, k) * VAT3(dPNE, ii, jj, kkp1));

        const auto TMP11_XUC = -VAT3(uPE, ii, jj, kk) *
                    (-VAT3(oNE, i, jm1, kp1) * VAT3(dPS, ii, jj, kkp1) -
                     VAT3(uSW, ip1, j, kp1) * VAT3(oPS, ii, jj, kkp1) -
                     VAT3(oE, i, j, kp1) * VAT3(dPC, ii, jj, kkp1) -
                     VAT3(uW, ip1, j, kp1) * VAT3(oPC, ii, jj, kkp1) -
                     VAT3(oNW, ip1, j, kp1) * VAT3(dPN, ii, jj, kkp1) -
                     VAT3(uNW, ip1, j, kp1) * VAT3(oPN, ii, jj, kkp1) -
                     VAT3(oN, ip1, jm1, kp1) * VAT3(dPSE, ii, jj, kkp1) -
                     VAT3(uS, ip1, j, kp1) * VAT3(oPSE, ii, jj, kkp1) +
                     VAT3(oC, ip1, j, kp1) * VAT3(dPE, ii, jj, kkp1) -
                     VAT3(uC, ip1, j, kp1) * VAT3(oPE, ii, jj, kkp1) -
                     VAT3(oN, ip1, j, kp1) * VAT3(dPNE, ii, jj, kkp1) -
                     VAT3(uN, ip1, j, kp1) * VAT3(oPNE, ii, jj, kkp1));

        const auto TMP12_XUC = -VAT3(oPNE, ii, jj, kk) *
                        (-VAT3(uSW, ip1, jp1, k) * VAT3(dPC, ii, jj, kkp1) -
                         VAT3(uW, ip1, jp1, k) * VAT3(dPN, ii, jj, kkp1) -
                         VAT3(uS, ip1, jp1, k) * VAT3(dPE, ii, jj, kkp1) -
                         VAT3(uC, ip1, jp1, k) * VAT3(dPNE, ii, jj, kkp1))

                    - VAT3(uPNE, ii, jj, kk) *
                          (-VAT3(oNE, i, j, kp1) * VAT3(dPC, ii, jj, kkp1) -
                           VAT3(uSW, ip1, jp1, kp1) * VAT3(oPC, ii, jj, kkp1) -
                           VAT3(oE, i, jp1, kp1) * VAT3(dPN, ii, jj, kkp1) -
                           VAT3(uW, ip1, jp1, kp1) * VAT3(oPN, ii, jj, kkp1) -
                           VAT3(oN, ip1, j, kp1) * VAT3(dPE, ii, jj, kkp1) -
                           VAT3(uS, ip1, jp1, kp1) * VAT3(oPE, ii, jj, kkp1) +
                           VAT3(oC, ip1, jp1, kp1) * VAT3(dPNE, ii, jj, kkp1) -
                           VAT3(uC, ip1, jp1, kp1) * VAT3(oPNE, ii, jj, kkp1));

        VAT3(XuC, ii, jj, kk) = TMP1_XUC + TMP2_XUC + TMP3_XUC + TMP4_XUC +
                                TMP5_XUC + TMP6_XUC + TMP7_XUC + TMP8_XUC +
                                TMP9_XUC + TMP10_XUC + TMP11_XUC + TMP12_XUC;

        // fprintf(data, "%19.12E\n", VAT3(XuC, ii, jj, kk));

        /* *************************************************************
         * *** > ONE;
         * *************************************************************/

        // VAT3(XoNE, ii,jj,kk) =
        const auto TMP1_XONE =
            -VAT3(dPC, ii, jj, kk) *
                (-VAT3(oNE, i, j, km1) * VAT3(dPSW, iip1, jjp1, kk) -
                 VAT3(uNE, i, j, km1) * VAT3(oPSW, iip1, jjp1, kk))

            - VAT3(oPC, ii, jj, kk) *
                  (-VAT3(uSW, ip1, jp1, km1) * VAT3(dPSW, iip1, jjp1, kk) -
                   VAT3(oNE, i, j, k) * VAT3(oPSW, iip1, jjp1, kk) -
                   VAT3(uNE, i, j, k) * VAT3(uPSW, iip1, jjp1, kk))

            - VAT3(uPC, ii, jj, kk) *
                  (-VAT3(uSW, ip1, jp1, k) * VAT3(oPSW, iip1, jjp1, kk) -
                   VAT3(oNE, i, j, kp1) * VAT3(uPSW, iip1, jjp1, kk))

            - VAT3(dPN, ii, jj, kk) *
                  (-VAT3(oE, i, jp1, km1) * VAT3(dPSW, iip1, jjp1, kk) -
                   VAT3(uE, i, jp1, km1) * VAT3(oPSW, iip1, jjp1, kk) -
                   VAT3(oNE, i, jp1, km1) * VAT3(dPW, iip1, jjp1, kk) -
                   VAT3(uNE, i, jp1, km1) * VAT3(oPW, iip1, jjp1, kk));

        const auto TMP2_XONE = -VAT3(oPN, ii, jj, kk) *
                        (-VAT3(uW, ip1, jp1, km1) * VAT3(dPSW, iip1, jjp1, kk) -
                         VAT3(oE, i, jp1, k) * VAT3(oPSW, iip1, jjp1, kk) -
                         VAT3(uE, i, jp1, k) * VAT3(uPSW, iip1, jjp1, kk) -
                         VAT3(uSW, ip1, jp2, km1) * VAT3(dPW, iip1, jjp1, kk) -
                         VAT3(oNE, i, jp1, k) * VAT3(oPW, iip1, jjp1, kk) -
                         VAT3(uNE, i, jp1, k) * VAT3(uPW, iip1, jjp1, kk))

                    - VAT3(uPN, ii, jj, kk) *
                          (-VAT3(uW, ip1, jp1, k) * VAT3(oPSW, iip1, jjp1, kk) -
                           VAT3(oE, i, jp1, kp1) * VAT3(uPSW, iip1, jjp1, kk) -
                           VAT3(uSW, ip1, jp2, k) * VAT3(oPW, iip1, jjp1, kk) -
                           VAT3(oNE, i, jp1, kp1) * VAT3(uPW, iip1, jjp1, kk))

                    - VAT3(dPE, ii, jj, kk) *
                          (-VAT3(oN, ip1, j, km1) * VAT3(dPSW, iip1, jjp1, kk) -
                           VAT3(uN, ip1, j, km1) * VAT3(oPSW, iip1, jjp1, kk) -
                           VAT3(oNE, ip1, j, km1) * VAT3(dPS, iip1, jjp1, kk) -
                           VAT3(uNE, ip1, j, km1) * VAT3(oPS, iip1, jjp1, kk));

        const auto TMP3_XONE = -VAT3(oPE, ii, jj, kk) *
                        (-VAT3(uS, ip1, jp1, km1) * VAT3(dPSW, iip1, jjp1, kk) -
                         VAT3(oN, ip1, j, k) * VAT3(oPSW, iip1, jjp1, kk) -
                         VAT3(uN, ip1, j, k) * VAT3(uPSW, iip1, jjp1, kk) -
                         VAT3(uSW, ip2, jp1, km1) * VAT3(dPS, iip1, jjp1, kk) -
                         VAT3(oNE, ip1, j, k) * VAT3(oPS, iip1, jjp1, kk) -
                         VAT3(uNE, ip1, j, k) * VAT3(uPS, iip1, jjp1, kk))

                    - VAT3(uPE, ii, jj, kk) *
                          (-VAT3(uS, ip1, jp1, k) * VAT3(oPSW, iip1, jjp1, kk) -
                           VAT3(oN, ip1, j, kp1) * VAT3(uPSW, iip1, jjp1, kk) -
                           VAT3(uSW, ip2, jp1, k) * VAT3(oPS, iip1, jjp1, kk) -
                           VAT3(oNE, ip1, j, kp1) * VAT3(uPS, iip1, jjp1, kk));

        const auto TMP4_XONE = -VAT3(dPNE, ii, jj, kk) *
                    (VAT3(oC, ip1, jp1, km1) * VAT3(dPSW, iip1, jjp1, kk) -
                     VAT3(uC, ip1, jp1, km1) * VAT3(oPSW, iip1, jjp1, kk) -
                     VAT3(oN, ip1, jp1, km1) * VAT3(dPW, iip1, jjp1, kk) -
                     VAT3(uN, ip1, jp1, km1) * VAT3(oPW, iip1, jjp1, kk) -
                     VAT3(oE, ip1, jp1, km1) * VAT3(dPS, iip1, jjp1, kk) -
                     VAT3(uE, ip1, jp1, km1) * VAT3(oPS, iip1, jjp1, kk) -
                     VAT3(oNE, ip1, jp1, km1) * VAT3(dPC, iip1, jjp1, kk) -
                     VAT3(uNE, ip1, jp1, km1) * VAT3(oPC, iip1, jjp1, kk));

        const auto TMP5_XONE = -VAT3(oPNE, ii, jj, kk) *
                    (-VAT3(uC, ip1, jp1, km1) * VAT3(dPSW, iip1, jjp1, kk) +
                     VAT3(oC, ip1, jp1, k) * VAT3(oPSW, iip1, jjp1, kk) -
                     VAT3(uC, ip1, jp1, k) * VAT3(uPSW, iip1, jjp1, kk) -
                     VAT3(uS, ip1, jp2, km1) * VAT3(dPW, iip1, jjp1, kk) -
                     VAT3(oN, ip1, jp1, k) * VAT3(oPW, iip1, jjp1, kk) -
                     VAT3(uN, ip1, jp1, k) * VAT3(uPW, iip1, jjp1, kk) -
                     VAT3(uW, ip2, jp1, km1) * VAT3(dPS, iip1, jjp1, kk) -
                     VAT3(oE, ip1, jp1, k) * VAT3(oPS, iip1, jjp1, kk) -
                     VAT3(uE, ip1, jp1, k) * VAT3(uPS, iip1, jjp1, kk) -
                     VAT3(uSW, ip2, jp2, km1) * VAT3(dPC, iip1, jjp1, kk) -
                     VAT3(oNE, ip1, jp1, k) * VAT3(oPC, iip1, jjp1, kk) -
                     VAT3(uNE, ip1, jp1, k) * VAT3(uPC, iip1, jjp1, kk));

        const auto TMP6_XONE = -VAT3(uPNE, ii, jj, kk) *
                    (-VAT3(uC, ip1, jp1, k) * VAT3(oPSW, iip1, jjp1, kk) +
                     VAT3(oC, ip1, jp1, kp1) * VAT3(uPSW, iip1, jjp1, kk) -
                     VAT3(uS, ip1, jp2, k) * VAT3(oPW, iip1, jjp1, kk) -
                     VAT3(oN, ip1, jp1, kp1) * VAT3(uPW, iip1, jjp1, kk) -
                     VAT3(uW, ip2, jp1, k) * VAT3(oPS, iip1, jjp1, kk) -
                     VAT3(oE, ip1, jp1, kp1) * VAT3(uPS, iip1, jjp1, kk) -
                     VAT3(uSW, ip2, jp2, k) * VAT3(oPC, iip1, jjp1, kk) -
                     VAT3(oNE, ip1, jp1, kp1) * VAT3(uPC, iip1, jjp1, kk));

        VAT3(XoNE, ii, jj, kk) = TMP1_XONE + TMP2_XONE + TMP3_XONE + TMP4_XONE +
                                 TMP5_XONE + TMP6_XONE;

        // fprintf(data, "%19.12E\n", VAT3(XoNE, ii, jj, kk));

        /* *************************************************************
         * *** > ONW;
         * *************************************************************/

        // VAT3(XoNW, ii,jj,kk) =
        const auto TMP1_XONW = -VAT3(dPW, ii, jj, kk) *
                        (-VAT3(oNW, im1, j, km1) * VAT3(dPS, iim1, jjp1, kk) -
                         VAT3(uNW, im1, j, km1) * VAT3(oPS, iim1, jjp1, kk) -
                         VAT3(oN, im1, j, km1) * VAT3(dPSE, iim1, jjp1, kk) -
                         VAT3(uN, im1, j, km1) * VAT3(oPSE, iim1, jjp1, kk))

                    -
                    VAT3(oPW, ii, jj, kk) *
                        (-VAT3(uSE, im2, jp1, km1) * VAT3(dPS, iim1, jjp1, kk) -
                         VAT3(oNW, im1, j, k) * VAT3(oPS, iim1, jjp1, kk) -
                         VAT3(uNW, im1, j, k) * VAT3(uPS, iim1, jjp1, kk) -
                         VAT3(uS, im1, jp1, km1) * VAT3(dPSE, iim1, jjp1, kk) -
                         VAT3(oN, im1, j, k) * VAT3(oPSE, iim1, jjp1, kk) -
                         VAT3(uN, im1, j, k) * VAT3(uPSE, iim1, jjp1, kk));

        const auto TMP2_XONW = -VAT3(uPW, ii, jj, kk) *
                        (-VAT3(uSE, im2, jp1, k) * VAT3(oPS, iim1, jjp1, kk) -
                         VAT3(oNW, im1, j, kp1) * VAT3(uPS, iim1, jjp1, kk) -
                         VAT3(uS, im1, jp1, k) * VAT3(oPSE, iim1, jjp1, kk) -
                         VAT3(oN, im1, j, kp1) * VAT3(uPSE, iim1, jjp1, kk))

                    -
                    VAT3(dPNW, ii, jj, kk) *
                        (-VAT3(oE, im2, jp1, km1) * VAT3(dPS, iim1, jjp1, kk) -
                         VAT3(uW, im1, jp1, km1) * VAT3(oPS, iim1, jjp1, kk) -
                         VAT3(oNW, im1, jp1, km1) * VAT3(dPC, iim1, jjp1, kk) -
                         VAT3(uNW, im1, jp1, km1) * VAT3(oPC, iim1, jjp1, kk) +
                         VAT3(oC, im1, jp1, km1) * VAT3(dPSE, iim1, jjp1, kk) -
                         VAT3(uC, im1, jp1, km1) * VAT3(oPSE, iim1, jjp1, kk) -
                         VAT3(oN, im1, jp1, km1) * VAT3(dPE, iim1, jjp1, kk) -
                         VAT3(uN, im1, jp1, km1) * VAT3(oPE, iim1, jjp1, kk));

        const auto TMP3_XONW = -VAT3(oPNW, ii, jj, kk) *
                    (-VAT3(uE, im2, jp1, km1) * VAT3(dPS, iim1, jjp1, kk) -
                     VAT3(oE, im2, jp1, k) * VAT3(oPS, iim1, jjp1, kk) -
                     VAT3(uW, im1, jp1, k) * VAT3(uPS, iim1, jjp1, kk) -
                     VAT3(uSE, im2, jp2, km1) * VAT3(dPC, iim1, jjp1, kk) -
                     VAT3(oNW, im1, jp1, k) * VAT3(oPC, iim1, jjp1, kk) -
                     VAT3(uNW, im1, jp1, k) * VAT3(uPC, iim1, jjp1, kk) -
                     VAT3(uC, im1, jp1, km1) * VAT3(dPSE, iim1, jjp1, kk) +
                     VAT3(oC, im1, jp1, k) * VAT3(oPSE, iim1, jjp1, kk) -
                     VAT3(uC, im1, jp1, k) * VAT3(uPSE, iim1, jjp1, kk) -
                     VAT3(uS, im1, jp2, km1) * VAT3(dPE, iim1, jjp1, kk) -
                     VAT3(oN, im1, jp1, k) * VAT3(oPE, iim1, jjp1, kk) -
                     VAT3(uN, im1, jp1, k) * VAT3(uPE, iim1, jjp1, kk));

        const auto TMP4_XONW = -VAT3(uPNW, ii, jj, kk) *
                        (-VAT3(uE, im2, jp1, k) * VAT3(oPS, iim1, jjp1, kk) -
                         VAT3(oE, im2, jp1, kp1) * VAT3(uPS, iim1, jjp1, kk) -
                         VAT3(uSE, im2, jp2, k) * VAT3(oPC, iim1, jjp1, kk) -
                         VAT3(oNW, im1, jp1, kp1) * VAT3(uPC, iim1, jjp1, kk) -
                         VAT3(uC, im1, jp1, k) * VAT3(oPSE, iim1, jjp1, kk) +
                         VAT3(oC, im1, jp1, kp1) * VAT3(uPSE, iim1, jjp1, kk) -
                         VAT3(uS, im1, jp2, k) * VAT3(oPE, iim1, jjp1, kk) -
                         VAT3(oN, im1, jp1, kp1) * VAT3(uPE, iim1, jjp1, kk))

                    - VAT3(dPC, ii, jj, kk) *
                          (-VAT3(oNW, i, j, km1) * VAT3(dPSE, iim1, jjp1, kk) -
                           VAT3(uNW, i, j, km1) * VAT3(oPSE, iim1, jjp1, kk));

        const auto TMP5_XONW =
            -VAT3(oPC, ii, jj, kk) *
                (-VAT3(uSE, im1, jp1, km1) * VAT3(dPSE, iim1, jjp1, kk) -
                 VAT3(oNW, i, j, k) * VAT3(oPSE, iim1, jjp1, kk) -
                 VAT3(uNW, i, j, k) * VAT3(uPSE, iim1, jjp1, kk))

            - VAT3(uPC, ii, jj, kk) *
                  (-VAT3(uSE, im1, jp1, k) * VAT3(oPSE, iim1, jjp1, kk) -
                   VAT3(oNW, i, j, kp1) * VAT3(uPSE, iim1, jjp1, kk))

            - VAT3(dPN, ii, jj, kk) *
                  (-VAT3(oE, im1, jp1, km1) * VAT3(dPSE, iim1, jjp1, kk) -
                   VAT3(uW, i, jp1, km1) * VAT3(oPSE, iim1, jjp1, kk) -
                   VAT3(oNW, i, jp1, km1) * VAT3(dPE, iim1, jjp1, kk) -
                   VAT3(uNW, i, jp1, km1) * VAT3(oPE, iim1, jjp1, kk));

        const auto TMP6_XONW = -VAT3(oPN, ii, jj, kk) *
                        (-VAT3(uE, im1, jp1, km1) * VAT3(dPSE, iim1, jjp1, kk) -
                         VAT3(oE, im1, jp1, k) * VAT3(oPSE, iim1, jjp1, kk) -
                         VAT3(uW, i, jp1, k) * VAT3(uPSE, iim1, jjp1, kk) -
                         VAT3(uSE, im1, jp2, km1) * VAT3(dPE, iim1, jjp1, kk) -
                         VAT3(oNW, i, jp1, k) * VAT3(oPE, iim1, jjp1, kk) -
                         VAT3(uNW, i, jp1, k) * VAT3(uPE, iim1, jjp1, kk))

                    -
                    VAT3(uPN, ii, jj, kk) *
                        (-VAT3(uE, im1, jp1, k) * VAT3(oPSE, iim1, jjp1, kk) -
                         VAT3(oE, im1, jp1, kp1) * VAT3(uPSE, iim1, jjp1, kk) -
                         VAT3(uSE, im1, jp2, k) * VAT3(oPE, iim1, jjp1, kk) -
                         VAT3(oNW, i, jp1, kp1) * VAT3(uPE, iim1, jjp1, kk));

        VAT3(XoNW, ii, jj, kk) = TMP1_XONW + TMP2_XONW + TMP3_XONW + TMP4_XONW +
                                 TMP5_XONW + TMP6_XONW;

        // fprintf(data, "%19.12E\n", VAT3(XoNW, ii, jj, kk));

        /* *************************************************************
         * *** > UE;
         * *************************************************************/
        // VAT3( XuE, ii,jj,kk) =
        const auto TMP1_XUE = -VAT3(oPS, ii, jj, kk) *
                       (-VAT3(uE, i, jm1, k) * VAT3(dPSW, iip1, jj, kkp1) -
                        VAT3(uNE, i, jm1, k) * VAT3(dPW, iip1, jj, kkp1))

                   - VAT3(uPS, ii, jj, kk) *
                         (-VAT3(oE, i, jm1, kp1) * VAT3(dPSW, iip1, jj, kkp1) -
                          VAT3(uE, i, jm1, kp1) * VAT3(oPSW, iip1, jj, kkp1) -
                          VAT3(oNE, i, jm1, kp1) * VAT3(dPW, iip1, jj, kkp1) -
                          VAT3(uNE, i, jm1, kp1) * VAT3(oPW, iip1, jj, kkp1))

                   - VAT3(oPC, ii, jj, kk) *
                         (-VAT3(uSE, i, j, k) * VAT3(dPSW, iip1, jj, kkp1) -
                          VAT3(uE, i, j, k) * VAT3(dPW, iip1, jj, kkp1) -
                          VAT3(uNE, i, j, k) * VAT3(dPNW, iip1, jj, kkp1));

        const auto TMP2_XUE = -VAT3(uPC, ii, jj, kk) *
                       (-VAT3(oNW, ip1, jm1, kp1) * VAT3(dPSW, iip1, jj, kkp1) -
                        VAT3(uSE, i, j, kp1) * VAT3(oPSW, iip1, jj, kkp1) -
                        VAT3(oE, i, j, kp1) * VAT3(dPW, iip1, jj, kkp1) -
                        VAT3(uE, i, j, kp1) * VAT3(oPW, iip1, jj, kkp1) -
                        VAT3(oNE, i, j, kp1) * VAT3(dPNW, iip1, jj, kkp1) -
                        VAT3(uNE, i, j, kp1) * VAT3(oPNW, iip1, jj, kkp1))

                   - VAT3(oPN, ii, jj, kk) *
                         (-VAT3(uSE, i, jp1, k) * VAT3(dPW, iip1, jj, kkp1) -
                          VAT3(uE, i, jp1, k) * VAT3(dPNW, iip1, jj, kkp1))

                   - VAT3(uPN, ii, jj, kk) *
                         (-VAT3(oNW, ip1, j, kp1) * VAT3(dPW, iip1, jj, kkp1) -
                          VAT3(uSE, i, jp1, kp1) * VAT3(oPW, iip1, jj, kkp1) -
                          VAT3(oE, i, jp1, kp1) * VAT3(dPNW, iip1, jj, kkp1) -
                          VAT3(uE, i, jp1, kp1) * VAT3(oPNW, iip1, jj, kkp1));

        const auto TMP3_XUE = -VAT3(oPSE, ii, jj, kk) *
                       (-VAT3(uC, ip1, jm1, k) * VAT3(dPSW, iip1, jj, kkp1) -
                        VAT3(uN, ip1, jm1, k) * VAT3(dPW, iip1, jj, kkp1) -
                        VAT3(uE, ip1, jm1, k) * VAT3(dPS, iip1, jj, kkp1) -
                        VAT3(uNE, ip1, jm1, k) * VAT3(dPC, iip1, jj, kkp1))

                   - VAT3(uPSE, ii, jj, kk) *
                         (VAT3(oC, ip1, jm1, kp1) * VAT3(dPSW, iip1, jj, kkp1) -
                          VAT3(uC, ip1, jm1, kp1) * VAT3(oPSW, iip1, jj, kkp1) -
                          VAT3(oN, ip1, jm1, kp1) * VAT3(dPW, iip1, jj, kkp1) -
                          VAT3(uN, ip1, jm1, kp1) * VAT3(oPW, iip1, jj, kkp1) -
                          VAT3(oE, ip1, jm1, kp1) * VAT3(dPS, iip1, jj, kkp1) -
                          VAT3(uE, ip1, jm1, kp1) * VAT3(oPS, iip1, jj, kkp1) -
                          VAT3(oNE, ip1, jm1, kp1) * VAT3(dPC, iip1, jj, kkp1) -
                          VAT3(uNE, ip1, jm1, kp1) * VAT3(oPC, iip1, jj, kkp1));

        const auto TMP4_XUE = -VAT3(oPE, ii, jj, kk) *
                   (-VAT3(uS, ip1, j, k) * VAT3(dPSW, iip1, jj, kkp1) -
                    VAT3(uC, ip1, j, k) * VAT3(dPW, iip1, jj, kkp1) -
                    VAT3(uN, ip1, j, k) * VAT3(dPNW, iip1, jj, kkp1) -
                    VAT3(uSE, ip1, j, k) * VAT3(dPS, iip1, jj, kkp1) -
                    VAT3(uE, ip1, j, k) * VAT3(dPC, iip1, jj, kkp1) -
                    VAT3(uNE, ip1, j, k) * VAT3(dPN, iip1, jj, kkp1));

        const auto TMP5_XUE = -VAT3(uPE, ii, jj, kk) *
                   (-VAT3(oN, ip1, jm1, kp1) * VAT3(dPSW, iip1, jj, kkp1) -
                    VAT3(uS, ip1, j, kp1) * VAT3(oPSW, iip1, jj, kkp1) +
                    VAT3(oC, ip1, j, kp1) * VAT3(dPW, iip1, jj, kkp1) -
                    VAT3(uC, ip1, j, kp1) * VAT3(oPW, iip1, jj, kkp1) -
                    VAT3(oN, ip1, j, kp1) * VAT3(dPNW, iip1, jj, kkp1) -
                    VAT3(uN, ip1, j, kp1) * VAT3(oPNW, iip1, jj, kkp1) -
                    VAT3(oNW, ip2, jm1, kp1) * VAT3(dPS, iip1, jj, kkp1) -
                    VAT3(uSE, ip1, j, kp1) * VAT3(oPS, iip1, jj, kkp1) -
                    VAT3(oE, ip1, j, kp1) * VAT3(dPC, iip1, jj, kkp1) -
                    VAT3(uE, ip1, j, kp1) * VAT3(oPC, iip1, jj, kkp1) -
                    VAT3(oNE, ip1, j, kp1) * VAT3(dPN, iip1, jj, kkp1) -
                    VAT3(uNE, ip1, j, kp1) * VAT3(oPN, iip1, jj, kkp1));

        const auto TMP6_XUE = -VAT3(oPNE, ii, jj, kk) *
                       (-VAT3(uS, ip1, jp1, k) * VAT3(dPW, iip1, jj, kkp1) -
                        VAT3(uC, ip1, jp1, k) * VAT3(dPNW, iip1, jj, kkp1) -
                        VAT3(uSE, ip1, jp1, k) * VAT3(dPC, iip1, jj, kkp1) -
                        VAT3(uE, ip1, jp1, k) * VAT3(dPN, iip1, jj, kkp1))

                   - VAT3(uPNE, ii, jj, kk) *
                         (-VAT3(oN, ip1, j, kp1) * VAT3(dPW, iip1, jj, kkp1) -
                          VAT3(uS, ip1, jp1, kp1) * VAT3(oPW, iip1, jj, kkp1) +
                          VAT3(oC, ip1, jp1, kp1) * VAT3(dPNW, iip1, jj, kkp1) -
                          VAT3(uC, ip1, jp1, kp1) * VAT3(oPNW, iip1, jj, kkp1) -
                          VAT3(oNW, ip2, j, kp1) * VAT3(dPC, iip1, jj, kkp1) -
                          VAT3(uSE, ip1, jp1, kp1) * VAT3(oPC, iip1, jj, kkp1) -
                          VAT3(oE, ip1, jp1, kp1) * VAT3(dPN, iip1, jj, kkp1) -
                          VAT3(uE, ip1, jp1, kp1) * VAT3(oPN, iip1, jj, kkp1));

        VAT3(XuE, ii, jj, kk) =
            TMP1_XUE + TMP2_XUE + TMP3_XUE + TMP4_XUE + TMP5_XUE + TMP6_XUE;

        // fprintf(data, "%19.12E\n", VAT3(XuE, ii, jj, kk));

        /* *************************************************************
         * *** > UW;
         * *************************************************************/
        // VAT3( XuW, ii,jj,kk) =
        const auto TMP1_XUW = -VAT3(oPSW, ii, jj, kk) *
                       (-VAT3(uW, im1, jm1, k) * VAT3(dPS, iim1, jj, kkp1) -
                        VAT3(uNW, im1, jm1, k) * VAT3(dPC, iim1, jj, kkp1) -
                        VAT3(uC, im1, jm1, k) * VAT3(dPSE, iim1, jj, kkp1) -
                        VAT3(uN, im1, jm1, k) * VAT3(dPE, iim1, jj, kkp1))

                   - VAT3(uPSW, ii, jj, kk) *
                         (-VAT3(oE, im2, jm1, kp1) * VAT3(dPS, iim1, jj, kkp1) -
                          VAT3(uW, im1, jm1, kp1) * VAT3(oPS, iim1, jj, kkp1) -
                          VAT3(oNW, im1, jm1, kp1) * VAT3(dPC, iim1, jj, kkp1) -
                          VAT3(uNW, im1, jm1, kp1) * VAT3(oPC, iim1, jj, kkp1) +
                          VAT3(oC, im1, jm1, kp1) * VAT3(dPSE, iim1, jj, kkp1) -
                          VAT3(uC, im1, jm1, kp1) * VAT3(oPSE, iim1, jj, kkp1) -
                          VAT3(oN, im1, jm1, kp1) * VAT3(dPE, iim1, jj, kkp1) -
                          VAT3(uN, im1, jm1, kp1) * VAT3(oPE, iim1, jj, kkp1));

        const auto TMP2_XUW = -VAT3(oPW, ii, jj, kk) *
                   (-VAT3(uSW, im1, j, k) * VAT3(dPS, iim1, jj, kkp1) -
                    VAT3(uW, im1, j, k) * VAT3(dPC, iim1, jj, kkp1) -
                    VAT3(uNW, im1, j, k) * VAT3(dPN, iim1, jj, kkp1) -
                    VAT3(uS, im1, j, k) * VAT3(dPSE, iim1, jj, kkp1) -
                    VAT3(uC, im1, j, k) * VAT3(dPE, iim1, jj, kkp1) -
                    VAT3(uN, im1, j, k) * VAT3(dPNE, iim1, jj, kkp1));

        const auto TMP3_XUW = -VAT3(uPW, ii, jj, kk) *
                   (-VAT3(oNE, im2, jm1, kp1) * VAT3(dPS, iim1, jj, kkp1) -
                    VAT3(uSW, im1, j, kp1) * VAT3(oPS, iim1, jj, kkp1) -
                    VAT3(oE, im2, j, kp1) * VAT3(dPC, iim1, jj, kkp1) -
                    VAT3(uW, im1, j, kp1) * VAT3(oPC, iim1, jj, kkp1) -
                    VAT3(oNW, im1, j, kp1) * VAT3(dPN, iim1, jj, kkp1) -
                    VAT3(uNW, im1, j, kp1) * VAT3(oPN, iim1, jj, kkp1) -
                    VAT3(oN, im1, jm1, kp1) * VAT3(dPSE, iim1, jj, kkp1) -
                    VAT3(uS, im1, j, kp1) * VAT3(oPSE, iim1, jj, kkp1) +
                    VAT3(oC, im1, j, kp1) * VAT3(dPE, iim1, jj, kkp1) -
                    VAT3(uC, im1, j, kp1) * VAT3(oPE, iim1, jj, kkp1) -
                    VAT3(oN, im1, j, kp1) * VAT3(dPNE, iim1, jj, kkp1) -
                    VAT3(uN, im1, j, kp1) * VAT3(oPNE, iim1, jj, kkp1));

        const auto TMP4_XUW = -VAT3(oPNW, ii, jj, kk) *
                       (-VAT3(uSW, im1, jp1, k) * VAT3(dPC, iim1, jj, kkp1) -
                        VAT3(uW, im1, jp1, k) * VAT3(dPN, iim1, jj, kkp1) -
                        VAT3(uS, im1, jp1, k) * VAT3(dPE, iim1, jj, kkp1) -
                        VAT3(uC, im1, jp1, k) * VAT3(dPNE, iim1, jj, kkp1))

                   - VAT3(uPNW, ii, jj, kk) *
                         (-VAT3(oNE, im2, j, kp1) * VAT3(dPC, iim1, jj, kkp1) -
                          VAT3(uSW, im1, jp1, kp1) * VAT3(oPC, iim1, jj, kkp1) -
                          VAT3(oE, im2, jp1, kp1) * VAT3(dPN, iim1, jj, kkp1) -
                          VAT3(uW, im1, jp1, kp1) * VAT3(oPN, iim1, jj, kkp1) -
                          VAT3(oN, im1, j, kp1) * VAT3(dPE, iim1, jj, kkp1) -
                          VAT3(uS, im1, jp1, kp1) * VAT3(oPE, iim1, jj, kkp1) +
                          VAT3(oC, im1, jp1, kp1) * VAT3(dPNE, iim1, jj, kkp1) -
                          VAT3(uC, im1, jp1, kp1) * VAT3(oPNE, iim1, jj, kkp1));

        const auto TMP5_XUW = -VAT3(oPS, ii, jj, kk) *
                       (-VAT3(uW, i, jm1, k) * VAT3(dPSE, iim1, jj, kkp1) -
                        VAT3(uNW, i, jm1, k) * VAT3(dPE, iim1, jj, kkp1))

                   -
                   VAT3(uPS, ii, jj, kk) *
                       (-VAT3(oE, im1, jm1, kp1) * VAT3(dPSE, iim1, jj, kkp1) -
                        VAT3(uW, i, jm1, kp1) * VAT3(oPSE, iim1, jj, kkp1) -
                        VAT3(oNW, i, jm1, kp1) * VAT3(dPE, iim1, jj, kkp1) -
                        VAT3(uNW, i, jm1, kp1) * VAT3(oPE, iim1, jj, kkp1))

                   - VAT3(oPC, ii, jj, kk) *
                         (-VAT3(uSW, i, j, k) * VAT3(dPSE, iim1, jj, kkp1) -
                          VAT3(uW, i, j, k) * VAT3(dPE, iim1, jj, kkp1) -
                          VAT3(uNW, i, j, k) * VAT3(dPNE, iim1, jj, kkp1));

        const auto TMP6_XUW = -VAT3(uPC, ii, jj, kk) *
                       (-VAT3(oNE, im1, jm1, kp1) * VAT3(dPSE, iim1, jj, kkp1) -
                        VAT3(uSW, i, j, kp1) * VAT3(oPSE, iim1, jj, kkp1) -
                        VAT3(oE, im1, j, kp1) * VAT3(dPE, iim1, jj, kkp1) -
                        VAT3(uW, i, j, kp1) * VAT3(oPE, iim1, jj, kkp1) -
                        VAT3(oNW, i, j, kp1) * VAT3(dPNE, iim1, jj, kkp1) -
                        VAT3(uNW, i, j, kp1) * VAT3(oPNE, iim1, jj, kkp1))

                   - VAT3(oPN, ii, jj, kk) *
                         (-VAT3(uSW, i, jp1, k) * VAT3(dPE, iim1, jj, kkp1) -
                          VAT3(uW, i, jp1, k) * VAT3(dPNE, iim1, jj, kkp1))

                   - VAT3(uPN, ii, jj, kk) *
                         (-VAT3(oNE, im1, j, kp1) * VAT3(dPE, iim1, jj, kkp1) -
                          VAT3(uSW, i, jp1, kp1) * VAT3(oPE, iim1, jj, kkp1) -
                          VAT3(oE, im1, jp1, kp1) * VAT3(dPNE, iim1, jj, kkp1) -
                          VAT3(uW, i, jp1, kp1) * VAT3(oPNE, iim1, jj, kkp1));

        VAT3(XuW, ii, jj, kk) =
            TMP1_XUW + TMP2_XUW + TMP3_XUW + TMP4_XUW + TMP5_XUW + TMP6_XUW;

        // fprintf(data, "%19.12E\n", VAT3(XuW, ii, jj, kk));

        /* *************************************************************
         * *** > UN;
         * *************************************************************/
        // VAT3( XuN, ii,jj,kk) =
        const auto TMP1_XUN = -VAT3(oPW, ii, jj, kk) *
                       (-VAT3(uN, im1, j, k) * VAT3(dPSW, ii, jjp1, kkp1) -
                        VAT3(uNE, im1, j, k) * VAT3(dPS, ii, jjp1, kkp1))

                   - VAT3(uPW, ii, jj, kk) *
                         (-VAT3(oN, im1, j, kp1) * VAT3(dPSW, ii, jjp1, kkp1) -
                          VAT3(uN, im1, j, kp1) * VAT3(oPSW, ii, jjp1, kkp1) -
                          VAT3(oNE, im1, j, kp1) * VAT3(dPS, ii, jjp1, kkp1) -
                          VAT3(uNE, im1, j, kp1) * VAT3(oPS, ii, jjp1, kkp1))

                   - VAT3(oPNW, ii, jj, kk) *
                         (-VAT3(uC, im1, jp1, k) * VAT3(dPSW, ii, jjp1, kkp1) -
                          VAT3(uN, im1, jp1, k) * VAT3(dPW, ii, jjp1, kkp1) -
                          VAT3(uE, im1, jp1, k) * VAT3(dPS, ii, jjp1, kkp1) -
                          VAT3(uNE, im1, jp1, k) * VAT3(dPC, ii, jjp1, kkp1));

        const auto TMP2_XUN = -VAT3(uPNW, ii, jj, kk) *
                       (VAT3(oC, im1, jp1, kp1) * VAT3(dPSW, ii, jjp1, kkp1) -
                        VAT3(uC, im1, jp1, kp1) * VAT3(oPSW, ii, jjp1, kkp1) -
                        VAT3(oN, im1, jp1, kp1) * VAT3(dPW, ii, jjp1, kkp1) -
                        VAT3(uN, im1, jp1, kp1) * VAT3(oPW, ii, jjp1, kkp1) -
                        VAT3(oE, im1, jp1, kp1) * VAT3(dPS, ii, jjp1, kkp1) -
                        VAT3(uE, im1, jp1, kp1) * VAT3(oPS, ii, jjp1, kkp1) -
                        VAT3(oNE, im1, jp1, kp1) * VAT3(dPC, ii, jjp1, kkp1) -
                        VAT3(uNE, im1, jp1, kp1) * VAT3(oPC, ii, jjp1, kkp1))

                   - VAT3(oPC, ii, jj, kk) *
                         (-VAT3(uNW, i, j, k) * VAT3(dPSW, ii, jjp1, kkp1) -
                          VAT3(uN, i, j, k) * VAT3(dPS, ii, jjp1, kkp1) -
                          VAT3(uNE, i, j, k) * VAT3(dPSE, ii, jjp1, kkp1));

        const auto TMP3_XUN = -VAT3(uPC, ii, jj, kk) *
                       (-VAT3(oNW, i, j, kp1) * VAT3(dPSW, ii, jjp1, kkp1) -
                        VAT3(uNW, i, j, kp1) * VAT3(oPSW, ii, jjp1, kkp1) -
                        VAT3(oN, i, j, kp1) * VAT3(dPS, ii, jjp1, kkp1) -
                        VAT3(uN, i, j, kp1) * VAT3(oPS, ii, jjp1, kkp1) -
                        VAT3(oNE, i, j, kp1) * VAT3(dPSE, ii, jjp1, kkp1) -
                        VAT3(uNE, i, j, kp1) * VAT3(oPSE, ii, jjp1, kkp1))

                   - VAT3(oPN, ii, jj, kk) *
                         (-VAT3(uW, i, jp1, k) * VAT3(dPSW, ii, jjp1, kkp1) -
                          VAT3(uNW, i, jp1, k) * VAT3(dPW, ii, jjp1, kkp1) -
                          VAT3(uC, i, jp1, k) * VAT3(dPS, ii, jjp1, kkp1) -
                          VAT3(uN, i, jp1, k) * VAT3(dPC, ii, jjp1, kkp1) -
                          VAT3(uE, i, jp1, k) * VAT3(dPSE, ii, jjp1, kkp1) -
                          VAT3(uNE, i, jp1, k) * VAT3(dPE, ii, jjp1, kkp1));

        const auto TMP4_XUN = -VAT3(uPN, ii, jj, kk) *
                   (-VAT3(oE, im1, jp1, kp1) * VAT3(dPSW, ii, jjp1, kkp1) -
                    VAT3(uW, i, jp1, kp1) * VAT3(oPSW, ii, jjp1, kkp1) -
                    VAT3(oNW, i, jp1, kp1) * VAT3(dPW, ii, jjp1, kkp1) -
                    VAT3(uNW, i, jp1, kp1) * VAT3(oPW, ii, jjp1, kkp1) +
                    VAT3(oC, i, jp1, kp1) * VAT3(dPS, ii, jjp1, kkp1) -
                    VAT3(uC, i, jp1, kp1) * VAT3(oPS, ii, jjp1, kkp1) -
                    VAT3(oN, i, jp1, kp1) * VAT3(dPC, ii, jjp1, kkp1) -
                    VAT3(uN, i, jp1, kp1) * VAT3(oPC, ii, jjp1, kkp1) -
                    VAT3(oE, i, jp1, kp1) * VAT3(dPSE, ii, jjp1, kkp1) -
                    VAT3(uE, i, jp1, kp1) * VAT3(oPSE, ii, jjp1, kkp1) -
                    VAT3(oNE, i, jp1, kp1) * VAT3(dPE, ii, jjp1, kkp1) -
                    VAT3(uNE, i, jp1, kp1) * VAT3(oPE, ii, jjp1, kkp1));

        const auto TMP5_XUN = -VAT3(oPE, ii, jj, kk) *
                       (-VAT3(uNW, ip1, j, k) * VAT3(dPS, ii, jjp1, kkp1) -
                        VAT3(uN, ip1, j, k) * VAT3(dPSE, ii, jjp1, kkp1))

                   - VAT3(uPE, ii, jj, kk) *
                         (-VAT3(oNW, ip1, j, kp1) * VAT3(dPS, ii, jjp1, kkp1) -
                          VAT3(uNW, ip1, j, kp1) * VAT3(oPS, ii, jjp1, kkp1) -
                          VAT3(oN, ip1, j, kp1) * VAT3(dPSE, ii, jjp1, kkp1) -
                          VAT3(uN, ip1, j, kp1) * VAT3(oPSE, ii, jjp1, kkp1))

                   - VAT3(oPNE, ii, jj, kk) *
                         (-VAT3(uW, ip1, jp1, k) * VAT3(dPS, ii, jjp1, kkp1) -
                          VAT3(uNW, ip1, jp1, k) * VAT3(dPC, ii, jjp1, kkp1) -
                          VAT3(uC, ip1, jp1, k) * VAT3(dPSE, ii, jjp1, kkp1) -
                          VAT3(uN, ip1, jp1, k) * VAT3(dPE, ii, jjp1, kkp1));

        const auto TMP6_XUN = -VAT3(uPNE, ii, jj, kk) *
                   (-VAT3(oE, i, jp1, kp1) * VAT3(dPS, ii, jjp1, kkp1) -
                    VAT3(uW, ip1, jp1, kp1) * VAT3(oPS, ii, jjp1, kkp1) -
                    VAT3(oNW, ip1, jp1, kp1) * VAT3(dPC, ii, jjp1, kkp1) -
                    VAT3(uNW, ip1, jp1, kp1) * VAT3(oPC, ii, jjp1, kkp1) +
                    VAT3(oC, ip1, jp1, kp1) * VAT3(dPSE, ii, jjp1, kkp1) -
                    VAT3(uC, ip1, jp1, kp1) * VAT3(oPSE, ii, jjp1, kkp1) -
                    VAT3(oN, ip1, jp1, kp1) * VAT3(dPE, ii, jjp1, kkp1) -
                    VAT3(uN, ip1, jp1, kp1) * VAT3(oPE, ii, jjp1, kkp1));

        VAT3(XuN, ii, jj, kk) =
            TMP1_XUN + TMP2_XUN + TMP3_XUN + TMP4_XUN + TMP5_XUN + TMP6_XUN;

        // fprintf(data, "%19.12E\n", VAT3(XuN, ii, jj, kk));

        /* *************************************************************
         * *** > US;
         * *************************************************************/

        // VAT3( XuS, ii,jj,kk) =
        const auto TMP1_XUS = -VAT3(oPSW, ii, jj, kk) *
                       (-VAT3(uS, im1, jm1, k) * VAT3(dPW, ii, jjm1, kkp1) -
                        VAT3(uC, im1, jm1, k) * VAT3(dPNW, ii, jjm1, kkp1) -
                        VAT3(uSE, im1, jm1, k) * VAT3(dPC, ii, jjm1, kkp1) -
                        VAT3(uE, im1, jm1, k) * VAT3(dPN, ii, jjm1, kkp1))

                   - VAT3(uPSW, ii, jj, kk) *
                         (-VAT3(oN, im1, jm2, kp1) * VAT3(dPW, ii, jjm1, kkp1) -
                          VAT3(uS, im1, jm1, kp1) * VAT3(oPW, ii, jjm1, kkp1) +
                          VAT3(oC, im1, jm1, kp1) * VAT3(dPNW, ii, jjm1, kkp1) -
                          VAT3(uC, im1, jm1, kp1) * VAT3(oPNW, ii, jjm1, kkp1) -
                          VAT3(oNW, i, jm2, kp1) * VAT3(dPC, ii, jjm1, kkp1) -
                          VAT3(uSE, im1, jm1, kp1) * VAT3(oPC, ii, jjm1, kkp1) -
                          VAT3(oE, im1, jm1, kp1) * VAT3(dPN, ii, jjm1, kkp1) -
                          VAT3(uE, im1, jm1, kp1) * VAT3(oPN, ii, jjm1, kkp1));

        const auto TMP2_XUS = -VAT3(oPW, ii, jj, kk) *
                       (-VAT3(uS, im1, j, k) * VAT3(dPNW, ii, jjm1, kkp1) -
                        VAT3(uSE, im1, j, k) * VAT3(dPN, ii, jjm1, kkp1))

                   -
                   VAT3(uPW, ii, jj, kk) *
                       (-VAT3(oN, im1, jm1, kp1) * VAT3(dPNW, ii, jjm1, kkp1) -
                        VAT3(uS, im1, j, kp1) * VAT3(oPNW, ii, jjm1, kkp1) -
                        VAT3(oNW, i, jm1, kp1) * VAT3(dPN, ii, jjm1, kkp1) -
                        VAT3(uSE, im1, j, kp1) * VAT3(oPN, ii, jjm1, kkp1))

                   - VAT3(oPS, ii, jj, kk) *
                         (-VAT3(uSW, i, jm1, k) * VAT3(dPW, ii, jjm1, kkp1) -
                          VAT3(uW, i, jm1, k) * VAT3(dPNW, ii, jjm1, kkp1) -
                          VAT3(uS, i, jm1, k) * VAT3(dPC, ii, jjm1, kkp1) -
                          VAT3(uC, i, jm1, k) * VAT3(dPN, ii, jjm1, kkp1) -
                          VAT3(uSE, i, jm1, k) * VAT3(dPE, ii, jjm1, kkp1) -
                          VAT3(uE, i, jm1, k) * VAT3(dPNE, ii, jjm1, kkp1));

        const auto TMP3_XUS = -VAT3(uPS, ii, jj, kk) *
                   (-VAT3(oNE, im1, jm2, kp1) * VAT3(dPW, ii, jjm1, kkp1) -
                    VAT3(uSW, i, jm1, kp1) * VAT3(oPW, ii, jjm1, kkp1) -
                    VAT3(oE, im1, jm1, kp1) * VAT3(dPNW, ii, jjm1, kkp1) -
                    VAT3(uW, i, jm1, kp1) * VAT3(oPNW, ii, jjm1, kkp1) -
                    VAT3(oN, i, jm2, kp1) * VAT3(dPC, ii, jjm1, kkp1) -
                    VAT3(uS, i, jm1, kp1) * VAT3(oPC, ii, jjm1, kkp1) +
                    VAT3(oC, i, jm1, kp1) * VAT3(dPN, ii, jjm1, kkp1) -
                    VAT3(uC, i, jm1, kp1) * VAT3(oPN, ii, jjm1, kkp1) -
                    VAT3(oNW, ip1, jm2, kp1) * VAT3(dPE, ii, jjm1, kkp1) -
                    VAT3(uSE, i, jm1, kp1) * VAT3(oPE, ii, jjm1, kkp1) -
                    VAT3(oE, i, jm1, kp1) * VAT3(dPNE, ii, jjm1, kkp1) -
                    VAT3(uE, i, jm1, kp1) * VAT3(oPNE, ii, jjm1, kkp1));

        const auto TMP4_XUS = -VAT3(oPC, ii, jj, kk) *
                       (-VAT3(uSW, i, j, k) * VAT3(dPNW, ii, jjm1, kkp1) -
                        VAT3(uS, i, j, k) * VAT3(dPN, ii, jjm1, kkp1) -
                        VAT3(uSE, i, j, k) * VAT3(dPNE, ii, jjm1, kkp1))

                   -
                   VAT3(uPC, ii, jj, kk) *
                       (-VAT3(oNE, im1, jm1, kp1) * VAT3(dPNW, ii, jjm1, kkp1) -
                        VAT3(uSW, i, j, kp1) * VAT3(oPNW, ii, jjm1, kkp1) -
                        VAT3(oN, i, jm1, kp1) * VAT3(dPN, ii, jjm1, kkp1) -
                        VAT3(uS, i, j, kp1) * VAT3(oPN, ii, jjm1, kkp1) -
                        VAT3(oNW, ip1, jm1, kp1) * VAT3(dPNE, ii, jjm1, kkp1) -
                        VAT3(uSE, i, j, kp1) * VAT3(oPNE, ii, jjm1, kkp1));

        const auto TMP5_XUS = -VAT3(oPSE, ii, jj, kk) *
                       (-VAT3(uSW, ip1, jm1, k) * VAT3(dPC, ii, jjm1, kkp1) -
                        VAT3(uW, ip1, jm1, k) * VAT3(dPN, ii, jjm1, kkp1) -
                        VAT3(uS, ip1, jm1, k) * VAT3(dPE, ii, jjm1, kkp1) -
                        VAT3(uC, ip1, jm1, k) * VAT3(dPNE, ii, jjm1, kkp1))

                   - VAT3(uPSE, ii, jj, kk) *
                         (-VAT3(oNE, i, jm2, kp1) * VAT3(dPC, ii, jjm1, kkp1) -
                          VAT3(uSW, ip1, jm1, kp1) * VAT3(oPC, ii, jjm1, kkp1) -
                          VAT3(oE, i, jm1, kp1) * VAT3(dPN, ii, jjm1, kkp1) -
                          VAT3(uW, ip1, jm1, kp1) * VAT3(oPN, ii, jjm1, kkp1) -
                          VAT3(oN, ip1, jm2, kp1) * VAT3(dPE, ii, jjm1, kkp1) -
                          VAT3(uS, ip1, jm1, kp1) * VAT3(oPE, ii, jjm1, kkp1) +
                          VAT3(oC, ip1, jm1, kp1) * VAT3(dPNE, ii, jjm1, kkp1) -
                          VAT3(uC, ip1, jm1, kp1) * VAT3(oPNE, ii, jjm1, kkp1));

        const auto TMP6_XUS = -VAT3(oPE, ii, jj, kk) *
                       (-VAT3(uSW, ip1, j, k) * VAT3(dPN, ii, jjm1, kkp1) -
                        VAT3(uS, ip1, j, k) * VAT3(dPNE, ii, jjm1, kkp1))

                   - VAT3(uPE, ii, jj, kk) *
                         (-VAT3(oNE, i, jm1, kp1) * VAT3(dPN, ii, jjm1, kkp1) -
                          VAT3(uSW, ip1, j, kp1) * VAT3(oPN, ii, jjm1, kkp1) -
                          VAT3(oN, ip1, jm1, kp1) * VAT3(dPNE, ii, jjm1, kkp1) -
                          VAT3(uS, ip1, j, kp1) * VAT3(oPNE, ii, jjm1, kkp1));

        VAT3(XuS, ii, jj, kk) =
            TMP1_XUS + TMP2_XUS + TMP3_XUS + TMP4_XUS + TMP5_XUS + TMP6_XUS;

        // fprintf(data, "%19.12E\n", VAT3(XuS, ii, jj, kk));

        /* *************************************************************
         * *** > UNE;
         * *************************************************************/
        // VAT3(XuNE, ii,jj,kk) =
        const auto TMP1_XUNE =
            VAT3(oPC, ii, jj, kk) * VAT3(uNE, i, j, k) *
                VAT3(dPSW, iip1, jjp1, kkp1)

            - VAT3(uPC, ii, jj, kk) *
                  (-VAT3(oNE, i, j, kp1) * VAT3(dPSW, iip1, jjp1, kkp1) -
                   VAT3(uNE, i, j, kp1) * VAT3(oPSW, iip1, jjp1, kkp1))

            - VAT3(oPN, ii, jj, kk) *
                  (-VAT3(uE, i, jp1, k) * VAT3(dPSW, iip1, jjp1, kkp1) -
                   VAT3(uNE, i, jp1, k) * VAT3(dPW, iip1, jjp1, kkp1))

            - VAT3(uPN, ii, jj, kk) *
                  (-VAT3(oE, i, jp1, kp1) * VAT3(dPSW, iip1, jjp1, kkp1) -
                   VAT3(uE, i, jp1, kp1) * VAT3(oPSW, iip1, jjp1, kkp1) -
                   VAT3(oNE, i, jp1, kp1) * VAT3(dPW, iip1, jjp1, kkp1) -
                   VAT3(uNE, i, jp1, kp1) * VAT3(oPW, iip1, jjp1, kkp1))

            - VAT3(oPE, ii, jj, kk) *
                  (-VAT3(uN, ip1, j, k) * VAT3(dPSW, iip1, jjp1, kkp1) -
                   VAT3(uNE, ip1, j, k) * VAT3(dPS, iip1, jjp1, kkp1));

        const auto TMP2_XUNE =
            -VAT3(uPE, ii, jj, kk) *
                (-VAT3(oN, ip1, j, kp1) * VAT3(dPSW, iip1, jjp1, kkp1) -
                 VAT3(uN, ip1, j, kp1) * VAT3(oPSW, iip1, jjp1, kkp1) -
                 VAT3(oNE, ip1, j, kp1) * VAT3(dPS, iip1, jjp1, kkp1) -
                 VAT3(uNE, ip1, j, kp1) * VAT3(oPS, iip1, jjp1, kkp1))

            - VAT3(oPNE, ii, jj, kk) *
                  (-VAT3(uC, ip1, jp1, k) * VAT3(dPSW, iip1, jjp1, kkp1) -
                   VAT3(uN, ip1, jp1, k) * VAT3(dPW, iip1, jjp1, kkp1) -
                   VAT3(uE, ip1, jp1, k) * VAT3(dPS, iip1, jjp1, kkp1) -
                   VAT3(uNE, ip1, jp1, k) * VAT3(dPC, iip1, jjp1, kkp1))

            - VAT3(uPNE, ii, jj, kk) *
                  (VAT3(oC, ip1, jp1, kp1) * VAT3(dPSW, iip1, jjp1, kkp1) -
                   VAT3(uC, ip1, jp1, kp1) * VAT3(oPSW, iip1, jjp1, kkp1) -
                   VAT3(oN, ip1, jp1, kp1) * VAT3(dPW, iip1, jjp1, kkp1) -
                   VAT3(uN, ip1, jp1, kp1) * VAT3(oPW, iip1, jjp1, kkp1) -
                   VAT3(oE, ip1, jp1, kp1) * VAT3(dPS, iip1, jjp1, kkp1) -
                   VAT3(uE, ip1, jp1, kp1) * VAT3(oPS, iip1, jjp1, kkp1) -
                   VAT3(oNE, ip1, jp1, kp1) * VAT3(dPC, iip1, jjp1, kkp1) -
                   VAT3(uNE, ip1, jp1, kp1) * VAT3(oPC, iip1, jjp1, kkp1));
        VAT3(XuNE, ii, jj, kk) = TMP1_XUNE + TMP2_XUNE;

        // fprintf(data, "%19.12E\n", VAT3(XuNE, ii, jj, kk));

        /* *************************************************************
         * *** > UNW;
         * *************************************************************/

        // VAT3(XuNW, ii,jj,kk) =
        const auto TMP1_XUNW =
            -VAT3(oPW, ii, jj, kk) *
                (-VAT3(uNW, im1, j, k) * VAT3(dPS, iim1, jjp1, kkp1) -
                 VAT3(uN, im1, j, k) * VAT3(dPSE, iim1, jjp1, kkp1))

            - VAT3(uPW, ii, jj, kk) *
                  (-VAT3(oNW, im1, j, kp1) * VAT3(dPS, iim1, jjp1, kkp1) -
                   VAT3(uNW, im1, j, kp1) * VAT3(oPS, iim1, jjp1, kkp1) -
                   VAT3(oN, im1, j, kp1) * VAT3(dPSE, iim1, jjp1, kkp1) -
                   VAT3(uN, im1, j, kp1) * VAT3(oPSE, iim1, jjp1, kkp1))

            - VAT3(oPNW, ii, jj, kk) *
                  (-VAT3(uW, im1, jp1, k) * VAT3(dPS, iim1, jjp1, kkp1) -
                   VAT3(uNW, im1, jp1, k) * VAT3(dPC, iim1, jjp1, kkp1) -
                   VAT3(uC, im1, jp1, k) * VAT3(dPSE, iim1, jjp1, kkp1) -
                   VAT3(uN, im1, jp1, k) * VAT3(dPE, iim1, jjp1, kkp1));

        const auto TMP2_XUNW =
            -VAT3(uPNW, ii, jj, kk) *
                (-VAT3(oE, im2, jp1, kp1) * VAT3(dPS, iim1, jjp1, kkp1) -
                 VAT3(uW, im1, jp1, kp1) * VAT3(oPS, iim1, jjp1, kkp1) -
                 VAT3(oNW, im1, jp1, kp1) * VAT3(dPC, iim1, jjp1, kkp1) -
                 VAT3(uNW, im1, jp1, kp1) * VAT3(oPC, iim1, jjp1, kkp1) +
                 VAT3(oC, im1, jp1, kp1) * VAT3(dPSE, iim1, jjp1, kkp1) -
                 VAT3(uC, im1, jp1, kp1) * VAT3(oPSE, iim1, jjp1, kkp1) -
                 VAT3(oN, im1, jp1, kp1) * VAT3(dPE, iim1, jjp1, kkp1) -
                 VAT3(uN, im1, jp1, kp1) * VAT3(oPE, iim1, jjp1, kkp1))

            + VAT3(oPC, ii, jj, kk) * VAT3(uNW, i, j, k) *
                  VAT3(dPSE, iim1, jjp1, kkp1)

            - VAT3(uPC, ii, jj, kk) *
                  (-VAT3(oNW, i, j, kp1) * VAT3(dPSE, iim1, jjp1, kkp1) -
                   VAT3(uNW, i, j, kp1) * VAT3(oPSE, iim1, jjp1, kkp1))

            - VAT3(oPN, ii, jj, kk) *
                  (-VAT3(uW, i, jp1, k) * VAT3(dPSE, iim1, jjp1, kkp1) -
                   VAT3(uNW, i, jp1, k) * VAT3(dPE, iim1, jjp1, kkp1))

            - VAT3(uPN, ii, jj, kk) *
                  (-VAT3(oE, im1, jp1, kp1) * VAT3(dPSE, iim1, jjp1, kkp1) -
                   VAT3(uW, i, jp1, kp1) * VAT3(oPSE, iim1, jjp1, kkp1) -
                   VAT3(oNW, i, jp1, kp1) * VAT3(dPE, iim1, jjp1, kkp1) -
                   VAT3(uNW, i, jp1, kp1) * VAT3(oPE, iim1, jjp1, kkp1));

        VAT3(XuNW, ii, jj, kk) = TMP1_XUNW + TMP2_XUNW;

        // fprintf(data, "%19.12E\n", VAT3(XuNW, ii, jj, kk));

        /* *************************************************************
         * *** > USE;
         * *************************************************************/

        // VAT3(XuSE, ii,jj,kk) =
        const auto TMP1_XUSE =
            -VAT3(oPS, ii, jj, kk) *
                (-VAT3(uSE, i, jm1, k) * VAT3(dPW, iip1, jjm1, kkp1) -
                 VAT3(uE, i, jm1, k) * VAT3(dPNW, iip1, jjm1, kkp1))

            - VAT3(uPS, ii, jj, kk) *
                  (-VAT3(oNW, ip1, jm2, kp1) * VAT3(dPW, iip1, jjm1, kkp1) -
                   VAT3(uSE, i, jm1, kp1) * VAT3(oPW, iip1, jjm1, kkp1) -
                   VAT3(oE, i, jm1, kp1) * VAT3(dPNW, iip1, jjm1, kkp1) -
                   VAT3(uE, i, jm1, kp1) * VAT3(oPNW, iip1, jjm1, kkp1))

            + VAT3(oPC, ii, jj, kk) * VAT3(uSE, i, j, k) *
                  VAT3(dPNW, iip1, jjm1, kkp1)

            - VAT3(uPC, ii, jj, kk) *
                  (-VAT3(oNW, ip1, jm1, kp1) * VAT3(dPNW, iip1, jjm1, kkp1) -
                   VAT3(uSE, i, j, kp1) * VAT3(oPNW, iip1, jjm1, kkp1));

        const auto TMP2_XUSE =
            -VAT3(oPSE, ii, jj, kk) *
                (-VAT3(uS, ip1, jm1, k) * VAT3(dPW, iip1, jjm1, kkp1) -
                 VAT3(uC, ip1, jm1, k) * VAT3(dPNW, iip1, jjm1, kkp1) -
                 VAT3(uSE, ip1, jm1, k) * VAT3(dPC, iip1, jjm1, kkp1) -
                 VAT3(uE, ip1, jm1, k) * VAT3(dPN, iip1, jjm1, kkp1))

            - VAT3(uPSE, ii, jj, kk) *
                  (-VAT3(oN, ip1, jm2, kp1) * VAT3(dPW, iip1, jjm1, kkp1) -
                   VAT3(uS, ip1, jm1, kp1) * VAT3(oPW, iip1, jjm1, kkp1) +
                   VAT3(oC, ip1, jm1, kp1) * VAT3(dPNW, iip1, jjm1, kkp1) -
                   VAT3(uC, ip1, jm1, kp1) * VAT3(oPNW, iip1, jjm1, kkp1) -
                   VAT3(oNW, ip2, jm2, kp1) * VAT3(dPC, iip1, jjm1, kkp1) -
                   VAT3(uSE, ip1, jm1, kp1) * VAT3(oPC, iip1, jjm1, kkp1) -
                   VAT3(oE, ip1, jm1, kp1) * VAT3(dPN, iip1, jjm1, kkp1) -
                   VAT3(uE, ip1, jm1, kp1) * VAT3(oPN, iip1, jjm1, kkp1))

            - VAT3(oPE, ii, jj, kk) *
                  (-VAT3(uS, ip1, j, k) * VAT3(dPNW, iip1, jjm1, kkp1) -
                   VAT3(uSE, ip1, j, k) * VAT3(dPN, iip1, jjm1, kkp1))

            - VAT3(uPE, ii, jj, kk) *
                  (-VAT3(oN, ip1, jm1, kp1) * VAT3(dPNW, iip1, jjm1, kkp1) -
                   VAT3(uS, ip1, j, kp1) * VAT3(oPNW, iip1, jjm1, kkp1) -
                   VAT3(oNW, ip2, jm1, kp1) * VAT3(dPN, iip1, jjm1, kkp1) -
                   VAT3(uSE, ip1, j, kp1) * VAT3(oPN, iip1, jjm1, kkp1));
        VAT3(XuSE, ii, jj, kk) = TMP1_XUSE + TMP2_XUSE;

        // fprintf(data, "%19.12E\n", VAT3(XuSE, ii, jj, kk));

        /* *************************************************************
         * *** > USW;
         * *************************************************************/

        // VAT3(XuSW, ii,jj,kk) =
        const auto TMP1_XUSW =
            -VAT3(oPSW, ii, jj, kk) *
                (-VAT3(uSW, im1, jm1, k) * VAT3(dPC, iim1, jjm1, kkp1) -
                 VAT3(uW, im1, jm1, k) * VAT3(dPN, iim1, jjm1, kkp1) -
                 VAT3(uS, im1, jm1, k) * VAT3(dPE, iim1, jjm1, kkp1) -
                 VAT3(uC, im1, jm1, k) * VAT3(dPNE, iim1, jjm1, kkp1))

            - VAT3(uPSW, ii, jj, kk) *
                  (-VAT3(oNE, im2, jm2, kp1) * VAT3(dPC, iim1, jjm1, kkp1) -
                   VAT3(uSW, im1, jm1, kp1) * VAT3(oPC, iim1, jjm1, kkp1) -
                   VAT3(oE, im2, jm1, kp1) * VAT3(dPN, iim1, jjm1, kkp1) -
                   VAT3(uW, im1, jm1, kp1) * VAT3(oPN, iim1, jjm1, kkp1) -
                   VAT3(oN, im1, jm2, kp1) * VAT3(dPE, iim1, jjm1, kkp1) -
                   VAT3(uS, im1, jm1, kp1) * VAT3(oPE, iim1, jjm1, kkp1) +
                   VAT3(oC, im1, jm1, kp1) * VAT3(dPNE, iim1, jjm1, kkp1) -
                   VAT3(uC, im1, jm1, kp1) * VAT3(oPNE, iim1, jjm1, kkp1))

            - VAT3(oPW, ii, jj, kk) *
                  (-VAT3(uSW, im1, j, k) * VAT3(dPN, iim1, jjm1, kkp1) -
                   VAT3(uS, im1, j, k) * VAT3(dPNE, iim1, jjm1, kkp1));

        const auto TMP2_XUSW =
            -VAT3(uPW, ii, jj, kk) *
                (-VAT3(oNE, im2, jm1, kp1) * VAT3(dPN, iim1, jjm1, kkp1) -
                 VAT3(uSW, im1, j, kp1) * VAT3(oPN, iim1, jjm1, kkp1) -
                 VAT3(oN, im1, jm1, kp1) * VAT3(dPNE, iim1, jjm1, kkp1) -
                 VAT3(uS, im1, j, kp1) * VAT3(oPNE, iim1, jjm1, kkp1))

            - VAT3(oPS, ii, jj, kk) *
                  (-VAT3(uSW, i, jm1, k) * VAT3(dPE, iim1, jjm1, kkp1) -
                   VAT3(uW, i, jm1, k) * VAT3(dPNE, iim1, jjm1, kkp1))

            - VAT3(uPS, ii, jj, kk) *
                  (-VAT3(oNE, im1, jm2, kp1) * VAT3(dPE, iim1, jjm1, kkp1) -
                   VAT3(uSW, i, jm1, kp1) * VAT3(oPE, iim1, jjm1, kkp1) -
                   VAT3(oE, im1, jm1, kp1) * VAT3(dPNE, iim1, jjm1, kkp1) -
                   VAT3(uW, i, jm1, kp1) * VAT3(oPNE, iim1, jjm1, kkp1))

            + VAT3(oPC, ii, jj, kk) * VAT3(uSW, i, j, k) *
                  VAT3(dPNE, iim1, jjm1, kkp1)

            - VAT3(uPC, ii, jj, kk) *
                  (-VAT3(oNE, im1, jm1, kp1) * VAT3(dPNE, iim1, jjm1, kkp1) -
                   VAT3(uSW, i, j, kp1) * VAT3(oPNE, iim1, jjm1, kkp1));
        VAT3(XuSW, ii, jj, kk) = TMP1_XUSW + TMP2_XUSW;

        // fprintf(data, "%19.12E\n", VAT3(XuSW, ii, jj, kk));
      });
}
